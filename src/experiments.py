"""Dual-Channel Token Recurrence experiment driver.

Usage:
    python -m src.experiments --model gpt2 --all
    python -m src.experiments --model gpt2 --prompt "..." --top-k 3 --steps 40
"""

import argparse
import json
import os

import torch

from .branch import (
    beta_mixture,
    hard_embed,
    norm_match,
    random_orthogonal_perturbation,
    soft_topk_embed,
    unrelated_mixture,
)
from .generate import (
    free_run_branch,
    inject,
    locked_visible_run,
    topk_from_logits,
)
from .metrics import (
    cosine_similarity,
    js_divergence,
    kl_divergence,
    top_token_prob_diff,
    topk_overlap,
)
from .model_utils import environment_metadata, load_model, run_prompt, set_seed
from . import report as report_mod

BETAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
UNRELATED_CANDIDATES = [" piano", " banana", " umbrella", " Tuesday", " helicopter"]


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------

def sanity_check(model, tok, prompt: str) -> dict:
    """Feed y via input_ids and via inputs_embeds=E[y]; logits must match."""
    ids = tok(prompt, return_tensors="pt").input_ids
    logits, _ = run_prompt(model, ids)
    y = int(logits.argmax(dim=-1).item())

    # Path A: ordinary hard token through input_ids.
    _, cache_a = run_prompt(model, ids)
    with torch.no_grad():
        out_a = model(input_ids=torch.tensor([[y]]), past_key_values=cache_a,
                      use_cache=True)
    logits_a = out_a.logits[:, -1, :]

    # Path B: identical token through inputs_embeds.
    logits_b, _ = inject(model, ids, hard_embed(model, y))

    max_abs = float((logits_a - logits_b).abs().max().item())
    argmax_equal = bool(logits_a.argmax().item() == logits_b.argmax().item())
    return {
        "prompt": prompt,
        "visible_token": tok.decode([y]),
        "max_abs_logit_diff": max_abs,
        "argmax_equal": argmax_equal,
    }


# ----------------------------------------------------------------------
# Branch-point helpers
# ----------------------------------------------------------------------

def branch_point(model, tok, prompt: str, k: int) -> dict:
    ids = tok(prompt, return_tensors="pt").input_ids
    logits, _ = run_prompt(model, ids)
    top_ids, top_probs = topk_from_logits(logits, k)
    return {
        "prompt": prompt,
        "prompt_ids": ids,
        "top_ids": top_ids,
        "top_probs": top_probs,
        "y": int(top_ids[0].item()),
        "z": int(top_ids[1].item()),
    }


def pick_unrelated_token(tok, top_ids) -> int:
    top_set = set(int(i) for i in top_ids)
    for cand in UNRELATED_CANDIDATES:
        enc = tok.encode(cand)
        if len(enc) == 1 and enc[0] not in top_set:
            return enc[0]
    raise RuntimeError("no single-token unrelated candidate found")


def build_control_embeds(model, tok, bp, beta_control: float, seed: int) -> dict:
    """Experiment-4 control embeddings (plus hard/soft), all [1,1,hidden]."""
    y, z = bp["y"], bp["z"]
    hard = hard_embed(model, y)
    soft_raw = soft_topk_embed(model, bp["top_ids"], bp["top_probs"])
    soft_nm = norm_match(soft_raw, hard)

    ctrl_b = norm_match(beta_mixture(model, y, z, beta_control), hard)
    distance = float((ctrl_b - hard).norm().item())
    gen = torch.Generator().manual_seed(seed)
    ctrl_c = random_orthogonal_perturbation(model, y, distance, gen)
    unrelated_id = pick_unrelated_token(tok, bp["top_ids"])
    ctrl_d_raw = unrelated_mixture(model, y, unrelated_id, beta_control)
    ctrl_d_nm = norm_match(ctrl_d_raw, hard)

    return {
        "embeds": {
            "hard": hard,
            "soft_raw": soft_raw,
            "soft_normmatched": soft_nm,
            "control_B_topk2_nm": ctrl_b,
            "control_C_random_orth": ctrl_c,
            "control_D_unrelated_nm": ctrl_d_nm,
        },
        "unrelated_token": tok.decode([unrelated_id]),
        "control_distance": distance,
        "beta_control": beta_control,
    }


# ----------------------------------------------------------------------
# Experiment 1 + 2: free-running continuations (one-shot handoff + dose)
# ----------------------------------------------------------------------

def free_run_record(model, tok, bp, steps: int) -> dict:
    y, z = bp["y"], bp["z"]
    hard = hard_embed(model, y)
    conditions = {}

    def run(name, embed):
        ids = free_run_branch(model, bp["prompt_ids"], embed, steps)
        conditions[name] = {
            "continuation": tok.decode(ids),
            "continuation_ids": ids,
        }

    run("hard", hard)
    soft_raw = soft_topk_embed(model, bp["top_ids"], bp["top_probs"])
    run("soft_raw", soft_raw)
    run("soft_normmatched", norm_match(soft_raw, hard))
    for beta in BETAS:
        mix = beta_mixture(model, y, z, beta)
        run(f"beta_{beta:.2f}", mix)
        run(f"beta_{beta:.2f}_nm", norm_match(mix, hard))

    return {
        "prompt": bp["prompt"],
        "visible_token_id": y,
        "visible_token_text": tok.decode([y]),
        "top2_token_text": tok.decode([z]),
        "topk": [
            {"token": tok.decode([int(i)]), "prob": float(p)}
            for i, p in zip(bp["top_ids"], bp["top_probs"])
        ],
        "conditions": conditions,
    }


# ----------------------------------------------------------------------
# Experiment 3 + 4: locked-visible impulse response with controls
# ----------------------------------------------------------------------

def locked_records(model, tok, bp, embeds: dict, steps: int, prompt_idx,
                   source: str) -> list:
    driver_tokens, per_step = locked_visible_run(
        model, bp["prompt_ids"], embeds, steps, driver="hard")
    rows = []
    for step, probs in enumerate(per_step):
        p_hard = probs["hard"]
        hard_top = p_hard.topk(10)
        hard_top10 = [
            {"token": tok.decode([int(i)]), "prob": float(v)}
            for v, i in zip(hard_top.values, hard_top.indices)
        ]
        for name, p in probs.items():
            if name == "hard":
                continue
            cond_top = p.topk(10)
            rows.append({
                "source": source,
                "prompt_idx": prompt_idx,
                "prompt": bp["prompt"],
                "condition": name,
                "step": step,
                "driver_token": tok.decode([driver_tokens[step]]),
                "js_divergence": js_divergence(p_hard, p),
                "kl_hard_soft": kl_divergence(p_hard, p),
                "cosine_prob": cosine_similarity(p_hard, p),
                "top10_overlap": topk_overlap(p_hard, p, 10),
                "top_token_prob_diff": top_token_prob_diff(p_hard, p),
                "hard_top10": hard_top10,
                "soft_top10": [
                    {"token": tok.decode([int(i)]), "prob": float(v)}
                    for v, i in zip(cond_top.values, cond_top.indices)
                ],
            })
    return rows


# ----------------------------------------------------------------------
# Phase B: automatic corpus scan
# ----------------------------------------------------------------------

def scan_corpus(model, tok, n_prefixes: int, prefix_len: int = 32,
                top1_max: float = 0.85, top2_min: float = 0.05,
                margin_max: float = 10.0):
    """Scan WikiText-2 prefixes for qualifying branch points.
    Returns (all_scanned, qualifying) lists of dicts. No cherry-picking:
    every scanned prefix is recorded with its stats."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    scanned, qualifying = [], []
    for text in ds["text"]:
        if len(scanned) >= n_prefixes:
            break
        text = text.strip()
        if len(text) < 100 or text.startswith("="):
            continue
        ids = tok(text, return_tensors="pt").input_ids[:, :prefix_len]
        if ids.shape[1] < prefix_len:
            continue
        logits, _ = run_prompt(model, ids)
        top_ids, top_probs = topk_from_logits(logits, 5)
        p1, p2 = float(top_probs[0]), float(top_probs[1])
        rec = {
            "prefix": tok.decode(ids[0]),
            "top1_prob": p1,
            "top2_prob": p2,
            "margin": p1 / max(p2, 1e-12),
            "top1_token": tok.decode([int(top_ids[0])]),
            "top2_token": tok.decode([int(top_ids[1])]),
            "qualifies": bool(p1 < top1_max and p2 > top2_min
                              and p1 / max(p2, 1e-12) < margin_max),
        }
        scanned.append(rec)
        if rec["qualifies"]:
            qualifying.append(rec)
    return scanned, qualifying


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def load_prompts(path: str = "prompts.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_single_prompt(args, model, tok):
    bp = branch_point(model, tok, args.prompt, args.top_k)
    rec = free_run_record(model, tok, bp, args.steps)
    print(f"\nPROMPT: {rec['prompt']}")
    print("TOP-K AT BRANCH POINT:")
    for t in rec["topk"]:
        print(f"    {t['token']!r:20s} {t['prob']:.4f}")
    print(f"VISIBLE TOKEN (all branches): {rec['visible_token_text']!r}\n")
    for name in ["hard", "soft_raw", "soft_normmatched"]:
        print(f"[{name}]\n{rec['prompt']}{rec['visible_token_text']}"
              f"{rec['conditions'][name]['continuation']}\n")
    print("DOSE RESPONSE (beta mixtures of top-1/top-2, visible token fixed):")
    base = rec["conditions"]["beta_0.00"]["continuation_ids"]
    for beta in BETAS:
        for suffix in ["", "_nm"]:
            key = f"beta_{beta:.2f}{suffix}"
            ids = rec["conditions"][key]["continuation_ids"]
            div = report_mod.first_divergence(base, ids)
            print(f"    {key:14s} first_diff={str(div):>4s}  "
                  f"{tok.decode(ids)!r}")


def run_all(args, model, tok):
    os.makedirs(args.outdir, exist_ok=True)
    meta = environment_metadata(args.model, args.seed)
    meta.update({
        "top_k": args.top_k,
        "free_run_steps": args.steps,
        "locked_steps": args.locked_steps,
        "beta_control": args.beta_control,
        "betas": BETAS,
    })
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Sanity check -------------------------------------------------
    prompts = load_prompts(args.prompts)
    sanity = [sanity_check(model, tok, p) for p in prompts[:3]]
    with open(os.path.join(args.outdir, "sanity.json"), "w") as f:
        json.dump(sanity, f, indent=2)
    worst = max(s["max_abs_logit_diff"] for s in sanity)
    print(f"[sanity] max |logit diff| input_ids vs inputs_embeds: {worst:.3e}")
    assert worst < 1e-3, "sanity check failed: embeds path != input_ids path"

    # --- Experiments 1+2: free-running continuations ------------------
    results_path = os.path.join(args.outdir, "results.jsonl")
    with open(results_path, "w") as f:
        for i, prompt in enumerate(prompts):
            bp = branch_point(model, tok, prompt, args.top_k)
            rec = free_run_record(model, tok, bp, args.steps)
            f.write(json.dumps(rec) + "\n")
            print(f"[free-run] {i+1}/{len(prompts)}: {prompt[:50]!r}")

    # --- Experiments 3+4: locked-visible impulse response -------------
    locked_path = os.path.join(args.outdir, "locked_results.jsonl")
    controls_meta = []
    with open(locked_path, "w") as f:
        for i, prompt in enumerate(prompts):
            bp = branch_point(model, tok, prompt, args.top_k)
            ctrl = build_control_embeds(model, tok, bp, args.beta_control,
                                        args.seed)
            controls_meta.append({
                "prompt": prompt,
                "unrelated_token": ctrl["unrelated_token"],
                "control_distance": ctrl["control_distance"],
            })
            rows = locked_records(model, tok, bp, ctrl["embeds"],
                                  args.locked_steps, i, source="hand")
            for r in rows:
                f.write(json.dumps(r) + "\n")
            print(f"[locked] {i+1}/{len(prompts)}: {prompt[:50]!r}")
    with open(os.path.join(args.outdir, "controls_meta.json"), "w") as f:
        json.dump(controls_meta, f, indent=2)

    # --- Phase B: automatic scan --------------------------------------
    scan_summary = {"enabled": args.scan_n > 0}
    if args.scan_n > 0:
        try:
            scanned, qualifying = scan_corpus(model, tok, args.scan_n)
            with open(os.path.join(args.outdir, "scan_branchpoints.jsonl"),
                      "w") as f:
                for r in scanned:
                    f.write(json.dumps(r) + "\n")
            run_set = qualifying[:args.scan_max_runs]
            scan_summary.update({
                "scanned": len(scanned),
                "qualifying": len(qualifying),
                "locked_runs": len(run_set),
                "cap_note": (f"locked runs capped at first {args.scan_max_runs} "
                             "qualifying prefixes in corpus order (no selection "
                             "by results)"),
            })
            with open(locked_path, "a") as f:
                for j, rec in enumerate(run_set):
                    bp = branch_point(model, tok, rec["prefix"], args.top_k)
                    ctrl = build_control_embeds(model, tok, bp,
                                                args.beta_control, args.seed)
                    embeds = {k: ctrl["embeds"][k] for k in
                              ["hard", "soft_normmatched",
                               "control_C_random_orth"]}
                    rows = locked_records(model, tok, bp, embeds,
                                          args.locked_steps, j, source="scan")
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                    print(f"[scan-locked] {j+1}/{len(run_set)}")
        except Exception as e:  # corpus download can fail offline
            scan_summary.update({"error": f"{type(e).__name__}: {e}"})
            print(f"[scan] SKIPPED: {e}")
    with open(os.path.join(args.outdir, "scan_summary.json"), "w") as f:
        json.dump(scan_summary, f, indent=2)

    # --- Plot + report -------------------------------------------------
    report_mod.plot_impulse_response(args.outdir)
    report_mod.generate(args.outdir)
    print(f"[done] outputs in {args.outdir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40,
                    help="free-run continuation length")
    ap.add_argument("--locked-steps", type=int, default=30)
    ap.add_argument("--beta-control", type=float, default=0.30,
                    help="beta used for experiment-4 controls")
    ap.add_argument("--prompt", default=None,
                    help="run a single prompt interactively")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--all", action="store_true",
                    help="run the full experiment suite")
    ap.add_argument("--scan-n", type=int, default=300,
                    help="number of WikiText-2 prefixes to scan (0 = skip)")
    ap.add_argument("--scan-max-runs", type=int, default=40,
                    help="max qualifying scan prefixes given locked runs")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    model, tok = load_model(args.model)

    if args.prompt:
        run_single_prompt(args, model, tok)
    elif args.all:
        run_all(args, model, tok)
    else:
        ap.error("specify --all or --prompt")


if __name__ == "__main__":
    main()
