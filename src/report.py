"""Generate the descriptive report and impulse-response plot from the saved
outputs. Purely descriptive: no quality judgements, no ranking."""

import json
import os
from collections import defaultdict

import numpy as np

BETAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def first_divergence(base_ids, other_ids):
    """Index of the first token where two continuations differ, else None."""
    for i, (a, b) in enumerate(zip(base_ids, other_ids)):
        if a != b:
            return i
    if len(base_ids) != len(other_ids):
        return min(len(base_ids), len(other_ids))
    return None


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _js_curves(locked_rows, source):
    """{condition: {step: [js values across prompts]}}"""
    curves = defaultdict(lambda: defaultdict(list))
    for r in locked_rows:
        if r["source"] != source:
            continue
        curves[r["condition"]][r["step"]].append(r["js_divergence"])
    return curves


def plot_impulse_response(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locked = _load_jsonl(os.path.join(outdir, "locked_results.jsonl"))
    if not locked:
        return
    sources = sorted({r["source"] for r in locked})
    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 5),
                             squeeze=False)
    for ax, source in zip(axes[0], sources):
        curves = _js_curves(locked, source)
        for cond in sorted(curves):
            steps = sorted(curves[cond])
            means = [float(np.mean(curves[cond][s])) for s in steps]
            ax.plot(steps, means, marker="o", markersize=3, label=cond)
        ax.set_yscale("log")
        ax.set_xlabel("tokens after injection")
        ax.set_ylabel("mean JS divergence vs hard (nats)")
        ax.set_title(f"Latent impulse response ({source} prompts)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latent_impulse_response.png"), dpi=150)
    plt.close(fig)


def _fmt(x, nd=4):
    return f"{x:.{nd}g}"


def generate(outdir):
    meta = _load_json(os.path.join(outdir, "metadata.json"), {})
    sanity = _load_json(os.path.join(outdir, "sanity.json"), [])
    controls_meta = _load_json(os.path.join(outdir, "controls_meta.json"), [])
    scan_summary = _load_json(os.path.join(outdir, "scan_summary.json"), {})
    results = _load_jsonl(os.path.join(outdir, "results.jsonl"))
    locked = _load_jsonl(os.path.join(outdir, "locked_results.jsonl"))

    lines = []
    w = lines.append
    w("# Dual-Channel Token Recurrence — V1 Report")
    w("")
    w('Working name: "Say Dog, Think Dog/Wolf". This report is descriptive '
      "only: it records what happened, with no quality judgements or "
      "ranking of continuations.")
    w("")

    # 1. Environment ----------------------------------------------------
    w("## 1. Environment")
    w("")
    for k, v in meta.items():
        w(f"- **{k}**: `{v}`")
    w("")

    # 2. Algorithm ------------------------------------------------------
    w("## 2. Exact algorithm")
    w("")
    w("For each prompt:")
    w("")
    w("1. Run the prompt with `use_cache=True`; take the next-token "
      "distribution `p = softmax(logits[:, -1, :])`.")
    w("2. The visible token is `y = argmax(p)`. **Every branch displays "
      "exactly this token.** The prompt is rerun from scratch for every "
      "branch so KV caches are fully independent.")
    w("3. Each branch feeds a different *internal* embedding at y's "
      "position via `inputs_embeds` + the prompt's `past_key_values`:")
    w("   - `hard`: `E[y]`")
    w("   - `soft_raw`: `sum_i p_i E[token_i]` over the renormalized top-k")
    w("   - `soft_normmatched`: `soft_raw` rescaled to `||E[y]||`")
    w("   - `beta_b` / `beta_b_nm`: `(1-b) E[y] + b E[z]` (z = top-2), raw "
      "and norm-matched")
    w("   - `control_B_topk2_nm`: norm-matched top1/top2 mixture at "
      f"beta = {meta.get('beta_control', 0.3)}")
    w("   - `control_C_random_orth`: `E[y]` + random perturbation "
      "orthogonal to `E[y]`, with the same L2 distance from `E[y]` as "
      "control B (its norm therefore differs slightly from `||E[y]||`; "
      "logged per prompt below)")
    w("   - `control_D_unrelated_nm`: norm-matched mixture of `E[y]` with "
      "an unrelated token at the same beta")
    w("4. **Free run** (Experiments 1–2): after the single injection, each "
      "branch returns to ordinary greedy hard-token decoding for "
      f"{meta.get('free_run_steps', 40)} steps.")
    w("5. **Locked-visible run** (Experiments 3–4): after injection, the "
      "hard branch's argmax token is fed identically to all branches for "
      f"{meta.get('locked_steps', 30)} steps, so visible text never "
      "diverges; per-step JS/KL/top-10-overlap between each branch's "
      "next-token distribution and the hard branch's are recorded.")
    w("")
    w("Decoding is greedy argmax throughout; no sampling; `model.eval()` "
      "with `torch.no_grad()`; no weights modified.")
    w("")

    # 3. Sanity check ----------------------------------------------------
    w("## 3. Sanity check: input_ids vs inputs_embeds")
    w("")
    w("Feeding token `y` through `input_ids` and feeding exactly `E[y]` "
      "through `inputs_embeds` must produce (near-)identical logits.")
    w("")
    w("| prompt | visible token | max abs logit diff | argmax equal |")
    w("|---|---|---|---|")
    for s in sanity:
        w(f"| {s['prompt'][:60]}… | `{s['visible_token']}` | "
          f"{_fmt(s['max_abs_logit_diff'])} | {s['argmax_equal']} |")
    w("")

    # 4/5. Free-run continuations -----------------------------------------
    w("## 4. Free-run continuations (all hand-written prompts, unfiltered)")
    w("")
    for rec in results:
        w(f"### Prompt: `{rec['prompt']}`")
        w("")
        w("Top-k at branch point: " + ", ".join(
            f"`{t['token']}` {t['prob']:.3f}" for t in rec["topk"]))
        w(f"Visible token in every branch: `{rec['visible_token_text']}`")
        w("")
        for name in ["hard", "soft_raw", "soft_normmatched"]:
            cont = rec["conditions"][name]["continuation"]
            w(f"- **{name}**: `{cont}`")
        w("")

        # Dose-response table (Experiment 2)
        base = rec["conditions"]["beta_0.00"]["continuation_ids"]
        w("Dose response (visible token identical for every beta):")
        w("")
        w("| beta | variant | first token differing from beta=0 | continuation |")
        w("|---|---|---|---|")
        for beta in BETAS:
            for suffix, label in [("", "raw"), ("_nm", "norm-matched")]:
                key = f"beta_{beta:.2f}{suffix}"
                c = rec["conditions"][key]
                div = first_divergence(base, c["continuation_ids"])
                div_s = "—" if div is None else str(div)
                cont = c["continuation"].replace("\n", " ").replace("|", "\\|")
                w(f"| {beta:.2f} | {label} | {div_s} | `{cont}` |")
        w("")

    # 6/7. Locked-visible impulse response --------------------------------
    w("## 5. Locked-visible impulse response")
    w("")
    w("![latent impulse response](latent_impulse_response.png)")
    w("")
    if controls_meta:
        w("Per-prompt control geometry (L2 distance of control B from "
          "`E[y]`, matched by control C; unrelated token used by control D):")
        w("")
        w("| prompt | control distance | unrelated token |")
        w("|---|---|---|")
        for c in controls_meta:
            w(f"| {c['prompt'][:60]}… | {_fmt(c['control_distance'])} | "
              f"`{c['unrelated_token']}` |")
        w("")

    for source in sorted({r["source"] for r in locked}):
        curves = _js_curves(locked, source)
        w(f"### Mean JS divergence vs hard by step ({source} prompts)")
        w("")
        steps_shown = [0, 1, 2, 5, 10, 20, 29]
        w("| condition | " + " | ".join(f"step {s}" for s in steps_shown) + " |")
        w("|---" * (len(steps_shown) + 1) + "|")
        for cond in sorted(curves):
            cells = []
            for s in steps_shown:
                vals = curves[cond].get(s, [])
                cells.append(_fmt(float(np.mean(vals))) if vals else "—")
            w(f"| {cond} | " + " | ".join(cells) + " |")
        w("")

    # 8. Aggregate scan statistics ----------------------------------------
    w("## 6. Automatic corpus scan (Phase B)")
    w("")
    if scan_summary.get("enabled"):
        if "error" in scan_summary:
            w(f"Scan skipped due to error: `{scan_summary['error']}`")
        else:
            w(f"- prefixes scanned: {scan_summary.get('scanned')}")
            w(f"- qualifying branch points (top1 < 0.85, top2 > 0.05, "
              f"top1/top2 < 10): {scan_summary.get('qualifying')}")
            w(f"- locked runs executed: {scan_summary.get('locked_runs')} "
              f"({scan_summary.get('cap_note', '')})")
            scan_rows = [r for r in locked if r["source"] == "scan"]
            if scan_rows:
                by_cond_step = defaultdict(lambda: defaultdict(list))
                for r in scan_rows:
                    by_cond_step[r["condition"]][r["step"]].append(
                        r["js_divergence"])
                w("")
                w("Aggregate JS divergence vs hard across scan branch points:")
                w("")
                w("| condition | step | mean | median | max |")
                w("|---|---|---|---|---|")
                for cond in sorted(by_cond_step):
                    for s in [0, 1, 5, 10, 20, 29]:
                        vals = by_cond_step[cond].get(s)
                        if not vals:
                            continue
                        w(f"| {cond} | {s} | {_fmt(float(np.mean(vals)))} | "
                          f"{_fmt(float(np.median(vals)))} | "
                          f"{_fmt(float(np.max(vals)))} |")
    else:
        w("Scan disabled (`--scan-n 0`).")
    w("")

    # Interpretation discipline -------------------------------------------
    w("## 7. Interpretation discipline")
    w("")
    w("A positive result establishes only that the hard visible token and "
      "the internal representation fed back at that token boundary can be "
      "separated, and that alternative internal handoffs can causally "
      "alter subsequent model behavior. It does **not** establish that "
      "soft handoffs are better, that the mixture is the true latent "
      "state, or anything about cognition.")
    w("")

    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(lines))
