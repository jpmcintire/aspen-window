"""A second model instance as a relevance monitor for the narration
channel.

The narrated-generation experiment showed that annotating every
uncertain step buries the one consequential annotation under dozens of
syntactic ones: entropy is not relevance. Here the monitor defines
relevance CAUSALLY: at each step of an ordinary greedy generation it
runs a k-step counterfactual lookahead -- inject the soft top-k
mixture at this position, drive both branches with the tokens the
generation actually produced, and score the step by the maximum JS
divergence the soft handoff would create over the lookahead horizon.

    relevance(t) = max_{j<=k} JS( soft branch dist at t+j ,
                                  actual (hard) dist at t+j )

Steps are then annotated under two budget-matched filters (top-N per
prompt): by entropy (baseline) and by consequence (the monitor), so
the two selection policies can be compared position by position.
The monitor is deterministic and judges nothing about quality; it
only predicts consequence. Descriptive only.

Run:  python -m src.relevance_monitor
Writes outputs/relevance_monitor.jsonl and outputs/monitored_transcripts.txt.
"""

import json
import os

import torch
import torch.nn.functional as F

from .branch import hard_embed, norm_match, soft_topk_embed
from .metrics import js_divergence
from .model_utils import load_model, run_prompt, set_seed

GATE_TOP1 = 0.85   # skip lookahead where the model is near-certain
LOOKAHEAD = 5
BUDGET = 5         # annotations per prompt for each filter


def generate_hard(model, tok, prompt_ids, steps):
    """Ordinary greedy generation, saving per-step full distributions."""
    logits, cache = run_prompt(model, prompt_ids)
    ids, dists = [], []
    for _ in range(steps):
        probs = F.softmax(logits.float(), dim=-1)[0]
        y = int(torch.argmax(probs))
        ids.append(y)
        dists.append(probs)
        with torch.no_grad():
            o = model(input_ids=torch.tensor([[y]]),
                      past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        logits = o.logits[:, -1, :]
    return ids, dists


def lookahead_score(model, prefix_ids, step_probs, driver_ids, top_k):
    """Counterfactual: soft handoff at this position, then the actual
    future tokens as drivers; score = max JS vs the actual dists."""
    tp, ti = torch.topk(step_probs, top_k)
    hard = hard_embed(model, int(ti[0]))
    soft = norm_match(soft_topk_embed(model, ti, tp), hard)
    with torch.no_grad():
        o = model(input_ids=prefix_ids, use_cache=True)
        cache = o.past_key_values
        o = model(inputs_embeds=soft, past_key_values=cache, use_cache=True)
    js_max = 0.0
    for j, (drv, actual) in enumerate(driver_ids):
        probs = F.softmax(o.logits[:, -1, :].float(), dim=-1)[0]
        js_max = max(js_max, js_divergence(probs, actual))
        with torch.no_grad():
            o = model(input_ids=torch.tensor([[drv]]),
                      past_key_values=o.past_key_values, use_cache=True)
    return js_max


def annotate(tok, ids, dists, chosen, top_k):
    parts = []
    for t, (y, probs) in enumerate(zip(ids, dists)):
        text = tok.decode([y])
        if t in chosen:
            tp, ti = torch.topk(probs, top_k)
            alts = ", ".join(
                f"{tok.decode([int(i)]).strip()} {round(float(p)*20)*5}%"
                for i, p in zip(ti[1:3], tp[1:3]))
            text += f"<or: {alts}>"
        parts.append(text)
    return "".join(parts)


def main(model_name="gpt2", seed=0, top_k=5, steps=60,
         prompts_path="prompts.txt", outdir="outputs"):
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)
    model, tok = load_model(model_name)
    prompts = [l.strip() for l in open(prompts_path) if l.strip()]

    jpath = os.path.join(outdir, "relevance_monitor.jsonl")
    tpath = os.path.join(outdir, "monitored_transcripts.txt")
    jf, tf = open(jpath, "w"), open(tpath, "w")
    for pi, prompt in enumerate(prompts):
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        ids, dists = generate_hard(model, tok, prompt_ids, steps)
        rows = []
        for t in range(steps - LOOKAHEAD):
            probs = dists[t]
            top1 = float(probs.max())
            ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            score = None
            if top1 < GATE_TOP1:
                prefix = torch.cat(
                    [prompt_ids, torch.tensor([ids[:t]], dtype=torch.long)],
                    dim=1) if t else prompt_ids
                drivers = [(ids[t + 1 + j], dists[t + 1 + j])
                           for j in range(LOOKAHEAD)]
                score = lookahead_score(model, prefix, probs, drivers, top_k)
            rows.append({"step": t, "token": tok.decode([ids[t]]),
                         "top1": top1, "entropy": ent,
                         "consequence": score})
        scored = [r for r in rows if r["consequence"] is not None]
        by_ent = sorted(rows, key=lambda r: -r["entropy"])[:BUDGET]
        by_con = sorted(scored, key=lambda r: -r["consequence"])[:BUDGET]
        ent_steps = {r["step"] for r in by_ent}
        con_steps = {r["step"] for r in by_con}
        rec = {"prompt": prompt,
               "overlap": len(ent_steps & con_steps),
               "entropy_picks": [(r["step"], r["token"], round(r["entropy"], 2))
                                 for r in by_ent],
               "consequence_picks": [(r["step"], r["token"],
                                      round(r["consequence"], 4))
                                     for r in by_con],
               "steps": rows}
        jf.write(json.dumps(rec) + "\n")
        tf.write(f"PROMPT: {prompt}\n\n[entropy filter, {BUDGET} annotations]\n"
                 f"{prompt}{annotate(tok, ids, dists, ent_steps, top_k)}\n\n"
                 f"[consequence monitor, {BUDGET} annotations]\n"
                 f"{prompt}{annotate(tok, ids, dists, con_steps, top_k)}\n\n"
                 + "=" * 72 + "\n\n")
        print(f"[monitor] {pi+1}/{len(prompts)}", flush=True)
    jf.close(); tf.close()
    print(f"[monitor] wrote {jpath} and {tpath}")


if __name__ == "__main__":
    main()
