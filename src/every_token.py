"""Every-token soft recurrence: instead of one injection, feed the
norm-matched top-k mixture at EVERY step and let uncertainty compound.

Conditions:
  hard        - ordinary greedy recurrence (baseline)
  soft_every  - internal input is always the norm-matched top-k mixture
  blend_0.5   - internal input is norm-matched 0.5*E[y] + 0.5*mixture
Visible token is always the argmax of the current distribution.

Per step we record entropy of the next-token distribution (the
"tension"), the L2 displacement of the internal input from E[y], and
the visible token, to see whether tension builds, collapses, or
runs away. Descriptive only.

Run:  python -m src.every_token
Writes outputs/every_token.jsonl.
"""

import json
import os

import torch
import torch.nn.functional as F

from .branch import hard_embed, norm_match, soft_topk_embed
from .model_utils import load_model, run_prompt, set_seed


def run_condition(model, tok, prompt_ids, steps, top_k, mode):
    logits, cache = run_prompt(model, prompt_ids)
    rows = []
    for step in range(steps):
        probs = F.softmax(logits.float(), dim=-1)[0]
        y = int(torch.argmax(probs))
        h = hard_embed(model, y)
        if mode == "hard":
            inp = h
        else:
            tp, ti = torch.topk(probs, top_k)
            soft = soft_topk_embed(model, ti, tp)
            if mode == "blend_0.5":
                soft = 0.5 * h + 0.5 * soft
            inp = norm_match(soft, h)
        rows.append({
            "step": step,
            "visible_token": tok.decode([y]),
            "entropy": float(-(probs * torch.log(probs.clamp_min(1e-12))).sum()),
            "top1_prob": float(probs[y]),
            "input_displacement": float((inp - h).norm()),
        })
        with torch.no_grad():
            out = model(inputs_embeds=inp, past_key_values=cache,
                        use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
    return rows


def main(model_name="gpt2", seed=0, top_k=5, steps=40,
         prompts_path="prompts.txt", outdir="outputs"):
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)
    model, tok = load_model(model_name)
    prompts = [l.strip() for l in open(prompts_path) if l.strip()]
    path = os.path.join(outdir, "every_token.jsonl")
    with open(path, "w") as f:
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids
            rec = {"prompt": p, "conditions": {}}
            for mode in ["hard", "soft_every", "blend_0.5"]:
                rows = run_condition(model, tok, ids, steps, top_k, mode)
                rec["conditions"][mode] = {
                    "continuation": "".join(r["visible_token"] for r in rows),
                    "per_step": rows,
                }
            f.write(json.dumps(rec) + "\n")
            print(f"[every-token] {i+1}/{len(prompts)}", flush=True)
    print(f"[every-token] wrote {path}")


if __name__ == "__main__":
    main()
