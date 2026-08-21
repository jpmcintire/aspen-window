"""Content-vs-magnitude analysis: does the injected direction transmit
*which* alternative was suppressed, beyond generically perturbing the
state?

For each hand prompt, track the exact probability of the branch-point
top-2 token z at every locked-visible step (full distributions, no
top-10 truncation), for the soft mixture and the dose-matched controls.

Run:  python -m src.z_tracking
Writes outputs/z_tracking.json.
"""

import json
import os

import numpy as np
import torch

from .experiments import branch_point, build_control_embeds
from .generate import locked_visible_run
from .model_utils import load_model, set_seed

CONDS = ["soft_normmatched", "control_B_topk2_nm",
         "control_D_unrelated_dm", "control_C_random_orth"]


def main(model_name="gpt2", seed=0, top_k=5, beta_control=0.30, steps=30,
         prompts_path="prompts.txt", outdir="outputs"):
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)
    model, tok = load_model(model_name)
    prompts = [l.strip() for l in open(prompts_path) if l.strip()]

    per_prompt = []
    for i, p in enumerate(prompts):
        bp = branch_point(model, tok, p, top_k)
        z = bp["z"]
        ctrl = build_control_embeds(model, tok, bp, beta_control, seed)
        embeds = {k: ctrl["embeds"][k] for k in ["hard"] + CONDS}
        _, per_step = locked_visible_run(model, bp["prompt_ids"], embeds,
                                         steps)
        rec = {
            "prompt": p,
            "z_token": tok.decode([z]),
            "p_z": {name: [float(s[name][z]) for s in per_step]
                    for name in ["hard"] + CONDS},
        }
        per_prompt.append(rec)
        print(f"[z-track] {i+1}/{len(prompts)}", flush=True)

    summary = {}
    tot_hard = sum(sum(r["p_z"]["hard"]) for r in per_prompt)
    for c in CONDS:
        tot = sum(sum(r["p_z"][c]) for r in per_prompt)
        n_up = sum(1 for r in per_prompt
                   if sum(r["p_z"][c]) > sum(r["p_z"]["hard"]) + 1e-9)
        summary[c] = {
            "cumulative_z_mass": tot,
            "delta_vs_hard": tot - tot_hard,
            "prompts_with_z_mass_up": n_up,
        }
        m = np.mean([np.array(r["p_z"][c]) - np.array(r["p_z"]["hard"])
                     for r in per_prompt], axis=0)
        summary[c]["mean_excess_by_step"] = [float(x) for x in m]
    summary["hard_cumulative_z_mass"] = tot_hard

    out = {"model": model_name, "seed": seed, "top_k": top_k,
           "beta_control": beta_control, "steps": steps,
           "summary": summary, "per_prompt": per_prompt}
    path = os.path.join(outdir, "z_tracking.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[z-track] wrote {path}")


if __name__ == "__main__":
    main()
