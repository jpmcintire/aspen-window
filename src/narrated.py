"""Dual-channel narrated generation: soft recurrence inside, narration
outside.

At each branch point, a template 'reporter' renders the model's
pre-token uncertainty as a human-readable phrase, e.g.:

    oxygen <or: helium 28%, carbon 3%>

Part 1 (transcripts): free-run generation with a blend-0.5 inner
channel, annotating uncertain steps, so the visible text carries the
second thoughts in public instead of in hidden state.

Part 2 (round-trip fidelity): serialize the branch-point state into a
phrase at several verbosity levels, PARSE THE ACTUAL STRING back,
rebuild the soft mixture from only what the phrase says, re-inject it,
and measure over 30 locked steps how much of the true soft branch's
causal effect the rebuilt branch reproduces:

    fidelity = 1 - mean JS(rebuilt, true_soft) / mean JS(hard, true_soft)

1.0 = the phrase carries the full state-channel payload; 0.0 = no
better than the hard token. The reporter is a deterministic template,
not a second model: V1 tests the channel's bandwidth, not a
narrator's intelligence. Descriptive only.

Run:  python -m src.narrated
Writes outputs/narrated_transcripts.txt and outputs/narration_fidelity.jsonl.
"""

import json
import os
import re

import torch
import torch.nn.functional as F

from .branch import hard_embed, norm_match, soft_topk_embed
from .experiments import branch_point
from .generate import locked_visible_run
from .metrics import js_divergence
from .model_utils import load_model, run_prompt, set_seed

ANNOTATE_BELOW_TOP1 = 0.60   # annotate steps where top-1 prob < this


# ---------------------------------------------------------------- reporter

def render_phrase(tok, top_ids, top_probs, style):
    """Serialize the branch-point uncertainty into the visible phrase."""
    alts = [(tok.decode([int(i)]).strip(), float(p))
            for i, p in zip(top_ids[1:], top_probs[1:])]
    if style == "full":
        body = ", ".join(f"{t} {p:.6f}" for t, p in alts)
    elif style == "coarse":
        body = ", ".join(f"{t} {round(p*20)*5}%" for t, p in alts)
    elif style == "names":
        body = ", ".join(t for t, _ in alts)
    elif style == "top2":
        body = alts[0][0]
    else:
        raise ValueError(style)
    return f"<or: {body}>"


def parse_phrase(tok, phrase, style, y_id, y_prob):
    """Rebuild (ids, probs) from ONLY the phrase text plus the visible
    token. Alternatives whose surface text does not re-encode to a
    single token (with a leading space) are dropped -- genuine
    serialization loss, counted by the caller."""
    body = re.match(r"<or: (.*)>$", phrase).group(1)
    ids, probs, dropped = [y_id], [float(y_prob)], 0

    def reencode(text):
        enc = tok.encode(" " + text)
        return enc[0] if len(enc) == 1 else None

    if style in ("full", "coarse"):
        for part in body.split(", "):
            text, val = part.rsplit(" ", 1)
            p = float(val[:-1]) / 100 if val.endswith("%") else float(val)
            i = reencode(text)
            if i is None or p <= 0:
                dropped += 1
                continue
            ids.append(i)
            probs.append(p)
    elif style == "names":
        names = body.split(", ")
        residual = 1.0 - float(y_prob)
        weights = [1.0 / (r + 1) for r in range(len(names))]  # Zipf guess
        wsum = sum(weights)
        for r, text in enumerate(names):
            i = reencode(text)
            if i is None:
                dropped += 1
                continue
            ids.append(i)
            probs.append(residual * weights[r] / wsum)
    elif style == "top2":
        i = reencode(body)
        if i is None:
            dropped += 1
        else:
            ids.append(i)
            probs.append(0.3 / 0.7 * float(y_prob))  # assume beta=0.3 shape
    t = torch.tensor(probs)
    return torch.tensor(ids), t / t.sum(), dropped


# ------------------------------------------------------- part 1: transcripts

def narrated_transcript(model, tok, prompt, steps, top_k):
    ids = tok(prompt, return_tensors="pt").input_ids
    logits, cache = run_prompt(model, ids)
    out_text = []
    n_annotated = 0
    for _ in range(steps):
        probs = F.softmax(logits.float(), dim=-1)[0]
        tp, ti = torch.topk(probs, top_k)
        y = int(ti[0])
        visible = tok.decode([y])
        if float(tp[0]) < ANNOTATE_BELOW_TOP1:
            visible += render_phrase(tok, ti, tp, "coarse")
            n_annotated += 1
        out_text.append(visible)
        h = hard_embed(model, y)
        soft = soft_topk_embed(model, ti, tp)
        inp = norm_match(0.5 * h + 0.5 * soft, h)
        with torch.no_grad():
            o = model(inputs_embeds=inp, past_key_values=cache,
                      use_cache=True)
        cache = o.past_key_values
        logits = o.logits[:, -1, :]
    return "".join(out_text), n_annotated


# -------------------------------------------------- part 2: round-trip test

def roundtrip_record(model, tok, bp, steps):
    y, tp, ti = bp["y"], bp["top_probs"], bp["top_ids"]
    hard = hard_embed(model, y)
    true_soft = norm_match(soft_topk_embed(model, ti, tp), hard)
    embeds = {"hard": hard, "true_soft": true_soft}
    phrases, drops = {}, {}
    for style in ["full", "coarse", "names", "top2"]:
        phrase = render_phrase(tok, ti, tp, style)
        r_ids, r_probs, dropped = parse_phrase(tok, phrase, style, y, tp[0])
        embeds[f"rebuilt_{style}"] = norm_match(
            soft_topk_embed(model, r_ids, r_probs), hard)
        phrases[style] = phrase
        drops[style] = dropped
    _, per_step = locked_visible_run(model, bp["prompt_ids"], embeds, steps)
    ref = [js_divergence(s["hard"], s["true_soft"]) for s in per_step]
    rec = {"prompt": bp["prompt"], "phrases": phrases,
           "dropped_alternatives": drops,
           "mean_js_hard_vs_true": sum(ref) / len(ref), "conditions": {}}
    for style in ["full", "coarse", "names", "top2"]:
        js = [js_divergence(s[f"rebuilt_{style}"], s["true_soft"])
              for s in per_step]
        rec["conditions"][style] = {
            "mean_js_rebuilt_vs_true": sum(js) / len(js),
            "fidelity": 1.0 - (sum(js) / len(js)) / (sum(ref) / len(ref)),
        }
    return rec


def main(model_name="gpt2", seed=0, top_k=5, steps=30, free_steps=60,
         prompts_path="prompts.txt", outdir="outputs"):
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)
    model, tok = load_model(model_name)
    prompts = [l.strip() for l in open(prompts_path) if l.strip()]

    tpath = os.path.join(outdir, "narrated_transcripts.txt")
    with open(tpath, "w") as f:
        for i, p in enumerate(prompts):
            text, n = narrated_transcript(model, tok, p, free_steps, top_k)
            f.write(f"PROMPT: {p}\n[{n} annotated steps of {free_steps}]\n"
                    f"{p}{text}\n\n{'='*72}\n\n")
            print(f"[narrate] {i+1}/{len(prompts)}", flush=True)

    fpath = os.path.join(outdir, "narration_fidelity.jsonl")
    with open(fpath, "w") as f:
        for i, p in enumerate(prompts):
            bp = branch_point(model, tok, p, top_k)
            rec = roundtrip_record(model, tok, bp, steps)
            f.write(json.dumps(rec) + "\n")
            print(f"[roundtrip] {i+1}/{len(prompts)}", flush=True)
    print(f"[narrated] wrote {tpath} and {fpath}")


if __name__ == "__main__":
    main()
