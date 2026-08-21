"""Acceptance criterion 1: feeding token y through input_ids and feeding
exactly E[y] through inputs_embeds must produce matching logits."""

import torch

from src.branch import hard_embed
from src.experiments import sanity_check
from src.generate import inject
from src.model_utils import run_prompt

PROMPTS = [
    "The animal slept beside the fireplace, barked at strangers, and followed its owner everywhere. It was a",
    "The capital of Australia is",
    "Water is made of hydrogen and",
]


def test_input_ids_vs_inputs_embeds(model_tok):
    model, tok = model_tok
    for prompt in PROMPTS:
        s = sanity_check(model, tok, prompt)
        assert s["max_abs_logit_diff"] < 1e-3, s
        assert s["argmax_equal"], s


def test_full_greedy_path_matches(model_tok):
    """A whole greedy continuation through the injection machinery must
    match ordinary input_ids-only greedy decoding."""
    model, tok = model_tok
    prompt = PROMPTS[0]
    ids = tok(prompt, return_tensors="pt").input_ids
    logits, _ = run_prompt(model, ids)
    y = int(logits.argmax().item())

    # Reference: pure input_ids greedy loop.
    ref = []
    cur = torch.cat([ids, torch.tensor([[y]])], dim=1)
    for _ in range(10):
        with torch.no_grad():
            out = model(input_ids=cur)
        nxt = int(out.logits[:, -1, :].argmax().item())
        ref.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

    # Experimental machinery with the hard embedding.
    from src.generate import free_run_branch
    got = free_run_branch(model, ids, hard_embed(model, y), 10)
    assert got == ref
