"""beta = 0 mixture must reproduce the hard branch exactly: the dose
response has a well-defined zero point."""

import torch

from src.branch import beta_mixture, hard_embed, norm_match
from src.generate import free_run_branch
from src.model_utils import run_prompt

PROMPT = "The chef sliced the vegetables with a sharp"


def test_beta_zero_equals_hard(model_tok):
    model, tok = model_tok
    ids = tok(PROMPT, return_tensors="pt").input_ids
    logits, _ = run_prompt(model, ids)
    y = int(logits.argmax().item())
    top2 = int(logits.topk(2).indices[0, 1].item())

    hard = hard_embed(model, y)
    mix0 = beta_mixture(model, y, top2, 0.0)
    assert torch.equal(hard, mix0)
    assert torch.allclose(norm_match(mix0, hard), hard)

    cont_hard = free_run_branch(model, ids, hard, 20)
    cont_beta0 = free_run_branch(model, ids, mix0, 20)
    assert cont_hard == cont_beta0
