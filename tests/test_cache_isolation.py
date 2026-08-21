"""Acceptance criterion 3: hard and soft branches maintain genuinely
independent KV caches — running one branch must not perturb another."""

import torch

from src.branch import hard_embed, norm_match, soft_topk_embed
from src.generate import free_run_branch, inject, topk_from_logits
from src.model_utils import run_prompt

PROMPT = "Although it resembled a wolf, genetic testing showed that the domesticated animal was a"


def test_branches_do_not_interfere(model_tok):
    model, tok = model_tok
    ids = tok(PROMPT, return_tensors="pt").input_ids
    logits, _ = run_prompt(model, ids)
    top_ids, top_probs = topk_from_logits(logits, 5)
    y = int(top_ids[0].item())

    hard = hard_embed(model, y)
    soft = norm_match(soft_topk_embed(model, top_ids, top_probs), hard)

    # Hard alone.
    hard_a = free_run_branch(model, ids, hard, 15)
    # Interleave a soft branch, then rerun hard.
    _ = free_run_branch(model, ids, soft, 15)
    hard_b = free_run_branch(model, ids, hard, 15)
    assert hard_a == hard_b, "soft branch perturbed the hard branch's cache"


def test_injection_states_differ_but_caches_are_separate(model_tok):
    model, tok = model_tok
    ids = tok(PROMPT, return_tensors="pt").input_ids
    logits, _ = run_prompt(model, ids)
    top_ids, top_probs = topk_from_logits(logits, 5)
    y = int(top_ids[0].item())

    hard = hard_embed(model, y)
    soft = soft_topk_embed(model, top_ids, top_probs)

    hard_logits, hard_cache = inject(model, ids, hard)
    soft_logits, soft_cache = inject(model, ids, soft)
    assert hard_cache is not soft_cache
    # The injected position's KV entries must differ between branches.
    hk = hard_cache.layers[0].keys
    sk = soft_cache.layers[0].keys
    assert not torch.allclose(hk[..., -1, :], sk[..., -1, :])
    # But the prompt positions must be identical (independent recomputation
    # of the same prompt).
    assert torch.allclose(hk[..., :-1, :], sk[..., :-1, :], atol=1e-5)
