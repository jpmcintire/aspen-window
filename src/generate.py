"""Manual autoregressive machinery. model.generate() is never used for
branching: everything is explicit model.forward() calls."""

import torch
import torch.nn.functional as F

from .model_utils import run_prompt


def topk_from_logits(logits: torch.Tensor, k: int):
    """logits: [1, vocab] -> (ids [k], probs [k]) of the top-k tokens."""
    probs = F.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_ids = probs.topk(k)
    return top_ids, top_probs


def inject(model, prompt_ids: torch.Tensor, embed: torch.Tensor):
    """Rerun the prompt from scratch (guaranteeing an independent KV cache),
    then feed `embed` via inputs_embeds at the next position.

    Returns (logits [1, vocab] after the injected position, cache).
    """
    _, cache = run_prompt(model, prompt_ids)
    with torch.no_grad():
        out = model(inputs_embeds=embed, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def step_hard(model, cache, token_id: int):
    """Ordinary hard-token step: feed token_id via input_ids on `cache`."""
    ids = torch.tensor([[token_id]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def greedy_continue(model, cache, logits: torch.Tensor, n_steps: int):
    """Ordinary greedy decoding for n_steps starting from `logits`/`cache`.
    Returns the list of generated token ids."""
    generated = []
    for _ in range(n_steps):
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        logits, cache = step_hard(model, cache, next_id)
    return generated


def free_run_branch(model, prompt_ids: torch.Tensor, embed: torch.Tensor, n_steps: int):
    """One-shot internal handoff followed by ordinary greedy generation.
    Returns the continuation token ids (tokens after the visible token y)."""
    logits, cache = inject(model, prompt_ids, embed)
    return greedy_continue(model, cache, logits, n_steps)


def locked_visible_run(model, prompt_ids: torch.Tensor, embeds: dict, n_steps: int,
                       driver: str = "hard"):
    """Locked-visible impulse response.

    embeds: {condition_name: [1,1,hidden] injected embedding}. Must include
    the `driver` condition. After injection, at every step the driver
    branch's argmax token is fed identically to ALL branches, so visible
    text never diverges; only KV state differs.

    Returns (driver_token_ids, per_step_probs) where per_step_probs is a
    list over steps of {condition_name: prob tensor [vocab]}.
    """
    states = {}
    for name, embed in embeds.items():
        logits, cache = inject(model, prompt_ids, embed)
        states[name] = (logits, cache)

    driver_tokens = []
    per_step = []
    for _ in range(n_steps):
        probs = {
            name: F.softmax(logits, dim=-1).squeeze(0)
            for name, (logits, _) in states.items()
        }
        per_step.append(probs)
        driver_tok = int(states[driver][0].argmax(dim=-1).item())
        driver_tokens.append(driver_tok)
        states = {
            name: step_hard(model, cache, driver_tok)
            for name, (_, cache) in states.items()
        }
    return driver_tokens, per_step
