"""Constructions of the internal embedding fed back at the branch point.

Every function returns a [1, 1, hidden] tensor. The externally visible
token is always the same hard token y; only this internal vector differs.
"""

import torch

from .model_utils import embedding_matrix, token_embed


def hard_embed(model, y_id: int) -> torch.Tensor:
    """Ordinary recurrence: E[y]."""
    return token_embed(model, y_id)


def soft_topk_embed(model, top_ids, top_probs) -> torch.Tensor:
    """sum_i p_i * E[token_i], with p renormalized over the top-k set."""
    E = embedding_matrix(model)
    top_ids = torch.as_tensor(top_ids, dtype=torch.long)
    p = torch.as_tensor(top_probs, dtype=E.dtype)
    p = p / p.sum()
    vecs = E[top_ids]                      # [k, hidden]
    mix = (p.unsqueeze(1) * vecs).sum(0)   # [hidden]
    return mix.view(1, 1, -1)


def norm_match(embed: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale `embed` so its L2 norm equals that of `reference`."""
    ref_norm = reference.norm()
    cur_norm = embed.norm()
    return embed * (ref_norm / cur_norm)


def beta_mixture(model, y_id: int, z_id: int, beta: float) -> torch.Tensor:
    """c(beta) = (1-beta) E[y] + beta E[z]."""
    ey = token_embed(model, y_id)
    ez = token_embed(model, z_id)
    return (1.0 - beta) * ey + beta * ez


def random_orthogonal_perturbation(
    model, y_id: int, distance: float, generator: torch.Generator
) -> torch.Tensor:
    """CONTROL C: E[y] + delta, where delta is a random direction orthogonal
    to E[y] with ||delta|| == distance (matched to CONTROL B's L2 distance
    from E[y])."""
    h = token_embed(model, y_id).view(-1)
    r = torch.randn(h.shape, generator=generator, dtype=h.dtype)
    # Remove the component along E[y] so the perturbation is orthogonal.
    r = r - (r @ h) / (h @ h) * h
    r = r / r.norm() * distance
    return (h + r).view(1, 1, -1)


def unrelated_mixture(model, y_id: int, unrelated_id: int, beta: float) -> torch.Tensor:
    """CONTROL D: (1-beta) E[y] + beta E[unrelated]."""
    return beta_mixture(model, y_id, unrelated_id, beta)


def toward_token_at_distance(model, y_id: int, other_id: int,
                             distance: float) -> torch.Tensor:
    """E[y] + delta, where delta points from E[y] toward E[other] with
    ||delta|| == distance. Used for the distance-matched CONTROL D: a
    beta-mixture's displacement direction is exactly (E[other] - E[y]),
    so this preserves 'unrelated token direction' while matching the L2
    displacement of CONTROL B exactly (norm-matching alone does not,
    because the displacement then depends on the angle between E[y] and
    E[other])."""
    h = token_embed(model, y_id).view(-1)
    o = token_embed(model, other_id).view(-1)
    d = o - h
    d = d / d.norm() * distance
    return (h + d).view(1, 1, -1)
