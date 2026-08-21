"""Distribution-comparison metrics with numerical safeguards."""

import torch

EPS = 1e-12


def _clamp(p: torch.Tensor) -> torch.Tensor:
    return p.clamp_min(EPS)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) in nats, with epsilon clamping."""
    p = _clamp(p.double())
    q = _clamp(q.double())
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p / q).log()).sum().item())


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence in nats (symmetric, bounded by ln 2)."""
    p = _clamp(p.double())
    q = _clamp(q.double())
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum().item()
                 + 0.5 * (q * (q / m).log()).sum().item())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float((a @ b / (a.norm() * b.norm() + EPS)).item())


def topk_overlap(p: torch.Tensor, q: torch.Tensor, k: int = 10) -> float:
    """|top-k(p) ∩ top-k(q)| / k."""
    top_p = set(p.topk(k).indices.tolist())
    top_q = set(q.topk(k).indices.tolist())
    return len(top_p & top_q) / k


def top_token_prob_diff(p_ref: torch.Tensor, q: torch.Tensor) -> float:
    """q's probability minus p_ref's probability for p_ref's argmax token."""
    top = int(p_ref.argmax().item())
    return float(q[top].item() - p_ref[top].item())
