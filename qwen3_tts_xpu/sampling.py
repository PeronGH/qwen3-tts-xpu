"""Graph-capturable sampling (no host syncs, no data-dependent shapes)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplerParams:
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float

    def width(self, vocab: int) -> int:
        """Number of candidates kept, i.e. the noise width needed per draw."""
        return min(self.top_k, vocab) if self.top_k > 0 else vocab


def sample(logits: torch.Tensor, p: SamplerParams, noise: torch.Tensor) -> torch.Tensor:
    """Pick one token per row of float32 `logits` [B, V].

    Matches transformers' warper order (temperature -> top-k -> top-p -> multinomial), except
    that ties at the k-th logit are dropped. `noise` [B, width] must hold Exp(1) samples; the
    draw uses the exponential race argmax(p_i / e_i), which is distributed as multinomial(p).
    """
    if not p.do_sample:
        return logits.argmax(dim=-1)
    if p.temperature != 1.0:
        logits = logits / p.temperature
    vals, idx = logits.topk(p.width(logits.shape[-1]), dim=-1)
    probs = vals.softmax(dim=-1)
    if p.top_p < 1.0:
        # transformers removes the ascending-sorted prefix whose cumulative mass is <= 1 - top_p,
        # always keeping the most likely token.
        tail_mass = probs.flip(-1).cumsum(-1).flip(-1)
        remove = tail_mass <= 1 - p.top_p
        remove[..., 0] = False
        probs = probs.masked_fill(remove, 0.0)
    choice = (probs / noise).argmax(dim=-1, keepdim=True)
    return idx.gather(-1, choice).squeeze(-1)


def apply_repetition_penalty(
    logits: torch.Tensor, seen: torch.Tensor, penalty: float
) -> torch.Tensor:
    """`seen` [V] is 1.0 for tokens generated so far."""
    if penalty == 1.0:
        return logits
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen.bool(), penalized, logits)
