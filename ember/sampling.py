"""Sampling parameters and the batched sampler.

Pipeline for one engine step (B rows that need a token):
  1. greedy rows (temperature == 0): argmax, done.
  2. other rows, batched on the model's device:
       logits / temperature -> top-k mask -> top-p (nucleus) mask -> softmax
  3. draw one uniform number per row from that request's own torch.Generator
     and pick the token by inverse CDF (batched cumsum + searchsorted on the
     device). A request with `seed` gets a generator seeded once and
     advanced only by its own draws, so its output does not depend on which
     other requests share the batch (up to floating-point differences in the
     logits themselves; see BENCHMARK.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0  # keep the smallest set of tokens whose probability mass >= top_p
    top_k: int = -1  # keep the k most likely tokens; -1 = all
    seed: int | None = None  # per-request RNG seed for reproducible sampling
    ignore_eos: bool = False  # benchmarks: force exactly max_tokens outputs

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def apply_top_k_top_p(logits: Tensor, top_k: Tensor, top_p: Tensor) -> Tensor:
    """Mask (set to -inf) logits outside each row's top-k and top-p sets.

    logits: [B, V] float32; top_k: [B] long (V = disabled); top_p: [B] float.
    """
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    ranks = torch.arange(logits.shape[-1], device=logits.device).expand_as(sorted_logits)
    drop = ranks >= top_k.unsqueeze(1)  # top-k: rank >= k
    probs = sorted_logits.masked_fill(drop, float("-inf")).softmax(dim=-1)
    # top-p: drop a token if the mass of strictly-more-likely tokens already reaches p.
    # The most likely token always survives (its preceding mass is 0).
    mass_before = probs.cumsum(dim=-1) - probs
    drop |= mass_before >= top_p.unsqueeze(1)
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample(
    logits: Tensor,
    params: list[SamplingParams],
    generators: list[torch.Generator | None] | None = None,
) -> list[int]:
    """Pick one token per row of `logits` [B, V] (float32).

    generators[i] is row i's RNG (None = torch's global RNG); only used for non-greedy rows.
    """
    gens = generators or [None] * len(params)
    out = logits.argmax(dim=-1).tolist()
    rows = [i for i, p in enumerate(params) if not p.greedy]
    if not rows:
        return out
    vocab = logits.shape[-1]
    dev = logits.device
    sub = logits.index_select(0, torch.tensor(rows, device=dev))
    temps = torch.tensor([params[i].temperature for i in rows], device=dev).unsqueeze(1)
    top_k = torch.tensor([params[i].top_k if params[i].top_k > 0 else vocab for i in rows], device=dev)
    top_p = torch.tensor([params[i].top_p for i in rows], device=dev)
    sub = sub / temps
    if any(params[i].top_k > 0 or params[i].top_p < 1 for i in rows):
        sub = apply_top_k_top_p(sub, top_k, top_p)
    # Inverse-CDF sampling, batched on device: token = first index whose cumulative
    # probability reaches u. Only the scalar u comes from each row's own generator,
    # so seeding semantics are per request. u is in (0, 1] so a zero-probability
    # (masked) token can never be picked. (A per-row torch.multinomial on the CPU cost
    # 54 of 77 ms per step at batch 32 - see BENCHMARK.md Stage 7.)
    u = torch.tensor([1.0 - float(torch.rand((), generator=gens[i])) for i in rows], device=dev)
    cdf = sub.softmax(dim=-1).cumsum(dim=-1)
    picked = torch.searchsorted(cdf, (u * cdf[:, -1]).unsqueeze(1)).squeeze(1).clamp_(max=vocab - 1)
    for j, tok in zip(rows, picked.tolist(), strict=True):
        out[j] = tok
    return out
