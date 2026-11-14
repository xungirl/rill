"""Sampling parameters and the sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0  # 0 = greedy
    ignore_eos: bool = False  # benchmarks: force exactly max_tokens outputs

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")


def sample(logits: Tensor, params: list[SamplingParams]) -> list[int]:
    """Pick one token per row of `logits` [B, V]."""
    if any(p.temperature > 0 for p in params):
        raise NotImplementedError("only greedy sampling in this stage")
    return torch.argmax(logits, dim=-1).tolist()
