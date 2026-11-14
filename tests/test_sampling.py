"""Sampler correctness: top-k/top-p masks, temperature, seeding, batch/preemption independence."""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ember.engine import EngineConfig, LLMEngine
from ember.model import Qwen3Model
from ember.sampling import SamplingParams, apply_top_k_top_p, sample
from ember.scheduler import SchedulerConfig


def reference_keep_set(logits: list[float], top_k: int, top_p: float) -> set[int]:
    """Brute-force definition: top-k by logit, renormalize, then smallest prefix with mass >= top_p."""
    order = sorted(range(len(logits)), key=lambda i: -logits[i])[:top_k]
    z = [math.exp(logits[i] - logits[order[0]]) for i in order]
    total = sum(z)
    keep, mass = set(), 0.0
    for i, w in zip(order, z, strict=True):
        if mass >= top_p - 1e-12:
            break
        keep.add(i)
        mass += w / total
    return keep


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    logits=st.lists(st.floats(-5, 5, allow_nan=False).map(lambda x: round(x, 3)), min_size=2, max_size=12, unique=True),
    top_k=st.integers(1, 12),
    top_p=st.floats(0.05, 1.0),
)
def test_top_k_top_p_mask_matches_reference(logits: list[float], top_k: int, top_p: float) -> None:
    t = torch.tensor([logits], dtype=torch.float32)
    k = min(top_k, len(logits))
    out = apply_top_k_top_p(t, torch.tensor([k]), torch.tensor([top_p]))
    kept = {i for i, v in enumerate(out[0].tolist()) if v != float("-inf")}
    assert kept == reference_keep_set(logits, k, top_p)
    # surviving logits are unchanged
    for i in kept:
        assert out[0, i].item() == pytest.approx(logits[i])


def test_greedy_and_degenerate_settings_pick_argmax() -> None:
    logits = torch.randn(6, 50)
    best = logits.argmax(-1).tolist()
    assert sample(logits, [SamplingParams(temperature=0.0)] * 6) == best
    g = [torch.Generator().manual_seed(i) for i in range(6)]
    assert sample(logits, [SamplingParams(temperature=1.0, top_k=1)] * 6, list(g)) == best
    assert sample(logits, [SamplingParams(temperature=1.0, top_p=1e-6)] * 6, list(g)) == best


def test_temperature_sampling_follows_softmax() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    temp = 0.7
    expected = torch.softmax(logits[0] / temp, -1)
    gen = torch.Generator().manual_seed(0)
    n = 20_000
    counts = torch.zeros(4)
    params = [SamplingParams(temperature=temp)]
    for _ in range(n):
        counts[sample(logits, params, [gen])[0]] += 1
    freq = counts / n
    # 4 standard errors of a binomial proportion: deterministic given the seed
    tol = 4 * torch.sqrt(expected * (1 - expected) / n)
    assert torch.all((freq - expected).abs() < tol), (freq, expected)


def test_top_k_never_samples_outside_top_k() -> None:
    logits = torch.arange(10, dtype=torch.float32).unsqueeze(0)  # token 9 most likely
    gen = torch.Generator().manual_seed(1)
    seen = {sample(logits, [SamplingParams(temperature=5.0, top_k=3)], [gen])[0] for _ in range(500)}
    assert seen == {7, 8, 9}


def test_invalid_params_rejected() -> None:
    for bad in ({"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}, {"top_k": -2}, {"temperature": -1.0}):
        with pytest.raises(ValueError, match="must be"):
            SamplingParams(**bad)


def _engine(model: Qwen3Model, num_blocks: int = 256, max_num_seqs: int = 8) -> LLMEngine:
    sched = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=64)
    return LLMEngine(model, EngineConfig(block_size=4, num_blocks=num_blocks, scheduler=sched))


def test_seeded_request_is_independent_of_batch_and_preemption(tiny_model: Qwen3Model) -> None:
    sp = SamplingParams(max_tokens=24, temperature=0.9, top_p=0.95, top_k=50, seed=1234, ignore_eos=True)
    prompt = [5, 9, 2, 7, 1]
    alone = _engine(tiny_model).generate([prompt], sp)[0]
    others = [[i, i + 1, i + 2] * 5 for i in range(1, 8)]
    busy = _engine(tiny_model).generate(
        [*others, prompt], [SamplingParams(max_tokens=30, temperature=1.0, ignore_eos=True)] * 7 + [sp]
    )
    assert busy[-1] == alone  # neighbours in the batch do not change a seeded request's tokens
    tight = _engine(tiny_model, num_blocks=20, max_num_seqs=8)
    out = tight.generate(
        [*others, prompt], [SamplingParams(max_tokens=30, temperature=1.0, ignore_eos=True)] * 7 + [sp]
    )
    assert tight.scheduler.num_preemptions > 0
    assert out[-1] == alone  # ... and neither does being preempted and recomputed


def test_different_seeds_give_different_samples(tiny_model: Qwen3Model) -> None:
    outs = {
        tuple(_engine(tiny_model).generate([[1, 2, 3]], SamplingParams(max_tokens=16, temperature=1.0, seed=s))[0])
        for s in range(6)
    }
    assert len(outs) > 1
