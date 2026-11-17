"""Continuous-batching scheduler + engine tests (tiny random model, CPU)."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch

from rill.engine import EngineConfig, LLMEngine
from rill.model import Qwen3Model
from rill.sampling import SamplingParams
from rill.scheduler import SchedulerConfig
from rill.sequence import SequenceStatus
from rill.simple_generate import generate_greedy


def make_prompts(n: int, vocab: int, seed: int, lo: int = 1, hi: int = 40) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(vocab - 1) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def make_engine(model: Qwen3Model, num_blocks: int = 256, block_size: int = 4, **sched: object) -> LLMEngine:
    scheduler = SchedulerConfig(**sched)  # type: ignore[arg-type]
    return LLMEngine(model, EngineConfig(block_size=block_size, num_blocks=num_blocks, scheduler=scheduler))


def reference(model: Qwen3Model, prompts: list[list[int]], max_tokens: list[int]) -> list[list[int]]:
    return [generate_greedy(model, p, m) for p, m in zip(prompts, max_tokens, strict=True)]


@pytest.mark.parametrize(
    ("label", "num_blocks", "sched"),
    [
        ("roomy", 256, {"max_num_seqs": 8, "max_num_batched_tokens": 256}),
        ("chunked", 256, {"max_num_seqs": 8, "max_num_batched_tokens": 7}),
        ("tight_memory", 24, {"max_num_seqs": 8, "max_num_batched_tokens": 64}),
        ("sequential", 256, {"max_num_seqs": 1, "max_num_batched_tokens": 64}),
        ("static", 256, {"max_num_seqs": 4, "max_num_batched_tokens": 64, "policy": "static"}),
    ],
)
def test_engine_matches_single_sequence_reference(
    tiny_model: Qwen3Model, label: str, num_blocks: int, sched: dict[str, Any]
) -> None:
    prompts = make_prompts(12, tiny_model.cfg.vocab_size, seed=len(label))
    max_tokens = [random.Random(i).randint(1, 30) for i in range(12)]
    engine = make_engine(tiny_model, num_blocks=num_blocks, **sched)
    params = [SamplingParams(max_tokens=m, ignore_eos=True) for m in max_tokens]
    got = engine.generate(prompts, params)
    assert got == reference(tiny_model, prompts, max_tokens)
    assert engine.block_manager.num_free_blocks == num_blocks  # everything freed
    if label == "tight_memory":
        assert engine.scheduler.num_preemptions > 0  # the scenario really exercised preemption


def test_step_invariants_under_random_load(tiny_model: Qwen3Model) -> None:
    cfg = SchedulerConfig(max_num_seqs=5, max_num_batched_tokens=16)
    engine = make_engine(tiny_model, num_blocks=20, **cfg.__dict__)
    rng = random.Random(3)
    pending = make_prompts(30, tiny_model.cfg.vocab_size, seed=9, hi=20)
    steps = 0
    while pending or engine.has_unfinished():
        if pending and rng.random() < 0.5:  # requests trickle in while others decode
            engine.add_request(pending.pop(), SamplingParams(max_tokens=rng.randint(1, 15), ignore_eos=True))
        sched = engine.scheduler
        before_running = list(sched.running)
        engine.step()
        steps += 1
        assert len(sched.running) <= cfg.max_num_seqs
        for s in sched.running:
            assert s.status is SequenceStatus.RUNNING
            assert len(s.block_table) == engine.block_manager.blocks_for(s.num_computed_tokens)
        for s in sched.waiting:
            assert s.block_table == [] and s.num_computed_tokens == 0
        # every previously running decode either progressed, finished or was preempted
        for s in before_running:
            assert s.is_finished or s.status in (SequenceStatus.RUNNING, SequenceStatus.WAITING)
    assert engine.block_manager.num_free_blocks == 20
    assert steps > 0


def test_token_budget_and_decode_first(tiny_model: Qwen3Model) -> None:
    engine = make_engine(tiny_model, max_num_seqs=4, max_num_batched_tokens=10)
    for p in make_prompts(4, tiny_model.cfg.vocab_size, seed=1, lo=25, hi=30):
        engine.add_request(p, SamplingParams(max_tokens=5, ignore_eos=True))
    saw_mixed = False
    while engine.has_unfinished():
        out = engine.scheduler.schedule()
        assert out.num_tokens <= 10
        decodes = [s for s in out.batch if s.num_new_tokens == 1 and s.seq.num_output_tokens > 0]
        prefills = [s for s in out.batch if s.seq.num_output_tokens == 0]
        if decodes and prefills:
            saw_mixed = True
            # decodes are scheduled before any prefill chunk
            assert out.batch.index(decodes[-1]) < out.batch.index(prefills[0])
        # finish the step the way the engine would
        logits = engine.runner.execute(out.batch)
        rows = iter(logits.argmax(-1).tolist())
        for item in out.batch:
            done = item.completes_sequence
            item.seq.num_computed_tokens += item.num_new_tokens
            if done:
                item.seq.append_token(next(rows))
                if item.seq.num_output_tokens >= item.seq.sampling.max_tokens:
                    engine.scheduler.finish(item.seq, "length")
    assert saw_mixed  # chunked prefill really overlapped with decoding


def test_static_policy_admits_only_when_drained(tiny_model: Qwen3Model) -> None:
    engine = make_engine(tiny_model, max_num_seqs=3, max_num_batched_tokens=512, policy="static")
    for i, p in enumerate(make_prompts(7, tiny_model.cfg.vocab_size, seed=4)):
        engine.add_request(p, SamplingParams(max_tokens=2 + 3 * i, ignore_eos=True))
    admitted_while_busy = False
    while engine.has_unfinished():
        before = {id(s) for s in engine.scheduler.running}
        engine.step()
        new = {id(s) for s in engine.scheduler.running} - before
        if new and before:
            admitted_while_busy = True
    assert not admitted_while_busy


def test_abort_frees_blocks(tiny_model: Qwen3Model) -> None:
    engine = make_engine(tiny_model, num_blocks=64)
    a = engine.add_request([1, 2, 3, 4, 5, 6, 7, 8, 9], SamplingParams(max_tokens=50, ignore_eos=True))
    engine.add_request([3, 4], SamplingParams(max_tokens=50, ignore_eos=True))
    engine.step()
    engine.step()
    used = engine.block_manager.num_used_blocks
    engine.abort(a)
    assert engine.block_manager.num_used_blocks < used
    assert len(engine.scheduler.running) == 1


def test_rejects_impossible_requests(tiny_model: Qwen3Model) -> None:
    engine = make_engine(tiny_model, num_blocks=4, block_size=4)
    with pytest.raises(ValueError, match="never fit"):
        engine.add_request(list(range(10)), SamplingParams(max_tokens=10))
    engine2 = make_engine(tiny_model, max_num_batched_tokens=8, chunked_prefill=False)
    with pytest.raises(ValueError, match="chunked prefill is off"):
        engine2.add_request(list(range(9)), SamplingParams(max_tokens=1))


def test_eos_stops_generation(tiny_model: Qwen3Model) -> None:
    prompt = [5, 6, 7]
    ref = generate_greedy(tiny_model, prompt, 20)
    engine = make_engine(tiny_model)
    engine.eos_ids = frozenset({ref[4]})
    out = engine.generate([prompt], SamplingParams(max_tokens=20))[0]
    assert out == ref[: ref.index(ref[4]) + 1]


@torch.no_grad()
def test_generate_preserves_input_order(tiny_model: Qwen3Model) -> None:
    prompts = make_prompts(6, tiny_model.cfg.vocab_size, seed=11)
    engine = make_engine(tiny_model, max_num_seqs=2)
    out = engine.generate(prompts, [SamplingParams(max_tokens=10 - i, ignore_eos=True) for i in range(6)])
    assert [len(o) for o in out] == [10, 9, 8, 7, 6, 5]


@pytest.mark.real_model
def test_real_model_continuous_batching_matches_reference_fp32() -> None:
    """With exact (fp32, CPU) arithmetic, continuous batching with chunked prefill and
    preemption reproduces single-sequence greedy decoding token for token."""
    from rill.loader import load_model, resolve_model_dir
    from rill.tokenizer import Tokenizer

    tok = Tokenizer(resolve_model_dir())
    model = load_model(device="cpu", dtype=torch.float32)
    texts = ["The capital of France is", "def quicksort(arr):", "List three primary colors:", "Hello, my name is"]
    prompts = [tok.encode(t) for t in texts]
    lengths = [20, 12, 16, 8]
    engine = make_engine(model, num_blocks=6, block_size=8, max_num_seqs=3, max_num_batched_tokens=6)
    out = engine.generate(prompts, [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths])
    assert out == reference(model, prompts, lengths)
    assert engine.scheduler.num_preemptions > 0
