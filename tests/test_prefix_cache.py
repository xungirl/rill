"""Prefix caching: hash-chained block reuse, LRU eviction, ref counts, engine equivalence."""

from __future__ import annotations

import random
from collections import Counter
from typing import ClassVar

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from ember.block_manager import BlockManager
from ember.engine import EngineConfig, LLMEngine
from ember.model import Qwen3Model
from ember.sampling import SamplingParams
from ember.scheduler import SchedulerConfig
from ember.sequence import Sequence

BS = 4


def admit_and_compute(bm: BlockManager, seq: Sequence) -> int:
    """What the scheduler + engine do for a fresh prompt: match, allocate rest, 'compute', cache."""
    cached = bm.match_prefix(seq)
    seq.num_computed_tokens = cached
    bm.allocate_slots(seq, seq.num_tokens - cached)
    seq.num_computed_tokens = seq.num_tokens
    bm.cache_full_blocks(seq)
    return cached


def seq(i: int, tokens: list[int]) -> Sequence:
    return Sequence(i, tokens, SamplingParams())


def test_shared_prefix_hits_full_blocks_only() -> None:
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    a = seq(0, list(range(10)))  # blocks [0-3][4-7][8,9]
    assert admit_and_compute(bm, a) == 0
    b = seq(1, [*range(8), 99, 98, 97])  # shares the two full blocks
    assert admit_and_compute(bm, b) == 8
    assert b.block_table[:2] == a.block_table[:2]
    assert bm.blocks[a.block_table[0]].ref_count == 2
    assert b.block_table[2] != a.block_table[2]  # partial / divergent block is private


def test_never_matches_the_final_token() -> None:
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    admit_and_compute(bm, seq(0, list(range(8))))
    again = seq(1, list(range(8)))  # identical prompt, exactly 2 full blocks
    assert admit_and_compute(bm, again) == 4  # last block recomputed: we need its logits


def test_chain_hash_depends_on_whole_prefix() -> None:
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    admit_and_compute(bm, seq(0, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    # second block's tokens equal, first block differs -> no hit for either
    assert admit_and_compute(bm, seq(1, [0, 0, 0, 0, 5, 6, 7, 8, 9])) == 0


def test_freed_blocks_stay_cached_and_are_revived() -> None:
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    a = seq(0, list(range(9)))
    admit_and_compute(bm, a)
    shared = a.block_table[:2]
    bm.free(a)
    assert bm.num_free_blocks == 16
    b = seq(1, list(range(9)))
    assert admit_and_compute(bm, b) == 8
    assert b.block_table[:2] == shared
    assert all(blk not in bm.free_block_ids for blk in shared)  # pulled out of the free list


def test_lru_eviction_order() -> None:
    bm = BlockManager(4, BS, enable_prefix_caching=True)
    old = seq(0, [1] * 4 + [2])  # 1 full block cached + 1 partial
    new = seq(1, [3] * 4 + [4])
    admit_and_compute(bm, old)
    bm.free(old)
    admit_and_compute(bm, new)
    bm.free(new)
    # all 4 blocks free; 2 hold cached contents. A 2-block request must reuse
    # the two uncached ones first, then a third request evicts the OLDEST cached one.
    filler = seq(2, [7] * 7)
    bm.match_prefix(filler)
    bm.allocate_slots(filler, 7)
    assert bm.num_evictions == 0
    third = seq(3, [8])
    bm.allocate_slots(third, 1)
    assert bm.num_evictions == 1
    bm.free(filler)
    bm.free(third)
    assert admit_and_compute(bm, seq(4, [3] * 4 + [4])) == 4  # newer prefix survived
    assert admit_and_compute(bm, seq(5, [1] * 4 + [2])) == 0  # older one was evicted


def test_hash_collision_is_rejected_by_token_check() -> None:
    bm = BlockManager(8, BS, enable_prefix_caching=True)
    a = seq(0, [1, 2, 3, 4, 5])
    admit_and_compute(bm, a)
    blk = bm.blocks[a.block_table[0]]
    blk.token_ids = (9, 9, 9, 9)  # simulate: same hash, different contents
    assert admit_and_compute(bm, seq(1, [1, 2, 3, 4, 5])) == 0


def test_disabled_cache_never_hits() -> None:
    bm = BlockManager(8, BS, enable_prefix_caching=False)
    admit_and_compute(bm, seq(0, list(range(9))))
    assert admit_and_compute(bm, seq(1, list(range(9)))) == 0
    assert not bm.cached


class PrefixCacheMachine(RuleBasedStateMachine):
    """Random admit/free over prompts that share prefixes from a small pool."""

    PREFIXES: ClassVar[list[list[int]]] = [[1] * 8, [2] * 4, [1] * 4 + [3] * 4]

    @initialize(num_blocks=st.integers(4, 30))
    def setup(self, num_blocks: int) -> None:
        self.bm = BlockManager(num_blocks, BS, enable_prefix_caching=True)
        self.live: list[Sequence] = []
        self.n = 0

    @rule(p=st.integers(0, 2), tail=st.lists(st.integers(10, 12), min_size=1, max_size=9))
    def admit(self, p: int, tail: list[int]) -> None:
        s = seq(self.n, self.PREFIXES[p] + tail)
        self.n += 1
        cached = self.bm.match_prefix(s)
        s.num_computed_tokens = cached
        if self.bm.can_allocate(s, s.num_tokens - cached):
            self.bm.allocate_slots(s, s.num_tokens - cached)
            s.num_computed_tokens = s.num_tokens
            self.bm.cache_full_blocks(s)
            self.live.append(s)
        else:
            self.bm.free(s)  # scheduler's rollback

    @precondition(lambda self: self.live)
    @rule(i=st.integers(0, 1000))
    def free(self, i: int) -> None:
        self.bm.free(self.live.pop(i % len(self.live)))

    @invariant()
    def consistent(self) -> None:
        if not hasattr(self, "bm"):
            return
        refs = Counter(b for s in self.live for b in s.block_table)
        for block in self.bm.blocks:
            assert block.ref_count == refs.get(block.block_id, 0)
        assert set(self.bm.free_block_ids) == {b.block_id for b in self.bm.blocks if b.ref_count == 0}
        for h, b in self.bm.cached.items():
            assert self.bm.blocks[b].block_hash == h
        assert sum(b.block_hash is not None for b in self.bm.blocks) == len(self.bm.cached)
        for s in self.live:  # every live sequence's tokens are still what its blocks hold
            for i, b in enumerate(s.block_table[: s.num_computed_tokens // BS]):
                if self.bm.blocks[b].block_hash is not None:
                    assert self.bm.blocks[b].token_ids == tuple(s.token_ids[i * BS : (i + 1) * BS])


PrefixCacheMachine.TestCase.settings = settings(
    max_examples=200, stateful_step_count=40, deadline=None, derandomize=True
)
test_prefix_cache_state_machine = PrefixCacheMachine.TestCase


@pytest.mark.parametrize("num_blocks", [256, 40])
def test_engine_outputs_identical_with_cache_on_and_off(tiny_model: Qwen3Model, num_blocks: int) -> None:
    rng = random.Random(5)
    system = [rng.randrange(500) for _ in range(37)]
    prompts = [system + [rng.randrange(500) for _ in range(rng.randint(1, 12))] for _ in range(16)]
    params = [SamplingParams(max_tokens=rng.randint(1, 20), ignore_eos=True) for _ in prompts]
    outs = {}
    engines = {}
    for on in (False, True):
        cfg = EngineConfig(
            block_size=BS,
            num_blocks=num_blocks,
            enable_prefix_caching=on,
            scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64),
        )
        engines[on] = LLMEngine(tiny_model, cfg)
        outs[on] = engines[on].generate(prompts, params)
    assert outs[True] == outs[False]
    assert engines[True].scheduler.prefix_hit_tokens > 0
    assert engines[False].scheduler.prefix_hit_tokens == 0
    if num_blocks == 40:
        assert engines[True].block_manager.num_evictions > 0  # small pool really evicted
