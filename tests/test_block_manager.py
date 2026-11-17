"""Block manager invariants, checked with hypothesis stateful testing."""

from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from rill.block_manager import BlockManager, NoFreeBlocksError
from rill.sampling import SamplingParams
from rill.sequence import Sequence


def make_seq(seq_id: int, n: int) -> Sequence:
    return Sequence(seq_id, list(range(n)), SamplingParams())


def grow(bm: BlockManager, seq: Sequence, n: int) -> None:
    """Allocate slots for n more tokens and mark them computed (what the engine does)."""
    bm.allocate_slots(seq, n)
    seq.num_computed_tokens += n
    while seq.num_tokens < seq.num_computed_tokens:
        seq.append_token(0)


def test_allocate_exact_block_counts() -> None:
    bm = BlockManager(num_blocks=10, block_size=4)
    s = make_seq(0, 9)
    grow(bm, s, 9)
    assert len(s.block_table) == 3  # ceil(9/4)
    grow(bm, s, 3)  # 12 tokens -> still 3 blocks
    assert len(s.block_table) == 3
    grow(bm, s, 1)  # 13th token needs a 4th block
    assert len(s.block_table) == 4
    assert bm.num_free_blocks == 6
    bm.free(s)
    assert bm.num_free_blocks == 10 and s.block_table == []


def test_allocation_failure_has_no_side_effects() -> None:
    bm = BlockManager(num_blocks=2, block_size=4)
    s = make_seq(0, 12)
    assert not bm.can_allocate(s, 12)
    with pytest.raises(NoFreeBlocksError):
        bm.allocate_slots(s, 12)
    assert s.block_table == [] and bm.num_free_blocks == 2


def test_slot_mapping() -> None:
    bm = BlockManager(num_blocks=8, block_size=4)
    a, b = make_seq(0, 3), make_seq(1, 6)
    grow(bm, a, 3)
    grow(bm, b, 6)
    assert b.block_table == [1, 2]
    assert BlockManager.slot(b, 0, 4) == 4
    assert BlockManager.slot(b, 5, 4) == 2 * 4 + 1


def test_double_free_is_caught() -> None:
    bm = BlockManager(num_blocks=4, block_size=4)
    s = make_seq(0, 4)
    grow(bm, s, 4)
    table = list(s.block_table)
    bm.free(s)
    s.block_table = table
    with pytest.raises(AssertionError):
        bm.free(s)


class BlockManagerMachine(RuleBasedStateMachine):
    """Random interleavings of allocate / decode / fork / free."""

    @initialize(num_blocks=st.integers(1, 40), block_size=st.sampled_from([1, 2, 4, 16]))
    def setup(self, num_blocks: int, block_size: int) -> None:
        self.bm = BlockManager(num_blocks, block_size)
        self.live: list[Sequence] = []
        self.next_id = 0

    @rule(n=st.integers(1, 70))
    def add(self, n: int) -> None:
        s = make_seq(self.next_id, n)
        self.next_id += 1
        if self.bm.can_allocate(s, n):
            grow(self.bm, s, n)
            self.live.append(s)
        else:
            free_before = self.bm.num_free_blocks
            with pytest.raises(NoFreeBlocksError):
                self.bm.allocate_slots(s, n)
            assert self.bm.num_free_blocks == free_before

    @precondition(lambda self: self.live)
    @rule(i=st.integers(0, 10_000), n=st.integers(1, 20))
    def decode(self, i: int, n: int) -> None:
        s = self.live[i % len(self.live)]
        # shared (forked) blocks are read-only; only grow sequences that own their tail
        if s.block_table and self.bm.blocks[s.block_table[-1]].ref_count > 1:
            return
        if self.bm.can_allocate(s, n):
            grow(self.bm, s, n)

    @precondition(lambda self: self.live)
    @rule(i=st.integers(0, 10_000))
    def fork(self, i: int) -> None:
        parent = self.live[i % len(self.live)]
        child = make_seq(self.next_id, parent.num_tokens)
        self.next_id += 1
        self.bm.fork(parent, child)
        child.num_computed_tokens = parent.num_computed_tokens
        self.live.append(child)

    @precondition(lambda self: self.live)
    @rule(i=st.integers(0, 10_000))
    def free(self, i: int) -> None:
        s = self.live.pop(i % len(self.live))
        self.bm.free(s)

    @invariant()
    def ref_counts_equal_table_membership(self) -> None:
        if not hasattr(self, "bm"):
            return
        refs = Counter(b for s in self.live for b in s.block_table)
        for block in self.bm.blocks:
            assert block.ref_count == refs.get(block.block_id, 0)

    @invariant()
    def free_list_is_exactly_the_unreferenced_blocks(self) -> None:
        if not hasattr(self, "bm"):
            return
        free = list(self.bm.free_block_ids)
        assert len(free) == len(set(free)), "duplicate in free list"
        assert set(free) == {b.block_id for b in self.bm.blocks if b.ref_count == 0}
        assert self.bm.num_free_blocks + self.bm.num_used_blocks == self.bm.num_blocks

    @invariant()
    def tables_cover_computed_tokens_with_minimal_blocks(self) -> None:
        if not hasattr(self, "bm"):
            return
        for s in self.live:
            assert len(s.block_table) == self.bm.blocks_for(s.num_computed_tokens)
            assert len(set(s.block_table)) == len(s.block_table)


BlockManagerMachine.TestCase.settings = settings(
    max_examples=200, stateful_step_count=40, deadline=None, derandomize=True
)
test_block_manager_state_machine = BlockManagerMachine.TestCase
