"""Paged KV-cache bookkeeping: fixed-size blocks, a free list, per-sequence
block tables and reference counts. Pure Python - no tensors live here.

The physical KV pool (ember/paged_attention.py) is an array of `num_blocks`
blocks, each holding K/V for `block_size` consecutive tokens. A sequence's
`block_table[i]` is the physical block holding its tokens
[i*block_size, (i+1)*block_size). Logical token t therefore lives in slot
    block_table[t // block_size] * block_size + t % block_size
Blocks are handed out on demand, so a sequence never holds more than
block_size - 1 unused slots (internal fragmentation), and there is no external
fragmentation at all because any free block can extend any sequence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ember.sequence import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0


class NoFreeBlocksError(RuntimeError):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    # ------------------------------------------------------------------ queries
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def usage(self) -> float:
        return self.num_used_blocks / self.num_blocks

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)  # ceil division

    def num_new_blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        """Blocks `seq` must gain so that its first num_computed+num_new tokens have slots."""
        total = self.blocks_for(seq.num_computed_tokens + num_new_tokens)
        return max(0, total - len(seq.block_table))

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self.num_new_blocks_needed(seq, num_new_tokens) <= self.num_free_blocks

    # ---------------------------------------------------------------- mutations
    def allocate_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        """Grow seq.block_table to cover `num_new_tokens` more tokens.

        One method for every case: whole-prompt prefill, one prefill chunk, or
        a single decode token (which needs a new block only every block_size
        steps). Raises NoFreeBlocksError without side effects if the pool is
        too small - the scheduler checks can_allocate first and preempts.
        """
        need = self.num_new_blocks_needed(seq, num_new_tokens)
        if need > self.num_free_blocks:
            raise NoFreeBlocksError(f"need {need} blocks, {self.num_free_blocks} free")
        for _ in range(need):
            seq.block_table.append(self._take_free_block())

    def free(self, seq: Sequence) -> None:
        """Drop seq's references. Blocks whose ref_count hits 0 return to the free list."""
        # Reverse order: the tail of a sequence is the least likely to be shared,
        # so it should be the first to be reused (matters once prefix caching lands).
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table.clear()

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Let `child` share all of `parent`'s blocks (ref_count += 1 each).

        Shared blocks must be treated as read-only; this engine only ever
        shares *full* blocks so no copy-on-write is needed (see prefix caching).
        """
        for block_id in parent.block_table:
            self.blocks[block_id].ref_count += 1
        child.block_table = list(parent.block_table)

    # ----------------------------------------------------------------- helpers
    def _take_free_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"free block {block_id} has ref_count {block.ref_count}"
        block.ref_count = 1
        return block_id

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"double free of block {block_id}"
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)

    @staticmethod
    def slot(seq: Sequence, position: int, block_size: int) -> int:
        """Physical slot index (into the flattened KV pool) of a logical token position."""
        return seq.block_table[position // block_size] * block_size + position % block_size
