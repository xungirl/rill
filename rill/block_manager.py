"""Paged KV-cache bookkeeping: fixed-size blocks, a free list, per-sequence
block tables, reference counts and (optionally) hash-based prefix caching.
Pure Python - no tensors live here.

The physical KV pool (rill/paged_attention.py) is an array of `num_blocks`
blocks, each holding K/V for `block_size` consecutive tokens. A sequence's
`block_table[i]` is the physical block holding its tokens
[i*block_size, (i+1)*block_size). Logical token t therefore lives in slot
    block_table[t // block_size] * block_size + t % block_size
Blocks are handed out on demand, so a sequence never holds more than
block_size - 1 unused slots (internal fragmentation), and there is no external
fragmentation at all because any free block can extend any sequence.

Prefix caching (vLLM-v1 style):
  * Once a block is FULL and its K/V are written, it gets a content hash that
    chains the previous block's hash with this block's token ids:
        h_i = hash((h_{i-1}, tokens[i*bs:(i+1)*bs]))
    K/V of a token depend on every token before it, and the chain captures that.
  * `cached` maps hash -> block id. When a new request arrives, its prompt's
    full blocks are hashed and looked up; hits are shared (ref_count += 1)
    and their tokens are skipped by prefill.
  * A block whose ref_count drops to 0 keeps its hash and goes to the END of
    the free list; a freed block with no cached contents goes to the FRONT.
    Allocation takes from the front, so the free list doubles as an LRU
    eviction queue: empty blocks are reused first, and a cached block is only
    overwritten when it is the least-recently freed cached one. A hit on a free cached block pulls it out of the
    free list (O(1) with an OrderedDict).
  * Only full blocks are ever shared, and a full block is never written again,
    so no copy-on-write is needed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from rill.sequence import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None  # set once the block is full and cached
    token_ids: tuple[int, ...] = ()  # kept to rule out hash collisions on lookup


class NoFreeBlocksError(RuntimeError):
    pass


def block_hash(prev_hash: int | None, token_ids: tuple[int, ...]) -> int:
    # Python's hash of a tuple of ints is deterministic (no per-process salt).
    return hash((prev_hash, token_ids))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        # insertion order = LRU order (front = least recently freed)
        self.free_block_ids: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))
        self.cached: dict[int, int] = {}  # block hash -> block id
        self.num_evictions = 0

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
        # Reverse order: a sequence's tail is the least likely to be shared by
        # future requests, so it lands nearer the front of the LRU and is evicted first.
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table.clear()
        seq.block_hashes.clear()

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Let `child` share all of `parent`'s blocks (ref_count += 1 each).

        Shared blocks must be treated as read-only; this engine only ever
        shares *full* blocks so no copy-on-write is needed.
        """
        for block_id in parent.block_table:
            self.blocks[block_id].ref_count += 1
        child.block_table = list(parent.block_table)

    # ------------------------------------------------------------ prefix cache
    def match_prefix(self, seq: Sequence) -> int:
        """Attach cached blocks for the longest cached prefix of a fresh sequence.

        Returns the number of tokens whose K/V are now already available. At
        least one token is always left to compute: the engine needs the last
        token's logits to sample.
        """
        assert not seq.block_table and seq.num_computed_tokens == 0
        if not self.enable_prefix_caching:
            return 0
        bs = self.block_size
        max_blocks = (seq.num_tokens - 1) // bs
        prev: int | None = None
        hits: list[int] = []
        for i in range(max_blocks):
            tokens = tuple(seq.token_ids[i * bs : (i + 1) * bs])
            h = block_hash(prev, tokens)
            block_id = self.cached.get(h)
            if block_id is None or self.blocks[block_id].token_ids != tokens:
                break
            hits.append(block_id)
            seq.block_hashes.append(h)
            prev = h
        for block_id in hits:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                del self.free_block_ids[block_id]  # revived from the LRU free list
            block.ref_count += 1
        seq.block_table.extend(hits)
        return len(hits) * bs

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register every newly *full and computed* block of seq in the cache.

        Called after each engine step, i.e. only once the K/V are really in the
        pool - registering earlier would let another sequence hit a block whose
        contents have not been written yet.
        """
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        full = seq.num_computed_tokens // bs
        for i in range(len(seq.block_hashes), full):
            prev = seq.block_hashes[-1] if seq.block_hashes else None
            tokens = tuple(seq.token_ids[i * bs : (i + 1) * bs])
            h = block_hash(prev, tokens)
            seq.block_hashes.append(h)
            block = self.blocks[seq.block_table[i]]
            if h not in self.cached and block.block_hash is None:
                block.block_hash = h
                block.token_ids = tokens
                self.cached[h] = block.block_id

    # ----------------------------------------------------------------- helpers
    def _take_free_block(self) -> int:
        block_id, _ = self.free_block_ids.popitem(last=False)  # least recently freed
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"free block {block_id} has ref_count {block.ref_count}"
        if block.block_hash is not None:  # evict its cached contents
            del self.cached[block.block_hash]
            block.block_hash = None
            block.token_ids = ()
            self.num_evictions += 1
        block.ref_count = 1
        return block_id

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"double free of block {block_id}"
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids[block_id] = None
            if block.block_hash is None:
                # nothing worth keeping: reuse before any cached block
                self.free_block_ids.move_to_end(block_id, last=False)
            # else: stays at the back = most recently used cached block

    @staticmethod
    def slot(seq: Sequence, position: int, block_size: int) -> int:
        """Physical slot index (into the flattened KV pool) of a logical token position."""
        return seq.block_table[position // block_size] * block_size + position % block_size
