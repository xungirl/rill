"""Paged KV cache storage and the attention backend that reads through block tables.

Storage: one K pool and one V pool per layer, shape
    [num_blocks * block_size, n_kv_heads, head_dim]
i.e. a flat array of token *slots*; block b owns slots [b*bs, (b+1)*bs).

A forward step processes a packed batch of T new tokens from B sequences
(any mix of prefill chunks and single decode tokens). Per layer we:
  1. scatter the new k/v into their slots           (slot_mapping  [T])
  2. gather each sequence's blocks back out          (block_ids     [B*max_blocks])
     into a padded [B, n_kv, S, d] tensor, S = max_blocks * bs
  3. run batched SDPA with a [B, 1, max_q, S] mask that encodes both
     "key exists" (j < context_len) and causality (j <= query position).
     Decode rows and prefill rows are attended in separate groups so a long
     prefill chunk does not pad every decode row to its length.

This is the same data layout vLLM uses, but step 2 materialises the gathered
K/V with ordinary PyTorch indexing instead of a fused paged-attention kernel
(there is no such kernel for MPS in PyTorch). That costs extra memory traffic
and is the main known performance gap; see README "Limitations".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rill.attention import sdpa
from rill.config import ModelConfig


class PagedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (cfg.num_hidden_layers, num_blocks * block_size, cfg.num_key_value_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    @staticmethod
    def num_blocks_for_budget(cfg: ModelConfig, budget_bytes: int, block_size: int, dtype: torch.dtype) -> int:
        per_block = cfg.kv_bytes_per_token(torch.empty((), dtype=dtype).element_size()) * block_size
        return budget_bytes // per_block


@dataclass
class AttentionGroup:
    """Sequences attended together in one padded SDPA call.

    Decode rows (1 new token) and prefill rows (many new tokens) form separate
    groups. Padding them together would pad every decode row up to the prefill
    chunk length: with one 1024-token chunk and 31 decodes, ~32x the attention
    work (measured as a serving collapse in Stage 6, see PROGRESS_LOG.md).
    """

    token_index: Tensor  # [T_g] rows of the packed batch that belong to this group
    query_index: Tensor  # [T_g] where each of those rows goes in the padded [B_g * max_q] layout
    block_ids: Tensor  # [B_g * max_blocks] block tables, padded with block 0 (masked out)
    mask: Tensor  # [B_g, 1, max_q, S] bool, S = max_blocks * block_size
    batch_size: int
    max_q: int
    max_blocks: int


@dataclass
class PagedAttentionMetadata:
    """Everything attention needs about the batch layout; built once per step,
    shared by all layers."""

    slot_mapping: Tensor  # [T] destination slot of each new token
    groups: list[AttentionGroup]
    num_tokens: int

    @classmethod
    def build(
        cls,
        block_tables: list[list[int]],
        query_lens: list[int],
        context_lens: list[int],
        block_size: int,
        device: torch.device | str,
    ) -> PagedAttentionMetadata:
        """context_lens[b] = number of tokens with K/V after this step (old + new);
        the new tokens of seq b sit at positions [context_len - query_len, context_len)."""
        slots: list[int] = []
        starts: list[int] = []  # offset of each sequence's first token in the packed batch
        for table, q_len, ctx in zip(block_tables, query_lens, context_lens, strict=True):
            starts.append(len(slots))
            for pos in range(ctx - q_len, ctx):
                slots.append(table[pos // block_size] * block_size + pos % block_size)
        decode_rows = [r for r, q in enumerate(query_lens) if q == 1]
        prefill_rows = [r for r, q in enumerate(query_lens) if q > 1]
        groups = [
            _build_group(rows, block_tables, query_lens, context_lens, starts, block_size, device)
            for rows in (decode_rows, prefill_rows)
            if rows
        ]
        return cls(torch.tensor(slots, dtype=torch.long, device=device), groups, len(slots))


def _build_group(
    rows: list[int],
    block_tables: list[list[int]],
    query_lens: list[int],
    context_lens: list[int],
    starts: list[int],
    block_size: int,
    device: torch.device | str,
) -> AttentionGroup:
    b = len(rows)
    q_lens = [query_lens[r] for r in rows]
    ctxs = [context_lens[r] for r in rows]
    max_q = max(q_lens)
    max_blocks = max(len(block_tables[r]) for r in rows)
    token_index: list[int] = []
    query_index: list[int] = []
    for i, r in enumerate(rows):
        token_index.extend(range(starts[r], starts[r] + query_lens[r]))
        query_index.extend(range(i * max_q, i * max_q + query_lens[r]))
    padded = [blk for r in rows for blk in block_tables[r] + [0] * (max_blocks - len(block_tables[r]))]

    # mask[b, 0, i, j]: query row i of seq b (absolute pos ctx-q_len+i) may see key j.
    ctx_t = torch.tensor(ctxs).view(b, 1, 1)
    qlen_t = torch.tensor(q_lens).view(b, 1, 1)
    i_t = torch.arange(max_q).view(1, max_q, 1)
    j_t = torch.arange(max_blocks * block_size).view(1, 1, -1)
    q_pos = ctx_t - qlen_t + i_t
    # Padding query rows (i >= q_len) still see key 0 (q_pos >= 0, ctx >= 1), so no
    # row is fully masked (which would give NaN); their outputs are dropped via query_index.
    mask = (j_t <= q_pos) & (j_t < ctx_t)

    def t(xs: list[int]) -> Tensor:
        return torch.tensor(xs, dtype=torch.long, device=device)

    return AttentionGroup(
        token_index=t(token_index),
        query_index=t(query_index),
        block_ids=t(padded),
        mask=mask.unsqueeze(1).to(device),
        batch_size=b,
        max_q=max_q,
        max_blocks=max_blocks,
    )


class PagedAttention:
    """AttentionBackend that stores/reads K/V through block tables."""

    def __init__(self, cache: PagedKVCache, meta: PagedAttentionMetadata) -> None:
        self.cache = cache
        self.meta = meta

    def _gather(self, pool: Tensor, g: AttentionGroup) -> Tensor:
        """[num_slots, n_kv, d] pool -> [B_g, n_kv, S, d] context of every sequence in the group.

        Copies whole blocks with index_select on a [num_blocks, block_size*n_kv*d]
        view. (On MPS this is ~6x faster than advanced indexing `pool[slot_ids]`,
        measured in benchmarks/micro_gather.py.)
        """
        n_kv, d = pool.shape[1], pool.shape[2]
        blocks = pool.view(self.cache.num_blocks, -1).index_select(0, g.block_ids)
        return blocks.view(g.batch_size, g.max_blocks * self.cache.block_size, n_kv, d).transpose(1, 2)

    def _attend(self, g: AttentionGroup, q: Tensor, k_pool: Tensor, v_pool: Tensor) -> Tensor:
        """q: [T_g, H, d] (this group's rows, packed) -> [T_g, H, d]."""
        n_heads, d = q.shape[1], q.shape[2]
        keys, vals = self._gather(k_pool, g), self._gather(v_pool, g)
        if g.max_q == 1:  # decode group: packed layout already is the padded layout
            q_pad = q.view(g.batch_size, 1, n_heads, d)
        else:
            q_pad = q.new_zeros(g.batch_size * g.max_q, n_heads, d).index_copy_(0, g.query_index, q)
            q_pad = q_pad.view(g.batch_size, g.max_q, n_heads, d)
        out = sdpa(q_pad.transpose(1, 2), keys, vals, g.mask)  # [B_g, H, max_q, d]
        out = out.transpose(1, 2).reshape(g.batch_size * g.max_q, n_heads, d)
        return out if g.max_q == 1 else out.index_select(0, g.query_index)

    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        m = self.meta
        k_pool, v_pool = self.cache.k[layer_idx], self.cache.v[layer_idx]
        # 1. write new K/V into their slots
        k_pool.index_copy_(0, m.slot_mapping, k)
        v_pool.index_copy_(0, m.slot_mapping, v)
        # 2./3. per group: gather blocks, pad queries, one SDPA call
        if len(m.groups) == 1:  # all-decode or all-prefill step: no regrouping needed
            return self._attend(m.groups[0], q, k_pool, v_pool)
        out = torch.empty_like(q)
        for g in m.groups:
            out.index_copy_(0, g.token_index, self._attend(g, q.index_select(0, g.token_index), k_pool, v_pool))
        return out
