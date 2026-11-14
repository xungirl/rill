"""Paged KV cache storage and the attention backend that reads through block tables.

Storage: one K pool and one V pool per layer, shape
    [num_blocks * block_size, n_kv_heads, head_dim]
i.e. a flat array of token *slots*; block b owns slots [b*bs, (b+1)*bs).

A forward step processes a packed batch of T new tokens from B sequences
(any mix of prefill chunks and single decode tokens). Per layer we:
  1. scatter the new k/v into their slots           (slot_mapping  [T])
  2. gather each sequence's full context back out   (gather_slots  [B, S])
     into a padded [B, n_kv, S, d] tensor, S = max_blocks * bs
  3. run batched SDPA with a [B, 1, max_q, S] mask that encodes both
     "key exists" (j < context_len) and causality (j <= query position).

This is the same data layout vLLM uses, but step 2 materialises the gathered
K/V with ordinary PyTorch indexing instead of a fused paged-attention kernel
(there is no such kernel for MPS in PyTorch). That costs extra memory traffic
and is the main known performance gap; see README "Limitations".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ember.attention import sdpa
from ember.config import ModelConfig


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
class PagedAttentionMetadata:
    """Everything attention needs about the batch layout; built once per step,
    shared by all layers."""

    slot_mapping: Tensor  # [T] destination slot of each new token
    gather_slots: Tensor  # [B, S] slot of every (padded) context position, S = max_blocks*bs
    query_index: Tensor  # [T] position of each packed token in the padded [B*max_q] layout
    mask: Tensor  # [B, 1, max_q, S] bool
    batch_size: int
    max_q: int

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
        b = len(block_tables)
        max_q = max(query_lens)
        max_blocks = max(len(t) for t in block_tables)
        slots: list[int] = []
        qidx: list[int] = []
        for row, (table, q_len, ctx) in enumerate(zip(block_tables, query_lens, context_lens, strict=True)):
            for pos in range(ctx - q_len, ctx):
                slots.append(table[pos // block_size] * block_size + pos % block_size)
            qidx.extend(range(row * max_q, row * max_q + q_len))
        padded = torch.tensor([t + [0] * (max_blocks - len(t)) for t in block_tables], dtype=torch.long)
        gather_slots = (padded.unsqueeze(-1) * block_size + torch.arange(block_size)).flatten(1)

        # mask[b, 0, i, j]: query row i of seq b (absolute pos ctx-q_len+i) may see key j.
        ctx_t = torch.tensor(context_lens).view(b, 1, 1)
        qlen_t = torch.tensor(query_lens).view(b, 1, 1)
        i = torch.arange(max_q).view(1, max_q, 1)
        j = torch.arange(max_blocks * block_size).view(1, 1, -1)
        q_pos = ctx_t - qlen_t + i
        # Padding query rows (i >= q_len) still see key 0 (q_pos >= 0, ctx >= 1), so no
        # row is fully masked (which would give NaN); their outputs are dropped via query_index.
        mask = (j <= q_pos) & (j < ctx_t)
        return cls(
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
            gather_slots=gather_slots.to(device),
            query_index=torch.tensor(qidx, dtype=torch.long, device=device),
            mask=mask.unsqueeze(1).to(device),
            batch_size=b,
            max_q=max_q,
        )


class PagedAttention:
    """AttentionBackend that stores/reads K/V through block tables."""

    def __init__(self, cache: PagedKVCache, meta: PagedAttentionMetadata) -> None:
        self.cache = cache
        self.meta = meta

    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        m = self.meta
        k_pool, v_pool = self.cache.k[layer_idx], self.cache.v[layer_idx]
        # 1. write new K/V into their slots
        k_pool.index_copy_(0, m.slot_mapping, k)
        v_pool.index_copy_(0, m.slot_mapping, v)
        # 2. gather each sequence's blocks -> [B, max_blocks*bs, n_kv, d] -> [B, n_kv, S, d]
        keys = k_pool[m.gather_slots].transpose(1, 2)
        vals = v_pool[m.gather_slots].transpose(1, 2)
        # 3. scatter packed queries into a padded [B, H, max_q, d] batch, attend, unpack
        n_heads, d = q.shape[1], q.shape[2]
        q_pad = q.new_zeros(m.batch_size * m.max_q, n_heads, d)
        q_pad[m.query_index] = q
        q_pad = q_pad.view(m.batch_size, m.max_q, n_heads, d).transpose(1, 2)
        out = sdpa(q_pad, keys, vals, m.mask)  # [B, H, max_q, d]
        out = out.transpose(1, 2).reshape(m.batch_size * m.max_q, n_heads, d)
        return out[m.query_index]
