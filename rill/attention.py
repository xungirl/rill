"""Attention backends: the only place that knows how K/V are stored.

A backend receives the current layer's freshly projected q/k/v for every token
in the packed batch, stores k/v wherever its cache lives, and returns the
attention output. The model never touches the cache directly.

Backends in this file are the simple ones (stage 1):
  * `NoCacheAttention`     - one sequence, no cache (parity tests, scoring).
  * `ContiguousKVCache`    - one sequence, a preallocated [max_len] buffer per
                             layer. This is the "naive" cache that the paged
                             cache (rill/paged_attention.py) is compared to.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class AttentionBackend(Protocol):
    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """q: [T, n_heads, d], k/v: [T, n_kv_heads, d] -> [T, n_heads, d]."""
        ...


def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, causal: bool = False) -> Tensor:
    """Batched scaled-dot-product attention with grouped-query heads.

    q: [B, n_heads, Tq, d]; k/v: [B, n_kv_heads, Tk, d]; mask: bool [B, 1, Tq, Tk]
    (True = may attend). `enable_gqa` lets each KV head serve n_heads/n_kv_heads
    query heads without materialising repeated K/V.
    """
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=True)


class NoCacheAttention:
    """Causal self-attention over a single sequence of T tokens, nothing cached."""

    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # [T, H, d] -> [1, H, T, d]
        out = sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], None, causal=True)
        return out[0].transpose(0, 1)


class ContiguousKVCache:
    """Per-layer K/V buffers of shape [max_len, n_kv_heads, d] for ONE sequence.

    The whole max_len is reserved up front whether or not the sequence ever
    gets that long - exactly the waste that paging removes.
    Usage: set `start_pos` to the position of the first token of this forward
    call, run the model, then advance `start_pos` by the number of tokens.
    """

    def __init__(
        self,
        num_layers: int,
        max_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        shape = (num_layers, max_len, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.start_pos = 0

    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        t = q.shape[0]
        end = self.start_pos + t
        if end > self.max_len:
            raise ValueError(f"sequence length {end} exceeds cache size {self.max_len}")
        self.k[layer_idx, self.start_pos : end] = k
        self.v[layer_idx, self.start_pos : end] = v
        keys = self.k[layer_idx, :end].transpose(0, 1)[None]  # [1, kvH, end, d]
        vals = self.v[layer_idx, :end].transpose(0, 1)[None]
        query = q.transpose(0, 1)[None]  # [1, H, t, d]
        if t == 1:
            mask = None  # a single new token may attend to everything before it
        else:
            # query i sits at absolute position start_pos + i; it sees keys j <= that.
            q_pos = torch.arange(self.start_pos, end, device=q.device)[:, None]
            k_pos = torch.arange(end, device=q.device)[None, :]
            mask = (k_pos <= q_pos)[None, None]
        return sdpa(query, keys, vals, mask)[0].transpose(0, 1)
