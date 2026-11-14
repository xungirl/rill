"""Single-request greedy generation with a contiguous KV cache (stage 1 path).

This is the simplest correct decoding loop: one prefill forward over the whole
prompt, then one forward per generated token, each reusing the cached K/V of
all previous tokens. The batched engine (ember/engine.py) must produce exactly
the same tokens; tests compare against this function.
"""

from __future__ import annotations

import torch

from ember.attention import ContiguousKVCache
from ember.model import Qwen3Model


@torch.inference_mode()
def generate_greedy(
    model: Qwen3Model,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: tuple[int, ...] = (),
) -> list[int]:
    cfg = model.cfg
    device = model.embed_tokens.weight.device
    dtype = model.embed_tokens.weight.dtype
    cache = ContiguousKVCache(
        cfg.num_hidden_layers,
        len(prompt_ids) + max_new_tokens,
        cfg.num_key_value_heads,
        cfg.head_dim,
        dtype,
        device,
    )
    out: list[int] = []
    tokens = torch.tensor(prompt_ids, device=device)
    for _ in range(max_new_tokens):
        positions = torch.arange(cache.start_pos, cache.start_pos + len(tokens), device=device)
        hidden, _ = model(tokens, positions, cache)
        cache.start_pos += len(tokens)
        next_id = int(model.compute_logits(hidden[-1:]).argmax(-1).item())
        out.append(next_id)
        if next_id in stop_ids:
            break
        tokens = torch.tensor([next_id], device=device)
    return out
