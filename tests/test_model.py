"""Unit tests for the from-scratch Qwen3 building blocks (tiny random model, CPU)."""

from __future__ import annotations

import torch

from rill.attention import ContiguousKVCache, NoCacheAttention
from rill.config import ModelConfig
from rill.model import Qwen3Model, RMSNorm, RotaryEmbedding
from rill.simple_generate import generate_greedy


def test_rmsnorm_matches_formula() -> None:
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(1.0, 9.0))
    x = torch.randn(3, 8)
    expected = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected)


def test_rope_scores_depend_only_on_relative_distance() -> None:
    rope = RotaryEmbedding(head_dim=16, max_positions=256, theta=10000.0)
    q = torch.randn(1, 1, 16)
    k = torch.randn(1, 1, 16)

    def score(pq: int, pk: int) -> float:
        rq, _ = rope(torch.tensor([pq]), q, k)
        _, rk = rope(torch.tensor([pk]), q, k)
        return float((rq * rk).sum())

    assert abs(score(10, 3) - score(110, 103)) < 1e-4
    assert abs(score(10, 3) - score(10, 4)) > 1e-6  # and it does depend on distance


def test_rope_position_zero_is_identity() -> None:
    rope = RotaryEmbedding(head_dim=16, max_positions=8, theta=10000.0)
    q, k = torch.randn(1, 4, 16), torch.randn(1, 2, 16)
    rq, rk = rope(torch.tensor([0]), q, k)
    torch.testing.assert_close(rq, q)
    torch.testing.assert_close(rk, k)


@torch.no_grad()
def test_incremental_decode_matches_full_forward(tiny_model: Qwen3Model) -> None:
    """Prefill + token-by-token decode through the KV cache must give the same
    logits as one causal forward over the whole sequence."""
    cfg = tiny_model.cfg
    ids = torch.randint(0, cfg.vocab_size, (20,))
    full, _ = tiny_model(ids, torch.arange(20), NoCacheAttention())
    full_logits = tiny_model.compute_logits(full)

    cache = ContiguousKVCache(cfg.num_hidden_layers, 32, cfg.num_key_value_heads, cfg.head_dim, torch.float32, "cpu")
    pieces = []
    for chunk in (ids[:7], ids[7:8], ids[8:15], *torch.split(ids[15:], 1)):
        pos = torch.arange(cache.start_pos, cache.start_pos + len(chunk))
        h, _ = tiny_model(chunk, pos, cache)
        cache.start_pos += len(chunk)
        pieces.append(tiny_model.compute_logits(h))
    torch.testing.assert_close(torch.cat(pieces), full_logits, atol=1e-5, rtol=1e-5)


def test_greedy_is_deterministic_and_respects_stop(tiny_model: Qwen3Model) -> None:
    prompt = [1, 2, 3, 4, 5]
    a = generate_greedy(tiny_model, prompt, 16)
    b = generate_greedy(tiny_model, prompt, 16)
    assert a == b and len(a) == 16
    stop = a[3]
    c = generate_greedy(tiny_model, prompt, 16, stop_ids=(stop,))
    assert c == a[: a.index(stop) + 1]


def test_tied_embeddings_share_storage(tiny_model: Qwen3Model, tiny_cfg: ModelConfig) -> None:
    assert tiny_cfg.tie_word_embeddings
    assert not hasattr(tiny_model, "lm_head")
    h = torch.randn(2, tiny_cfg.hidden_size)
    torch.testing.assert_close(tiny_model.compute_logits(h), h @ tiny_model.embed_tokens.weight.T)


def test_kv_bytes_per_token_for_qwen3_0_6b_shape() -> None:
    # 2 (K,V) * 28 layers * 8 kv-heads * 128 dim * 2 bytes = 114,688 B = 112 KiB
    cfg = ModelConfig(151936, 1024, 3072, 28, 16, 8, 128)
    assert cfg.kv_bytes_per_token(2) == 114_688
