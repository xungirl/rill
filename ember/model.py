"""Qwen3 decoder-only transformer, written from scratch in plain PyTorch.

Shapes use a *flattened token* layout: every tensor that flows between layers
is `[T, ...]`, where T is the total number of tokens in the batch across all
sequences (a "packed" or "varlen" batch, like vLLM). Only attention needs to
know which token belongs to which sequence, and that knowledge lives entirely
in the `AttentionBackend` passed to `forward`. This keeps the model itself
agnostic of how the KV cache is laid out (none / contiguous / paged).

Per layer:
    x = x + Attn(RMSNorm(x))        # GQA, per-head q/k RMSNorm, RoPE
    x = x + MLP(RMSNorm(x))         # SwiGLU
Logits = RMSNorm(x) @ embed_tokens.T  (tied embeddings)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ember.attention import AttentionBackend
from ember.config import ModelConfig


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight, computed in float32 (as HF does)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(in_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE), "rotate-half" (non-interleaved) variant.

    cos/sin tables for every position are precomputed once in float32 and
    indexed by absolute position, so a decode step at position 1000 costs the
    same as one at position 0.
    """

    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [P, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [P, d]
        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, positions: Tensor, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # q: [T, n_heads, d], k: [T, n_kv_heads, d]; cos/sin broadcast over heads.
        cos = self.cos[positions].unsqueeze(1).to(q.dtype)
        sin = self.sin[positions].unsqueeze(1).to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class Attention(nn.Module):
    """Grouped-query attention with Qwen3's per-head RMSNorm on q and k.

    q/k/v projections are fused into one matmul (`qkv_proj`); the loader
    concatenates the three checkpoint matrices at load time. Fewer, larger
    matmuls means fewer kernel launches, which matters on MPS.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = cfg.q_size
        self.kv_size = cfg.kv_size
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.rope = rope

    def forward(self, x: Tensor, positions: Tensor, backend: AttentionBackend) -> Tensor:
        t = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(t, self.n_heads, self.head_dim))
        k = self.k_norm(k.view(t, self.n_kv_heads, self.head_dim))
        v = v.view(t, self.n_kv_heads, self.head_dim)
        q, k = self.rope(positions, q, k)
        out = backend.forward(self.layer_idx, q, k, v)  # [T, n_heads, d]
        return self.o_proj(out.reshape(t, self.q_size))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)); gate and up fused into one matmul."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx, rope)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, positions: Tensor, backend: AttentionBackend) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, backend)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i, rope) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Shared by all layers; registered once here so .to(device) moves it.
        self.rope = rope

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        backend: AttentionBackend,
        return_layer_outputs: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Run all decoder layers on a packed batch of tokens.

        Args:
            input_ids: [T] token ids (all sequences concatenated).
            positions: [T] absolute position of each token within its sequence.
            backend: attention implementation that owns the KV cache.
        Returns:
            (final-normed hidden states [T, H], per-layer outputs if requested)
        """
        x = self.embed_tokens(input_ids)
        layer_outputs: list[Tensor] = []
        for layer in self.layers:
            x = layer(x, positions, backend)
            if return_layer_outputs:
                layer_outputs.append(x)
        return self.norm(x), layer_outputs

    def compute_logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits. Call only on rows you sample from:
        the [H, V] projection is the single largest matmul in a 0.6B model."""
        if self.cfg.tie_word_embeddings:
            return hidden @ self.embed_tokens.weight.T
        return self.lm_head(hidden)
