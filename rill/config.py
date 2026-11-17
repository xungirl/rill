"""Model hyper-parameters.

`ModelConfig` mirrors the subset of a Hugging Face Qwen3 `config.json` that the
forward pass actually needs. Keeping it as a plain frozen dataclass (instead of
importing `transformers.PretrainedConfig`) means the engine has zero runtime
dependency on `transformers`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    eos_token_ids: tuple[int, ...] = ()

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def kv_bytes_per_token(self, dtype_bytes: int) -> int:
        """Bytes of K+V cache one token occupies across all layers."""
        return 2 * self.num_hidden_layers * self.kv_size * dtype_bytes

    @classmethod
    def from_pretrained_dir(cls, path: str | Path) -> ModelConfig:
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())
        eos: list[int] = []
        gen_path = path / "generation_config.json"
        if gen_path.exists():
            gen_eos = json.loads(gen_path.read_text()).get("eos_token_id", [])
            eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
        if not eos:
            cfg_eos = raw.get("eos_token_id", [])
            eos = cfg_eos if isinstance(cfg_eos, list) else [cfg_eos]
        if raw.get("rope_scaling"):
            raise NotImplementedError("rope_scaling is not supported")
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            eos_token_ids=tuple(eos),
        )


def tiny_config(vocab_size: int = 512) -> ModelConfig:
    """A miniature Qwen3-shaped config for fast CPU unit tests (random weights)."""
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        eos_token_ids=(vocab_size - 1,),
    )
