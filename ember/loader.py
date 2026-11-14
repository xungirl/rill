"""Load Qwen3 weights straight from a `.safetensors` checkpoint.

No `transformers` involved: we resolve the local Hugging Face cache directory
by hand, read tensors with `safetensors.safe_open`, and copy them into our own
modules. The only renaming needed is fusing q/k/v -> qkv_proj and
gate/up -> gate_up_proj.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors import safe_open

from ember.config import ModelConfig
from ember.model import Qwen3Model, RotaryEmbedding

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def resolve_model_dir(name_or_path: str = DEFAULT_MODEL) -> Path:
    """Return a directory containing config.json + *.safetensors.

    Accepts a local path, or a repo id already present in the HF hub cache
    (~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{rev}).
    Never downloads anything.
    """
    p = Path(name_or_path).expanduser()
    if (p / "config.json").exists():
        return p
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_dir = hub / ("models--" + name_or_path.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        snap = repo_dir / "snapshots" / ref.read_text().strip()
        if (snap / "config.json").exists():
            return snap
    raise FileNotFoundError(f"model {name_or_path!r} not found locally (looked in {repo_dir})")


def _checkpoint_name_map(cfg: ModelConfig) -> dict[str, list[str]]:
    """Our parameter name -> list of checkpoint tensor names concatenated on dim 0."""
    mapping: dict[str, list[str]] = {
        "embed_tokens.weight": ["model.embed_tokens.weight"],
        "norm.weight": ["model.norm.weight"],
    }
    if not cfg.tie_word_embeddings:
        mapping["lm_head.weight"] = ["lm_head.weight"]
    for i in range(cfg.num_hidden_layers):
        src = f"model.layers.{i}."
        dst = f"layers.{i}."
        mapping[dst + "input_layernorm.weight"] = [src + "input_layernorm.weight"]
        mapping[dst + "post_attention_layernorm.weight"] = [src + "post_attention_layernorm.weight"]
        mapping[dst + "self_attn.qkv_proj.weight"] = [src + f"self_attn.{n}_proj.weight" for n in "qkv"]
        mapping[dst + "self_attn.o_proj.weight"] = [src + "self_attn.o_proj.weight"]
        mapping[dst + "self_attn.q_norm.weight"] = [src + "self_attn.q_norm.weight"]
        mapping[dst + "self_attn.k_norm.weight"] = [src + "self_attn.k_norm.weight"]
        mapping[dst + "mlp.gate_up_proj.weight"] = [src + "mlp.gate_proj.weight", src + "mlp.up_proj.weight"]
        mapping[dst + "mlp.down_proj.weight"] = [src + "mlp.down_proj.weight"]
    return mapping


def load_model(
    name_or_path: str = DEFAULT_MODEL,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3Model:
    model_dir = resolve_model_dir(name_or_path)
    cfg = ModelConfig.from_pretrained_dir(model_dir)
    with torch.device("meta"):
        model = Qwen3Model(cfg)  # no memory allocated yet
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in files]
    where = {k: h for h in handles for k in h.keys()}  # noqa: SIM118 - safe_open has no __iter__

    state: dict[str, torch.Tensor] = {}
    for ours, theirs in _checkpoint_name_map(cfg).items():
        parts = [where[name].get_tensor(name) for name in theirs]
        state[ours] = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=False, assign=True)  # assign: replace meta params
    _materialize_buffers(model, device)
    _check_no_meta(model)
    return model.eval()


def random_model(
    cfg: ModelConfig,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Qwen3Model:
    """Random-init model for unit tests / CI (no checkpoint needed)."""
    gen = torch.Generator().manual_seed(seed)
    model = Qwen3Model(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))
    return model.to(device=device, dtype=dtype).eval()


def _materialize_buffers(model: Qwen3Model, device: str | torch.device) -> None:
    """RoPE tables were created on 'meta'; rebuild them for real on `device`."""

    cfg = model.cfg
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta).to(device)
    model.rope = rope
    for layer in model.layers:
        layer.self_attn.rope = rope  # type: ignore[union-attr]


def _check_no_meta(model: Qwen3Model) -> None:
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise RuntimeError(f"weights missing from checkpoint: {left[:5]}")
