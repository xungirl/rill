"""Weight-only int8 quantization (W8A16) for the decoder's linear layers.

Each weight matrix W [out, in] is stored as int8 with one fp16 scale per output
channel (symmetric, per-channel):
    scale[o] = max_i |W[o, i]| / 127
    Wq[o, i] = round(W[o, i] / scale[o])  in [-127, 127]
    y = x @ (Wq * scale).T   computed by torch._weight_int8pack_mm, which reads
                             int8 weights and dequantizes inside the kernel.
Activations stay fp16. Decode at small batch is weight-bandwidth bound, so
halving the bytes read per step is where any speed-up comes from.

Embeddings (tied with the LM head), norms and the KV cache are left in fp16.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ember.model import Qwen3Model


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.detach()
        scale = w.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
        wq = torch.round(w.float() / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_int8: Tensor
        self.scale: Tensor
        self.register_buffer("weight_int8", wq.contiguous())
        self.register_buffer("scale", scale.to(w.dtype))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        y = torch._weight_int8pack_mm(x.reshape(-1, self.in_features), self.weight_int8, self.scale)
        return y.view(*shape[:-1], self.out_features)

    def dequantized(self) -> Tensor:
        return self.weight_int8.to(self.scale.dtype) * self.scale.unsqueeze(1)


def quantize_int8(model: Qwen3Model) -> Qwen3Model:
    """Replace every decoder projection with Int8Linear, in place. Returns the model."""
    for layer in model.layers:
        for parent, name in [
            (layer.self_attn, "qkv_proj"),
            (layer.self_attn, "o_proj"),
            (layer.mlp, "gate_up_proj"),
            (layer.mlp, "down_proj"),
        ]:
            linear = getattr(parent, name)
            assert isinstance(linear, nn.Linear)
            setattr(parent, name, Int8Linear(linear))
    return model


def weight_bytes(model: nn.Module) -> int:
    """Bytes of parameters + persistent buffers (i.e. what is loaded from the checkpoint)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for n, b in model.named_buffers() if "rope" not in n)
    return total
