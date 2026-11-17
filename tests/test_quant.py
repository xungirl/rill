"""Weight-only int8 quantization."""

from __future__ import annotations

import copy

import torch
from torch import nn

from rill.attention import NoCacheAttention
from rill.model import Qwen3Model
from rill.quant import Int8Linear, quantize_int8, weight_bytes


def test_int8_linear_roundtrip_error_is_bounded() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False)
    q = Int8Linear(lin)
    assert q.weight_int8.dtype == torch.int8
    # symmetric per-channel rounding: |W - Wq*s| <= s/2 elementwise
    err = (lin.weight - q.dequantized()).abs()
    assert torch.all(err <= q.scale.unsqueeze(1) / 2 + 1e-7)
    x = torch.randn(5, 64)
    torch.testing.assert_close(q(x), x @ q.dequantized().T, atol=1e-4, rtol=1e-4)


def test_quantized_model_stays_close_and_shrinks(tiny_model: Qwen3Model) -> None:
    qmodel = quantize_int8(copy.deepcopy(tiny_model))
    ids = torch.randint(0, tiny_model.cfg.vocab_size, (16,))
    pos = torch.arange(16)
    with torch.no_grad():
        ref = tiny_model.compute_logits(tiny_model(ids, pos, NoCacheAttention())[0])
        got = qmodel.compute_logits(qmodel(ids, pos, NoCacheAttention())[0])
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.05, rel
    n_int8 = sum(isinstance(m, Int8Linear) for m in qmodel.modules())
    assert n_int8 == 4 * tiny_model.cfg.num_hidden_layers
    assert weight_bytes(qmodel) < weight_bytes(tiny_model)
