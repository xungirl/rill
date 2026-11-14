"""Parity against the Hugging Face `transformers` reference implementation.

`transformers` is used ONLY here, as an oracle. Marked `real_model`: needs the
Qwen/Qwen3-0.6B checkpoint in the local HF cache.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest
import torch

from ember.attention import NoCacheAttention
from ember.loader import load_model, resolve_model_dir
from ember.model import Qwen3Model
from ember.simple_generate import generate_greedy
from ember.tokenizer import Tokenizer

pytestmark = pytest.mark.real_model

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain what a KV cache is in one sentence.",
    "1, 2, 3, 5, 8, 13,",
    "Once upon a time, in a small village by the sea,",
]


@cache
def _ours() -> Qwen3Model:
    return load_model(device="cpu", dtype=torch.float32)


@cache
def _hf() -> Any:  # transformers model, used only as an oracle
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(resolve_model_dir()), dtype=torch.float32).eval()


@cache
def _tok() -> Tokenizer:
    return Tokenizer(resolve_model_dir())


def test_tokenizer_and_chat_template_match_hf() -> None:
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(str(resolve_model_dir()))
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi! 你好 🙂"}]
    ours = _tok().apply_chat_template(msgs, enable_thinking=False)
    theirs = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert ours == theirs
    for p in PROMPTS:
        assert _tok().encode(p) == hf_tok.encode(p, add_special_tokens=False)


@torch.no_grad()
@pytest.mark.parametrize("prompt", PROMPTS[:2])
def test_per_layer_hidden_states_and_logits_match_hf_fp32(prompt: str) -> None:
    ids = torch.tensor(_tok().encode(prompt))
    final, layers = _ours()(ids, torch.arange(len(ids)), NoCacheAttention(), return_layer_outputs=True)
    ref = _hf()(ids[None], output_hidden_states=True)
    # HF hidden_states = (embeddings, layer_1, ..., layer_{L-1}, norm(layer_L))
    for i, ours_h in enumerate(layers[:-1]):
        assert (ours_h - ref.hidden_states[i + 1][0]).abs().max().item() < 1e-3, f"layer {i}"
    assert (final - ref.hidden_states[-1][0]).abs().max().item() < 1e-3
    logits = _ours().compute_logits(final)
    assert (logits - ref.logits[0]).abs().max().item() < 1e-3


@torch.no_grad()
def test_greedy_decoding_token_identical_to_hf_generate() -> None:
    hf = _hf()
    for p in PROMPTS:
        ids = _tok().encode(p)
        ours = generate_greedy(_ours(), ids, max_new_tokens=24)
        ref = hf.generate(
            torch.tensor([ids]), max_new_tokens=24, do_sample=False, eos_token_id=None, min_new_tokens=24
        )[0, len(ids) :].tolist()
        assert ours == ref, p
