"""Stage 1 parity report: our Qwen3 forward vs Hugging Face transformers.

Writes results/stage1_parity.json with
  * per-layer max |ours - HF| of hidden states (fp32 and bf16, CPU),
  * max |logits diff|,
  * greedy-decoding token agreement on a fixed prompt set.

Run: uv run python -m benchmarks.parity_report
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from benchmarks.common import write_result
from rill.attention import NoCacheAttention
from rill.loader import load_model, resolve_model_dir
from rill.simple_generate import generate_greedy
from rill.tokenizer import Tokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain what a KV cache is in one sentence.",
    "1, 2, 3, 5, 8, 13,",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
    "SELECT name FROM users WHERE",
    "Translate to French: The weather is nice today.",
]
NEW_TOKENS = 64


@torch.no_grad()
def layer_diffs(dtype: torch.dtype, tok: Tokenizer) -> dict[str, object]:
    ours = load_model(device="cpu", dtype=dtype)
    hf: Any = AutoModelForCausalLM.from_pretrained(str(resolve_model_dir()), dtype=dtype).eval()
    ids = torch.tensor(tok.encode(PROMPTS[2]))
    final, layers = ours(ids, torch.arange(len(ids)), NoCacheAttention(), return_layer_outputs=True)
    ref = hf(ids[None], output_hidden_states=True)
    per_layer = [float((h - ref.hidden_states[i + 1][0]).abs().max()) for i, h in enumerate(layers[:-1])]
    per_layer.append(float((final - ref.hidden_states[-1][0]).abs().max()))
    logits = ours.compute_logits(final).float()
    return {
        "prompt_tokens": len(ids),
        "per_layer_max_abs_diff": per_layer,
        "logits_max_abs_diff": float((logits - ref.logits[0].float()).abs().max()),
        "argmax_agree_all_positions": bool((logits.argmax(-1) == ref.logits[0].argmax(-1)).all()),
    }


@torch.no_grad()
def greedy_agreement(
    tok: Tokenizer,
    ours_device: str,
    ours_dtype: torch.dtype,
    ref_device: str = "cpu",
    ref_dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    hf: Any = AutoModelForCausalLM.from_pretrained(str(resolve_model_dir()), dtype=ref_dtype)
    hf = hf.to(ref_device).eval()
    ours = load_model(device=ours_device, dtype=ours_dtype)
    rows = []
    for p in PROMPTS:
        ids = tok.encode(p)
        a = generate_greedy(ours, ids, NEW_TOKENS)
        b = hf.generate(
            torch.tensor([ids], device=ref_device),
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,
            do_sample=False,
            eos_token_id=None,
        )[0, len(ids) :].tolist()
        first_div = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
        rows.append({"prompt": p, "identical": a == b, "first_divergence": first_div})
    return {
        "ours": f"{ours_device}/{ours_dtype}",
        "reference": f"HF transformers {ref_device}/{ref_dtype}",
        "new_tokens": NEW_TOKENS,
        "identical": sum(r["identical"] for r in rows),
        "total": len(rows),
        "rows": rows,
    }


def main() -> None:
    tok = Tokenizer(resolve_model_dir())
    out = {
        "fp32_cpu": layer_diffs(torch.float32, tok),
        "bf16_cpu": layer_diffs(torch.bfloat16, tok),
        "greedy_fp32_cpu": greedy_agreement(tok, "cpu", torch.float32),
        "greedy_bf16_mps": greedy_agreement(tok, "mps", torch.bfloat16),
        "greedy_fp16_mps": greedy_agreement(tok, "mps", torch.float16),
        # Same device + dtype on both sides isolates implementation error from precision loss.
        "greedy_bf16_mps_vs_hf_bf16_mps": greedy_agreement(tok, "mps", torch.bfloat16, "mps", torch.bfloat16),
        "greedy_fp16_mps_vs_hf_fp16_mps": greedy_agreement(tok, "mps", torch.float16, "mps", torch.float16),
    }
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("rows", "per_layer_max_abs_diff")})
    write_result("stage1_parity", out)


if __name__ == "__main__":
    main()
