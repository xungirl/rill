# ember

A from-scratch LLM inference engine for Qwen3, written in plain PyTorch: own model code, paged KV cache,
continuous batching (work in progress, see Project Journey). Runs on an Apple-silicon laptop (MPS) or CPU.

## Project Journey

### 1. Model from scratch
- **Goal**: a Qwen3 forward pass that loads the official `safetensors` checkpoint with no `transformers` code.
- **Decision & alternatives**: write each block (RMSNorm, RoPE, GQA attention with Qwen3's per-head q/k norm,
  SwiGLU, tied embeddings) as a small `nn.Module`; tensors use a *packed token* layout `[T, ...]` so a batch can
  mix sequences without padding. Attention goes through an `AttentionBackend` protocol so the KV-cache layout can
  change later without touching the model. Alternative was wrapping HF modules — rejected: it hides exactly the
  parts an inference engine must control. q/k/v and gate/up matrices are fused at load time (fewer kernels).
- **Built**: `ember/model.py`, `ember/attention.py`, `ember/loader.py`, `ember/tokenizer.py`, `ember/simple_generate.py`.
- **Verified**: per-layer hidden states and logits identical to HF (max abs diff 0.0, fp32 and bf16 on CPU);
  greedy decoding token-identical to `transformers.generate` on 8 prompts × 64 tokens (fp32 CPU, fp16 MPS).
- **Measured**: single-request decode 45.9 tok/s on MPS fp16 vs 18.6 tok/s CPU fp32 (35.3 CPU bf16).
- **Problems / fixes**: bf16 on MPS diverged from the reference after 1–57 tokens on 5/8 prompts (even HF's own
  bf16 MPS run disagrees with ours on 2/8): bf16's 8-bit mantissa flips near-tie argmaxes. fp16 matched 8/8 → fp16
  is the MPS default.

### Key metrics

| metric | value | how measured | source |
|---|---|---|---|
| logits / per-layer max abs diff vs HF (fp32, 28 layers) | 0.0 | `benchmarks/parity_report.py` | `results/stage1_parity.json` |
| greedy token parity vs HF, fp16 MPS | 8/8 prompts × 64 tokens | same | `results/stage1_parity.json` |
| single-request decode, MPS fp16 | 45.9 ± 0.8 tok/s (n=3) | `benchmarks/bench_single.py` | `results/stage1_single_request.json` |
| single-request decode, CPU fp32 | 18.6 ± 0.0 tok/s (n=3) | same | `results/stage1_single_request.json` |

### Next steps
Paged KV cache (Stage 2) → continuous batching (3) → prefix caching (4) → OpenAI-compatible server (5) →
load testing (6) → sampling & int8 (7) → ops polish (8).

## How to run

```bash
uv sync                                  # Python 3.15, torch 2.15
uv run pytest -q                         # unit tests (tiny random model) + real-model parity tests
uv run python -m benchmarks.parity_report
uv run python -m benchmarks.bench_single
```
The real-model tests/benchmarks expect `Qwen/Qwen3-0.6B` in the local Hugging Face cache; nothing is downloaded.

## Limitations
- Only the Qwen3 dense architecture (no MoE, no rope scaling / sliding window).
- Pure PyTorch eager; no custom kernels, no CUDA graphs. Batch-1 decode is launch-overhead bound.
