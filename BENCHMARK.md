# ember benchmarks

All numbers below were measured on **one laptop**: Apple M2 (8 CPU cores, 24 GB unified memory), macOS,
Python 3.15.2, PyTorch 2.15.0 (MPS backend). Everything runs locally; servers are on localhost.
Raw JSON for every table is in `results/`. Each JSON records the machine, date and the 1-minute load average
before/after each run: another process was using the CPU during this work, so key runs wait until
`load1 <= 4` (`benchmarks/common.py: wait_for_quiet`) and are repeated 3x.

## Stage 1 — single request, from-scratch model

### Parity vs Hugging Face transformers
Command: `uv run python -m benchmarks.parity_report` → `results/stage1_parity.json` (2026-11-14)

| check | result |
|---|---|
| per-layer hidden-state max abs diff, 28 layers, fp32 CPU | 0.0 for every layer |
| logits max abs diff, fp32 CPU / bf16 CPU | 0.0 / 0.0 |
| greedy 64 tokens, 8 prompts: ours fp32 CPU vs HF fp32 CPU | 8/8 token-identical |
| ours **fp16 MPS** vs HF fp32 CPU | 8/8 token-identical |
| ours bf16 MPS vs HF fp32 CPU | 3/8 identical (first divergence at token 1–57) |
| ours bf16 MPS vs HF bf16 MPS | 6/8 identical |
| ours fp16 MPS vs HF fp16 MPS | 8/8 identical |

Take-away: the implementation is exact (bitwise on CPU); bf16's 8-bit mantissa flips near-tie argmaxes, so
**fp16 is the default on MPS**.

### Single-request speed (contiguous KV cache, greedy, 29-token chat prompt, 128 new tokens)
Command: `uv run python -m benchmarks.bench_single` → `results/stage1_single_request.json`
(1 warmup + 3 runs per config; load1 at start 2.9–4.0). An earlier run with some CPU rows at load1 > 4 is kept
as `results/stage1_single_request.run1.json` and not used.

| device / dtype | decode tok/s (mean ± sd, n=3) | prefill + first token, ms (mean ± sd) |
|---|---|---|
| MPS / fp16 | **45.9 ± 0.8** | 38.0 ± 4.5 |
| MPS / bf16 | 43.2 ± 1.2 | 61.3 ± 25.7 |
| CPU / bf16 | 35.3 ± 0.3 | 269.6 ± 26.8 |
| CPU / fp32 | 18.6 ± 0.0 | 146.6 ± 0.5 |

MPS wins clearly on prefill (compute-bound) but only modestly on batch-1 decode: at batch 1 the per-step cost is
dominated by kernel-launch / Python overhead of eager PyTorch, not by memory bandwidth. That is the motivation
for batching (Stage 3).
