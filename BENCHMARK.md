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

## Stage 2 — paged KV cache: memory utilization

Command: `uv run python -m benchmarks.bench_memory` → `results/stage2_memory.json` (2026-11-14, load1 2.5 → 1.9)

Workload: the 64 hand-written chat prompts in `benchmarks/data/prompts.txt` (17–34 prompt tokens, mean 21.8), run as
one static batch through the paged runner on MPS fp16, greedy, `max_tokens=256`, stop at EOS. Real outputs:
37/64 stopped at EOS, output length 11–256 (mean 188.3). At every one of the 256 steps we record each live
sequence's length and compare how many KV slots each policy would hold for exactly those sequences.

| policy | slots reserved per request | utilization (avg over steps) | utilization (at final lengths) | requests that fit in 2 GiB of fp16 KV (18,724 tokens)\* |
|---|---|---|---|---|
| naive, reserve `max_model_len=2048` | 2048 | 7.3% | 10.3% | 9 |
| naive, reserve `prompt + max_tokens` | prompt + 256 | 53.7% | 75.6% | 67 |
| paged, block 8 | ⌈len/8⌉·8 | 96.7% | 98.5% | 87 |
| **paged, block 16 (default)** | ⌈len/16⌉·16 | **93.3%** | **95.8%** | **85** |
| paged, block 32 | ⌈len/32⌉·32 | 87.9% | 93.6% | 83 |

\* `budget_tokens × num_requests / total reserved slots at final length`; i.e. how many requests of this length mix
fit if all are at their final length at once. The naive policies must reserve up front; paging only holds what has
been written, so its waste is the unfilled tail of the last block (< block_size tokens per sequence) and there is
no external fragmentation. Smaller blocks waste less but mean longer block tables and more gather indices; 16 is
the default (as in vLLM).
