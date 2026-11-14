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

## Stage 3 — continuous batching

### Decode throughput vs batch size, and the gather fix
Command: `uv run python -m benchmarks.bench_decode_batch [--tag ...]` → `results/stage3_decode_batch.json`
(and `results/stage3_decode_batch_before_gather_fix.json`, same script run on the previous commit's
`paged_attention.py`). B identical 21-token prompts, 128 forced tokens each, all decoding in lock-step; 3 runs per B.

| batch | tok/s before fix | tok/s after fix | ms/step after |
|---|---|---|---|
| 1 | 39.5 | 42.6 | 23.5 |
| 4 | 114.7 | 137.4 | 29.1 |
| 8 | 183.3 | 246.2 | 32.5 |
| 16 | 232.2 | 335.4 | 47.7 |
| 32 | 311.8 | **517.8** | 61.8 |
| 64 | 366.3 | **670.4** | 95.5 |

The fix: `results/micro_gather.json` (`uv run python -m benchmarks.micro_gather`) — 56 K/V gathers of a batch-32
decode step take 81.0 ms with advanced indexing `pool[slot_ids]` vs 12.7 ms with block-level `index_select`
(bitwise-identical results). Batch-1 paged decode (42.6 tok/s) is ~7% below the contiguous-cache path of Stage 1
(45.9 tok/s): the price of metadata building and the gather.

### Sequential vs static vs continuous batching (offline)
Command: `uv run python -m benchmarks.bench_batching` → `results/stage3_batching.json` (3 repeats per mode,
load1 at start 1.4–2.5). Workload: 48 chat prompts (1,041 prompt tokens total), output lengths uniform[16, 256]
(seed 0) forced with `ignore_eos` → exactly 7,126 output tokens in every mode; all requests submitted at t=0;
token budget 2,048/step; MPS fp16. TTFT and e2e are measured from t=0, so they include queueing.

| mode | output tok/s (mean, cv) | req/s | TTFT p50 | e2e p50 | e2e p99 | mean TPOT | preemptions |
|---|---|---|---|---|---|---|---|
| sequential (1 at a time) | 41.4 (0.9%) | 0.28 | 84.71 s | 87.8 s | 171.06 s | 24.0 ms | 0 |
| static, B=8 | 157.9 (0.6%) | 1.06 | 18.93 s | 23.2 s | 44.98 s | 31.1 ms | 0 |
| continuous, B=8 | 219.6 (0.8%) | 1.48 | 13.30 s | 17.4 s | 32.39 s | 34.9 ms | 0 |
| static, B=16 | 239.8 (0.8%) | 1.61 | 10.33 s | 17.3 s | 29.72 s | 43.2 ms | 0 |
| continuous, B=16 | 273.4 (0.4%) | 1.84 | 6.97 s | 14.4 s | 26.00 s | 51.3 ms | 0 |
| static, B=32 | 308.2 (0.6%) | 2.08 | 0.40 s | 12.3 s | 23.12 s | 51.5 ms | 0 |
| **continuous, B=32** | **377.3 (0.0%)** | **2.54** | 0.40 s | 11.3 s | 18.87 s | 62.3 ms | 0 |
| continuous, B=32, KV pool 160 blocks | 278.6 (0.4%) | 1.88 | 0.39 s | 15.4 s | 25.45 s | 69.3 ms | 34 |

- Continuous vs static at the same concurrency: +39% (B=8), +14% (B=16), +22% (B=32) throughput, lower p99 e2e.
  Static batches idle the slots of finished short requests until the longest one ends.
- Continuous B=32 vs sequential: 9.1× throughput. TPOT rises with batch size (24 → 62 ms): the
  throughput/latency trade-off.
- With a deliberately tiny KV pool (2,560 token slots) the scheduler preempted 34 times and still finished every
  request, at −26% throughput vs the 2 GiB pool.
- **Batch invariance (honest caveat)**: on MPS fp16, 38–43 of 48 outputs are token-identical to the sequential
  run (identical across repeats of the same mode). Changing batch composition changes the fp16 matmul/SDPA shapes
  and hence rounding, which flips near-tie argmaxes. In fp32 on CPU, continuous batching with chunked prefill and
  preemption reproduces single-sequence greedy exactly (`tests/test_scheduler.py::test_real_model_continuous_batching_matches_reference_fp32`).

## Stage 4 — prefix caching

Command: `uv run python -m benchmarks.bench_prefix` → `results/stage4_prefix_cache.json` (3 repeats per cell,
fresh engine per run, load1 at start 1.3–2.2). Workload: a 587-token system prompt
(`benchmarks/data/system_prompt.txt`) + 32 different user questions → 613.9 prompt tokens on average; 32 forced
output tokens per request; MPS fp16; block size 16.

| scenario | metric | cache off | cache on | change |
|---|---|---|---|---|
| one request at a time | TTFT, requests 2–32 (mean) | 361.5 ms | **39.5 ms** | 9.2× lower |
| one request at a time | TTFT, all 32 (mean) | 362.2 ms | 50.2 ms | 7.2× lower |
| one request at a time | first request TTFT (cold, always a miss) | 383.1 ms | 383.4 ms | — |
| one request at a time | prompt tokens actually prefilled | 19,646 | 1,294 | hit rate 93% |
| 32 at t=0, continuous B=16 | output tok/s | 48.6 | **151.0** | 3.1× |
| 32 at t=0, continuous B=16 | TTFT mean (incl. queueing) | 9.57 s | 3.23 s | 3.0× lower |
| 32 at t=0, continuous B=16 | prompt tokens prefilled / hit rate | 19,646 / 0% | 3,070 / 84% | — |

All cells have cv ≤ 0.5%. Hit rate is lower in the batched case because the first 16 requests are admitted in the
same steps, before any block of the shared prefix has been written (blocks become shareable only after their K/V
exist), so several of them compute the prefix themselves.

Correctness: greedy outputs cache-on vs cache-off are identical for 29/32 (one-at-a-time) and 31/32 (batched)
requests on MPS fp16 — a hit computes only the suffix, i.e. different matmul shapes and fp16 rounding. On the tiny
fp32 model the outputs are identical in all cases, including a 40-block pool that forces LRU evictions
(`tests/test_prefix_cache.py::test_engine_outputs_identical_with_cache_on_and_off`).

## Stage 5 — serving API (functional, not a benchmark)

`results/stage5_smoke.txt` is a captured session against the real model
(`uv run python -m ember.server --port 8100`): a chat completion ("The capital of France is Paris.", TTFT 194 ms
for the first request after start-up), an SSE stream with `include_usage`, a stop-sequence request
(`" 7"` → text ends at `" 6,"`, `finish_reason: "stop"`), and the Prometheus metrics it produced. Serving
performance is measured under load in Stage 6.
