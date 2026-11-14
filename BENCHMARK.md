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

## Stage 6 — load testing the HTTP servers

Commands (each: start server, warm up, sweep, 3 repeats with the load gate, stop server):
```bash
uv run python -m benchmarks.bench_serving --backend ember --port 8100 --rates 0.5,1,2,3,4,6 --concurrency 1,8,32 \
    --repeats 3 --server-args "--max-inflight 512"
uv run python -m benchmarks.bench_serving --backend hf --port 8101 --rates 0.2,0.5,1 --concurrency 1 --repeats 3 \
    --min-requests 16
OLLAMA_NUM_PARALLEL=8 ollama serve &   # Ollama 0.34.4 from Homebrew, model qwen3:0.6b (Q4_K_M GGUF, 522 MB)
uv run python -m benchmarks.bench_serving --backend ollama --rates 0.5,1,2,3,4,6 --concurrency 1,8,32 --repeats 3
uv run python -m benchmarks.summarize_serving results/stage6_serving_*.json
```
Workload (`benchmarks/loadgen.py`, synthetic, seeded; ShareGPT was not downloaded): prompt length
LogNormal(ln 160, 0.6) clipped [16, 1024], output length LogNormal(ln 100, 0.6) clipped [16, 400]; prompt text is
decoded random vocabulary ids. Measured averages: ~215 prompt / ~110 output tokens per request (ember). Each
repeat uses a different seed, so ± includes workload variation. Open loop = Poisson arrivals; points send
max(30, 40·λ) requests. TTFT = send → first streamed text chunk; TPOT = (e2e − TTFT)/(tokens − 1). req/s is
completed requests / (first send → last finish), so it sits below λ at low load (the tail is included).
Everything on one laptop over localhost; the load generator shares the CPU with the server.

**ember** (`results/stage6_serving_ember.json`, 3 repeats)

| load | output tok/s (mean ± sd) | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | e2e p50 | e2e p99 | errors |
|---|---|---|---|---|---|---|---|---|---|
| λ=0.5 req/s | 56.7 ± 5.6 | 0.50 | 0.15 s | 0.43 s | 34 ms | 43 ms | 3.32 s | 10.66 s | 0 |
| λ=1 req/s | 103.3 ± 12.8 | 0.90 | 0.07 s | 0.28 s | 37 ms | 53 ms | 4.02 s | 11.79 s | 0 |
| λ=2 req/s | 162.3 ± 8.7 | 1.39 | 0.14 s | 0.47 s | 77 ms | 113 ms | 7.96 s | 22.67 s | 0 |
| λ=3 req/s | 187.3 ± 2.9 | 1.62 | 3.23 s | 15.52 s | 145 ms | 171 ms | 20.10 s | 43.41 s | 0 |
| λ=4 req/s | 198.9 ± 6.3 | 1.73 | 13.46 s | 39.18 s | 149 ms | 174 ms | 30.84 s | 60.59 s | 0 |
| λ=6 req/s | 201.4 ± 6.7 | 1.74 | 35.31 s | 79.43 s | 148 ms | 177 ms | 53.04 s | 98.29 s | 0 |
| closed, 1 users | 37.3 ± 0.4 | 0.33 | 0.10 s | 0.38 s | 26 ms | 28 ms | 2.68 s | 8.03 s | 0 |
| closed, 8 users | 156.8 ± 9.7 | 1.38 | 0.10 s | 0.13 s | 45 ms | 49 ms | 4.58 s | 13.42 s | 0 |
| closed, 32 users | 197.1 ± 6.7 | 1.72 | 0.49 s | 0.85 s | 133 ms | 158 ms | 13.03 s | 32.83 s | 0 |

load1 at the start of each point: 1.2–2.4

**hf** (`results/stage6_serving_hf.json`, 3 repeats)

| load | output tok/s (mean ± sd) | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | e2e p50 | e2e p99 | errors |
|---|---|---|---|---|---|---|---|---|---|
| λ=0.2 req/s | 26.4 ± 3.2 | 0.22 | 0.34 s | 3.21 s | 17 ms | 18 ms | 2.68 s | 6.87 s | 0 |
| λ=0.5 req/s | 50.0 ± 6.9 | 0.43 | 5.91 s | 11.46 s | 17 ms | 18 ms | 8.11 s | 12.52 s | 0 |
| λ=1 req/s | 54.1 ± 0.7 | 0.47 | 22.22 s | 44.00 s | 17 ms | 19 ms | 23.26 s | 45.81 s | 0 |
| closed, 1 users | 54.9 ± 0.2 | 0.45 | 0.13 s | 1.74 s | 17 ms | 18 ms | 1.96 s | 5.47 s | 0 |

load1 at the start of each point: 1.3–2.4

**ollama** (`results/stage6_serving_ollama.json`, 3 repeats)

| load | output tok/s (mean ± sd) | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | e2e p50 | e2e p99 | errors |
|---|---|---|---|---|---|---|---|---|---|
| λ=0.5 req/s | 58.9 ± 5.8 | 0.52 | 0.10 s | 0.29 s | 12 ms | 30 ms | 1.37 s | 5.15 s | 0 |
| λ=1 req/s | 114.1 ± 15.6 | 1.00 | 0.06 s | 0.30 s | 23 ms | 34 ms | 2.09 s | 6.66 s | 0 |
| λ=2 req/s | 185.3 ± 13.2 | 1.59 | 0.31 s | 3.32 s | 34 ms | 42 ms | 4.17 s | 11.21 s | 0 |
| λ=3 req/s | 217.3 ± 7.4 | 1.88 | 4.80 s | 16.99 s | 33 ms | 46 ms | 9.06 s | 22.75 s | 0 |
| λ=4 req/s | 206.1 ± 23.9 | 1.80 | 17.25 s | 45.40 s | 36 ms | 51 ms | 20.93 s | 50.91 s | 0 |
| λ=6 req/s | 207.5 ± 2.4 | 1.79 | 36.98 s | 86.19 s | 35 ms | 47 ms | 41.46 s | 92.06 s | 0 |
| closed, 1 users | 93.0 ± 2.1 | 0.82 | 0.13 s | 0.48 s | 9 ms | 10 ms | 1.08 s | 2.85 s | 0 |
| closed, 8 users | 199.3 ± 8.1 | 1.76 | 0.10 s | 0.13 s | 37 ms | 40 ms | 3.69 s | 11.00 s | 0 |
| closed, 32 users | 205.9 ± 11.0 | 1.79 | 12.09 s | 14.52 s | 36 ms | 41 ms | 15.14 s | 22.09 s | 0 |

load1 at the start of each point: 2.1–3.9

### Reading the numbers
- **Saturation knee (ember)**: between λ=2 and λ=3 req/s. At λ=2 TTFT p99 is 0.47 s; at λ=3 it is 15.5 s and
  climbs linearly with λ afterwards (queueing), while throughput plateaus at ~200 tok/s / ~1.74 req/s.
  This workload is prefill-heavy (~2 prompt tokens per output token), so the plateau is far below the 517.8 tok/s
  decode-only number of Stage 3.
- **vs naive HF server**: peak sustained throughput 201.4 vs 54.1 tok/s (**3.7×**); HF saturates already at
  ~0.47 req/s, and at λ=1 its TTFT p50 is 22.2 s vs ember's 0.07 s.
- **vs Ollama (llama.cpp Metal, 4-bit Q4_K_M weights, 8 parallel slots)**: not like-for-like (ember runs fp16
  weights in eager PyTorch). Ollama peaks at 217.3 tok/s (ember reaches ~93% of that) and is clearly better at
  low concurrency: single-user TPOT 9 ms vs ember 26 ms. Ollama's Qwen3 runs in thinking mode via its
  chat template and streams hidden thinking tokens as empty-text chunks; for Ollama TTFT is taken at the first
  chunk (`loadgen.TTFT_ON_ANY_CHUNK`), and `ignore_eos` is not supported (outputs still averaged ~111 tokens).
- **Single-stream latency is ember's weak spot**: HF `generate` (17 ms TPOT) beats ember's server at
  concurrency 1 (26 ms) even though an isolated ember decode step is not slower (21.0 ms vs 22.6 ms for a bare HF
  forward with cache, scratch measurement, not committed). The difference is per-step Python overhead in the
  engine (scheduling, metadata, sampling bookkeeping) plus the engine-thread/event-loop hand-off and GIL sharing.
- **Bug found by this benchmark**: the first ember sweep saturated at ~128 tok/s with TTFT p99 16 s at λ=2
  (`results/stage6_serving_ember_before_split_attention.log`). Cause: mixed prefill+decode steps padded every
  decode row to the prefill chunk length inside one SDPA call. Splitting decode and prefill rows into separate
  attention groups fixed it (mixed 1024-chunk step 4479.9 → 687.5 ms, `results/micro_mixed_step.json`).
  Note: the Stage 3/4 tables above were measured before this fix (their workloads are decode-dominated or
  sequential, so they are mostly unaffected; Stage 8 reruns them on the final code).

## Stage 7 — sampling and weight-only int8

Command: `uv run python -m benchmarks.bench_quality` → `results/stage7_quality_speed.json` (load1 1.3–2.0).
An earlier run before the sampler fix is kept as `results/stage7_quality_speed_before_sampler_fix.json`.

### Accuracy: perplexity on a fixed local text
Corpus: CPython 3.15's `pydoc_data.topics` (stdlib documentation text, topics joined in sorted order), first
16 windows × 1,024 tokens = 16,368 scored tokens. Not a standard LM benchmark — compare rows with each other only.

| config | perplexity | Δ vs fp32 | decoder+embedding weight bytes |
|---|---|---|---|
| fp32, CPU (reference) | 11.2848 | — | — |
| fp16, MPS | 11.2859 | +0.01% | 1,192,099,840 (1.19 GB) |
| int8 W8A16 (per-channel symmetric), MPS | 11.3000 | +0.13% | 752,386,048 (0.75 GB, −37%) |

The embedding / tied LM head (26% of parameters) stays fp16, which is why weights shrink 37% rather than 50%.

### Speed (engine decode, 128 forced tokens, 3 runs each)

| config | batch 1 tok/s | batch 32 tok/s |
|---|---|---|
| fp16, greedy | 42.4 ± 0.1 | 538.4 ± 0.3 |
| fp16, temperature 0.8 + top-p 0.95 | 38.8 ± 0.3 | 398.5 ± 0.6 |
| int8, greedy | **51.0 ± 0.5 (+20%)** | 532.2 ± 1.1 (−1%) |

- int8 helps where decode is weight-bandwidth bound (batch 1) and does nothing at batch 32 where the step is
  dominated by attention / gather / launch overhead and the int8 kernel has to dequantize.
- **Sampler fix**: in the first run top-p sampling at batch 32 ran at only 235.8 tok/s: a per-row
  `torch.multinomial` over the 151,936-token vocabulary on the CPU cost ~54 ms of a 77 ms sampling step
  (scratch profiling). Inverse-CDF sampling on the device (one uniform per row from the request's own generator,
  batched `cumsum` + `searchsorted`) brought the sampler to ~22 ms and top-p throughput to 398.5 tok/s (+69%).
  The remaining gap to greedy is mostly the full-vocabulary sort needed for top-p.

## Stage 8 — reruns on the final code

The Stage 3 and Stage 4 tables were measured before two later fixes (split decode/prefill attention groups in
Stage 6, inverse-CDF sampler in Stage 7). Both benchmarks were rerun unchanged on the final code; the originals
are kept as `results/stage3_batching_at_stage3.json` and `results/stage4_prefix_cache_at_stage4.json`, and the
reruns overwrite `results/stage3_batching.json` / `results/stage4_prefix_cache.json` (logs:
`results/stage8_rerun_*.log`; load1 at start 1.3–3.9, all under the gate).

| benchmark | at its stage | final code |
|---|---|---|
| sequential, 48 req | 41.4 tok/s | 41.7 tok/s |
| static B=32 | 308.2 tok/s | 305.7 tok/s |
| continuous B=32 | 377.3 tok/s | 377.2 tok/s |
| continuous B=32, tight KV (preemptions) | 278.6 tok/s (34) | 286.9 tok/s (34) |
| prefix cache, TTFT of hits (cache off → on) | 361.5 → 39.5 ms | 359.8 → 39.6 ms |
| prefix cache, batched B=16 throughput (off → on) | 48.6 → 151.0 tok/s (3.1×) | **63.1 → 153.5 tok/s (2.4×)** |
| outputs identical cache on vs off (fp16) | 29/32, 31/32 | 29/32, 32/32 |

Offline batching is decode-dominated (short prompts), so it is unchanged within noise. The cache-off batched
prefix run is prefill-heavy (16 × 614-token prompts in mixed steps) and got 30% faster from the split-attention fix,
which shrinks the relative benefit of the prefix cache from 3.1× to 2.4×; the README reports the final numbers.

## Stage 8 — Docker and CI (functional)
- `docker build -t ember:cpu .` under colima (aarch64 VM, Docker 29.5.2): built in 1:44, 539 MB content
  (`results/stage8_docker_build.log`). Torch inside is `2.15.0+cpu` from the PyTorch CPU index.
- Container smoke test with the HF cache mounted read-only: `/health` ok, chat completion "The capital of France
  is Paris." (CPU fp32 in the VM: TTFT 0.91 s, e2e 3.70 s for 8 tokens) — `results/stage8_docker_smoke.txt`.
- CI steps run locally: ruff, ruff format --check, mypy (strict), pytest without the model (64 passed, 6 skipped)
  and with it (70 passed) — `results/stage8_ci_local.txt`. GitHub Actions itself was not run (no remote).
- Full test suite 10× in a row: `results/stage8_test_10x.txt`.
