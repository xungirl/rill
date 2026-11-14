# Capacity notes (Qwen3-0.6B on one Apple M2, 24 GB)

Values marked **computed** follow from the model config; values marked **measured** cite a results file.

## Memory

| item | value | kind |
|---|---|---|
| parameters | 596.0 M (26% of them in the tied embedding) | computed from config |
| fp16 weights | 1.19 GB | computed |
| KV cache per token, fp16 (2 × 28 layers × 8 kv-heads × 128 × 2 B) | 112 KiB (114,688 B) | computed; `tests/test_model.py` |
| default KV pool (`--kv-cache-gib 2`) | 18,724 tokens = 1,170 blocks of 16 | computed (`PagedKVCache.num_blocks_for_budget`) |
| KV slot utilization, paged bs=16 (64 real chat generations) | 93.3% of allocated slots hold tokens (step avg) | measured, `results/stage2_memory.json` |
| int8 (W8A16) weights | see Stage 7 in BENCHMARK.md | measured, `results/stage7_quality_speed.json` |

Rule of thumb: concurrent sequences the KV pool can hold ≈ `18,724 / (avg prompt + avg output tokens)`; e.g. ~65
sequences of 290 tokens. The default `--max-num-seqs 32` is therefore compute-bound, not memory-bound, on this
model; raise `--kv-cache-gib` only if you raise `--max-num-seqs` or serve much longer contexts. If the pool is too
small the scheduler preempts (recompute) instead of failing: in `results/stage3_batching.json` a 2,560-slot pool
cost −26% throughput with 34 preemptions.

## Throughput / latency envelope (measured, see BENCHMARK.md Stage 6)

Workload: ~215 prompt / ~110 output tokens per request, Poisson arrivals, MPS fp16, defaults
(`--max-num-seqs 32 --max-num-batched-tokens 1024 --kv-cache-gib 2`), `results/stage6_serving_ember.json`:

| arrival rate | throughput | TTFT p50 / p99 | TPOT p50 | verdict |
|---|---|---|---|---|
| 1 req/s | 103.3 tok/s | 0.07 / 0.28 s | 37 ms | comfortable |
| 2 req/s | 162.3 tok/s | 0.14 / 0.47 s | 77 ms | last stable point |
| 3 req/s | 187.3 tok/s | 3.23 / 15.52 s | 145 ms | past the knee: queue grows |
| 6 req/s | 201.4 tok/s | 35.31 / 79.43 s | 148 ms | saturated (~1.74 req/s served) |

So one M2 serves about **1.7 req/s (~200 output tok/s) of this traffic at saturation; keep offered load ≤ ~2
req/s for sub-second p99 TTFT**. With Little's law at λ = 2 req/s and e2e p50 ≈ 8 s, ~16 requests are in flight
on average, comfortably inside `--max-num-seqs 32`; a `--max-inflight` of ~64 would start rejecting only once
the queue holds several seconds of work. Throughput per request scales roughly with (prompt + output) tokens:
prefill-heavier traffic lowers the req/s ceiling.

## Sizing the knobs

- `--max-inflight` (429 back-pressure): by Little's law, in-flight requests L = λ × W. Allow roughly
  `λ_knee × (target e2e)`; beyond that, requests only wait in the queue and should rather be rejected early.
- `--max-num-batched-tokens` (chunk size): a mixed step with a 1,024-token prefill chunk takes 687.5 ms of forward
  time vs 129.7 ms with a 128-token chunk (`results/micro_mixed_step.json`), and every running request waits for it.
  Smaller chunks → smoother inter-token latency, more steps per prompt.
- `--max-num-seqs`: decode throughput keeps growing with batch (517.8 tok/s at 32, 670.4 at 64 in
  `results/stage3_decode_batch.json`) at the price of TPOT (61.8 → 95.5 ms per step).
- Prefix caching is free capacity-wise (cached blocks live in the free list and are evicted LRU on demand); keep it on.
