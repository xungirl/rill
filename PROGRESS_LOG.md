# Progress log

## 2026-11-14 — Stage 1: model from scratch
- Implemented Qwen3 (RMSNorm, RoPE, GQA with per-head q/k RMSNorm, SwiGLU, tied embeddings) in plain PyTorch,
  loading `model.safetensors` directly (fused qkv / gate-up at load time). No `transformers` in `ember/`.
- Attention is behind an `AttentionBackend` protocol (no-cache, contiguous cache now; paged later).
- Verified: per-layer and logits diff vs HF = 0.0 (fp32 & bf16 CPU); greedy 8×64 tokens identical in fp32 CPU and
  fp16 MPS. Found bf16 on MPS diverges (3/8 vs fp32 reference, 6/8 vs HF bf16 MPS) → fp16 default on MPS.
- Measured single-request decode: MPS fp16 45.9 tok/s, CPU bf16 35.3, CPU fp32 18.6.
- Problem: first benchmark run had CPU rows with load1 > 4 (another job on the machine) → reran with the gate;
  first run kept as `results/stage1_single_request.run1.json`.

## 2026-11-14 — Stage 2: paged KV cache
- `BlockManager` (free list, block tables, ref counts, one `allocate_slots` for prefill/chunk/decode), flat
  per-layer KV pools, `PagedAttention` backend (scatter new K/V by `slot_mapping`, gather by block table, batched
  SDPA with a combined causal + length mask), `ModelRunner` building packed batches.
- Tests: hypothesis state machine (200 examples × 40 steps) checking ref_count == table membership, free list ==
  unreferenced blocks, minimal block count; paged mixed prefill/decode batches == full forward for block sizes
  1/3/4/16; real model batched paged greedy == contiguous greedy.
- Measured on 64 real generations: paged (bs=16) holds 93.3% useful slots on average vs 53.7% for
  `prompt+max_tokens` reservation and 7.3% for `max_model_len` reservation.
- Problem: the first benchmark run OOM'd on MPS — I had sized the pool as 64 × max_model_len (14 GiB), i.e. the
  naive policy itself. Fixed by sizing to Σ(prompt + max_tokens).
- Problem: I had added a "fully masked padding row → NaN" guard; working through the mask math showed a padding
  row always sees key 0, so the guard was dead code and was removed (test documents it).

## 2026-11-14 — Stage 3: continuous batching
- `Scheduler` (decode-first iteration-level scheduling, token budget, chunked prefill, FCFS admission,
  preempt-newest with recompute, `policy="static"` baseline) and `LLMEngine` (schedule → execute → sample → update).
- Tests: engine output == single-sequence reference for roomy / chunked (budget 7) / tight-memory (preempting) /
  sequential / static configs on the tiny model; random-arrival invariant test; decode-before-prefill ordering;
  static admits only when drained; abort frees blocks; real model fp32 CPU with preemption == reference.
- Perf bug found by calibration: B=32 decode step 102 ms. Op-level timing showed 81 ms in K/V gathers;
  micro-benchmark of 4 gather variants → block-level `index_select` (6.4× faster). Decode throughput at B=32
  311.8 → 517.8 tok/s, B=64 366.3 → 670.4 tok/s (before/after JSON both kept).
- Measured: continuous B=32 377.3 tok/s vs static 308.2 vs sequential 41.4 (48 requests, 3 repeats, cv ≤ 0.9%).
- Caveat recorded: fp16 MPS outputs differ from the sequential run for 5–10 of 48 requests (batch-shape rounding).
- Note: a few seconds of CPU-only unit tests overlapped the static-8 run (variance stayed < 1%).

## 2026-11-14 — Stage 4: prefix caching
- Chained block hashes (`hash(prev_hash, tokens)`), `cached: hash → block`, token-id check against collisions,
  `match_prefix` on admission (never the final token), `cache_full_blocks` after each step (only written K/V is
  shareable), `OrderedDict` free list doubling as LRU eviction queue.
- Design fix found by a failing test: pure free-time LRU evicted cached blocks while empty blocks were available;
  freed blocks without cached content now go to the front of the free list.
- Tests: unit tests (hits, last-token rule, chain dependency, revival from free list, LRU order, collision),
  hypothesis state machine with caching, engine cache on/off equality incl. evictions.
- Measured: TTFT for cache hits 361.5 → 39.5 ms; batched throughput 48.6 → 151.0 tok/s on a 587-token shared
  system prompt workload. fp16 output agreement on/off 29–31/32.
- Problem: first benchmark run crashed at the end (cv of an all-zero hit-rate series divided by zero) and lost
  its results; fixed `summarize` and reran.

## 2026-11-14 — Stage 5: serving API
- FastAPI app with `/v1/completions`, `/v1/chat/completions` (JSON + SSE, `stream_options.include_usage`),
  `/v1/models`, `/health`, `/metrics`; `AsyncLLMEngine` with a dedicated engine thread; 429 back-pressure, 504
  timeouts, abort on disconnect / stop string / timeout, stop-string hold-back, JSON request logs.
- Tests (10) run a real uvicorn server on localhost with the tiny model + a generated 512-token byte-level
  tokenizer: streaming == non-streaming text, chat role chunk, stop strings never leak, disconnect frees blocks,
  429 then recovery, 504, validation errors.
- Debugging story: the 429 test kept getting 200 — the server was *correctly* aborting: `next(resp.iter_lines())`
  discards the iterator, which closes the response, i.e. a client disconnect. I had also briefly added an
  `is_disconnected()` watcher to streaming responses; it is unnecessary (Starlette cancels the body iterator on
  disconnect) and was removed.
- Port 8000 is used by another app on this machine → smoke test and benchmarks use 8100/8101.

## 2026-11-14 — Stage 6: load testing
- Own load generator (open-loop Poisson + closed-loop, SSE, usage-based token counts), naive HF baseline server,
  sweep orchestrator with load gating, summarizer.
- **Found a real perf bug**: first ember sweep saturated at ~128 tok/s with TTFT p99 16 s at λ=2. Mixed
  prefill+decode steps padded every decode row to the prefill chunk length in one SDPA call (~32× attention work
  with one 1024-token chunk + 31 decodes). Fixed by attending decode rows and prefill rows in separate groups:
  mixed-step forward 4479.9 → 687.5 ms. Rerun: knee between λ=2 and 3 req/s, plateau ~201 tok/s.
- Baselines: HF `generate` server (one at a time) peaks at 54.1 tok/s → ember 3.7×. Ollama 0.34.4
  (qwen3:0.6b Q4_K_M, 522 MB pull) peaks at 217.3 tok/s and has much lower single-user TPOT (9 vs 26 ms).
- Problems: port 8000 occupied by an unrelated app → 8100/8101; Ollama hides Qwen3 thinking tokens as empty
  chunks → TTFT measured at first chunk for Ollama (documented); a shell `&&` chain skipped a `mv` because ruff
  failed, harmless (log was regenerated).

## 2026-11-14 — Stage 7: sampling & int8
- Sampling: temperature / top-k / top-p batched on device, per-request `torch.Generator` (seed) kept across
  preemption; stop sequences + max_tokens from Stage 5. Tests: hypothesis check of the mask vs a brute-force
  reference, empirical frequencies vs softmax (4σ), seeded output independent of batch neighbours and preemption,
  reproducible over HTTP.
- Chose weight-only int8 over speculative decoding: Qwen3-0.6B has no smaller draft model, and n-gram
  speculation needs KV rollback in the paged cache — int8 could be finished and measured properly.
- Measured: ppl 11.2848 (fp32) / 11.2859 (fp16) / 11.3000 (int8, +0.13%); weights −37%; batch-1 decode +20%,
  batch-32 −1%.
- Problem: top-p at batch 32 was 56% slower than greedy; per-row CPU multinomial was the cost → inverse-CDF on
  device (+69% throughput for sampled decoding).

## 2026-11-14 — Stage 8: ops polish
- Dockerfile (CPU, uv, non-root, healthcheck, mounted HF cache), `.dockerignore`; Linux torch from the PyTorch CPU
  index via `[tool.uv.sources]` (lockfile has no NVIDIA packages). Built and smoke-tested under colima, then
  stopped colima and restored the original Docker context.
- CI workflow (ruff, format, strict mypy, pytest). All steps run locally and pass; real-model tests skip without
  the checkpoint (64 passed / 6 skipped) and run with it (70 passed).
- docs/ARCHITECTURE.md (layers, request walk-through, invariants table), docs/CAPACITY.md (memory, measured
  envelope, knob sizing).
- Reran Stage 3/4 benchmarks on the final code: offline batching unchanged (377.2 tok/s); prefix cache batched
  gain is 2.4× now (cache-off got faster from the Stage 6 attention fix). Originals kept as `*_at_stageN.json`.
- Problem: container could not find the model at first — mount path assumed root; fixed with `HF_HUB_CACHE`.
- Full test suite run 10× → `results/stage8_test_10x.txt`.
