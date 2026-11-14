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
