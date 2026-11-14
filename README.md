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

### 2. Paged KV cache
- **Goal**: stop reserving `max_len` KV slots per request; hold only what has been written.
- **Decision & alternatives**: vLLM-style paging — a pool of fixed-size blocks (16 tokens), a free list, a block
  table per sequence and per-block ref counts. A single `allocate_slots(seq, n)` covers whole-prompt prefill,
  prefill chunks and decode. The pool is a flat `[num_blocks*block_size, kv_heads, head_dim]` tensor per layer, so
  writing is one `index_copy_` by slot id. Reading uses plain PyTorch gather into a padded batch + one batched SDPA
  call per layer; a fused paged-attention kernel (vLLM/FlashInfer) does not exist for MPS, and writing a Metal
  kernel was out of scope. Alternative considered: a per-sequence Python loop over SDPA (no padding) — rejected
  because kernel launches scale with batch size, which is exactly what hurts on MPS.
- **Built**: `ember/block_manager.py`, `ember/paged_attention.py`, `ember/model_runner.py`, `ember/sequence.py`.
- **Verified**: hypothesis state-machine test of block-manager invariants; paged attention over mixed
  prefill-chunk/decode batches equals the full causal forward (block sizes 1, 3, 4, 16); real-model batched paged
  greedy == contiguous greedy.
- **Measured** (64 real generations): KV slot utilization 93.3% (paged, bs=16) vs 53.7% (reserve prompt+max_tokens)
  vs 7.3% (reserve max_model_len) averaged over decode steps; in a 2 GiB budget that is 85 vs 67 vs 9 requests.
- **Problems / fixes**: the first memory benchmark OOM'd because I sized the pool with the naive formula
  (64 × 2048 slots = 14 GiB) — a nice demonstration of the problem itself; resized to Σ(prompt+max_tokens).

### 3. Continuous batching
- **Goal**: serve many requests at once without waiting for the slowest member of a batch.
- **Decision & alternatives**: iteration-level scheduling (Orca / vLLM): every step re-decides who runs. Running
  sequences are served first (decode = 1 token, unfinished prefill = next chunk), then waiting requests are
  admitted FCFS under a per-step token budget (chunked prefill). Out of KV blocks → preempt the newest running
  sequence and *recompute* it later rather than swap: on unified memory, swapping to "CPU memory" buys nothing.
  Alternatives: prefill-first scheduling (better TTFT, worse ITL stalls); static batching (kept as a policy flag
  so the baseline uses identical kernels).
- **Built**: `ember/scheduler.py`, `ember/engine.py`.
- **Verified**: on the tiny model, outputs equal single-sequence greedy for roomy, chunked (budget 7 tokens),
  memory-tight (preempting), sequential and static configs; on the real model in fp32 with preemption too.
- **Measured**: 48 requests with output lengths uniform[16,256]: continuous B=32 **377.3 tok/s** vs static B=32
  308.2 vs sequential 41.4 (9.1×); p99 e2e 18.9 s vs 23.1 s vs 171.1 s. Decode-only throughput 517.8 tok/s at
  B=32 and 670.4 at B=64.
- **Problems / fixes**: first calibration showed 102 ms per decode step at B=32. Timing each op found 81 ms in
  the K/V gather — MPS advanced indexing runs at ~7 GB/s. Block-level `index_select` is 6.4× faster
  (micro-benchmark) → +66% throughput at B=32. Also: fp16 batching is not bit-batch-invariant (38–43/48 outputs
  match the sequential run); fp32 is exact — documented rather than hidden.

### 4. Prefix caching
- **Goal**: requests sharing a prefix (system prompt, few-shot examples, chat history) should not recompute it.
- **Decision & alternatives**: vLLM-v1-style block hashing: each *full, already-computed* block is registered
  under `hash(prev_block_hash, block_tokens)`, so a hit implies the whole prefix matches. New requests attach the
  longest cached run of blocks (ref_count += 1) and prefill only the rest. Freed cached blocks stay in the free
  list (an `OrderedDict` used as an LRU queue) until the space is needed — caching never reduces capacity. Only
  full blocks are shared and they are never written again, so no copy-on-write is required. Alternative: a radix
  tree over tokens (SGLang) — finer-grained matching but more complex eviction; not needed for this workload.
- **Built**: prefix-cache half of `ember/block_manager.py`, admission in `ember/scheduler.py`, registration in
  `ember/engine.py`.
- **Verified**: unit + state-machine tests (ref counts, LRU order, collision safety, hash ↔ block bijection);
  engine outputs identical with cache on/off on the fp32 tiny model, including forced evictions.
- **Measured** (587-token shared system prompt, 32 requests): TTFT of cache hits **361.5 → 39.5 ms (9.2×)**;
  batched throughput **48.6 → 151.0 tok/s (3.1×)**; 93% of prompt tokens served from cache.
- **Problems / fixes**: a test exposed that plain "least recently freed" eviction threw away cached blocks while
  empty blocks were free → empty blocks are now handed out first. In fp16, 29–31/32 greedy outputs match cache-off
  (different matmul shapes on a hit); exact in fp32.

### Key metrics

| metric | value | how measured | source |
|---|---|---|---|
| logits / per-layer max abs diff vs HF (fp32, 28 layers) | 0.0 | `benchmarks/parity_report.py` | `results/stage1_parity.json` |
| greedy token parity vs HF, fp16 MPS | 8/8 prompts × 64 tokens | same | `results/stage1_parity.json` |
| single-request decode, MPS fp16 | 45.9 ± 0.8 tok/s (n=3) | `benchmarks/bench_single.py` | `results/stage1_single_request.json` |
| single-request decode, CPU fp32 | 18.6 ± 0.0 tok/s (n=3) | same | `results/stage1_single_request.json` |
| KV slot utilization, paged bs=16 vs reserve prompt+max_tokens | 93.3% vs 53.7% (step avg, 64 real requests) | `benchmarks/bench_memory.py` | `results/stage2_memory.json` |
| requests fitting in 2 GiB KV: paged vs naive max_model_len=2048 | 85 vs 9 | same | `results/stage2_memory.json` |
| offline throughput, continuous vs static vs sequential (B=32, 48 req) | 377.3 vs 308.2 vs 41.4 tok/s (n=3, cv ≤ 0.9%) | `benchmarks/bench_batching.py` | `results/stage3_batching.json` |
| decode throughput at batch 32 / 64 | 517.8 / 670.4 tok/s | `benchmarks/bench_decode_batch.py` | `results/stage3_decode_batch.json` |
| K/V gather fix (56 gathers, B=32) | 81.0 → 12.7 ms | `benchmarks/micro_gather.py` | `results/micro_gather.json` |
| prefix cache: TTFT of hits, 614-token prompts | 361.5 → 39.5 ms (n=3, cv ≤ 0.5%) | `benchmarks/bench_prefix.py` | `results/stage4_prefix_cache.json` |
| prefix cache: batched throughput (B=16) | 48.6 → 151.0 tok/s | same | `results/stage4_prefix_cache.json` |

### Next steps
OpenAI-compatible server (5) →
load testing (6) → sampling & int8 (7) → ops polish (8).

## How to run

```bash
uv sync                                  # Python 3.15, torch 2.15
uv run pytest -q                         # unit tests (tiny random model) + real-model parity tests
uv run python -m benchmarks.parity_report
uv run python -m benchmarks.bench_single
uv run python -m benchmarks.bench_memory
uv run python -m benchmarks.bench_decode_batch
uv run python -m benchmarks.bench_batching
uv run python -m benchmarks.bench_prefix
```
The real-model tests/benchmarks expect `Qwen/Qwen3-0.6B` in the local Hugging Face cache; nothing is downloaded.

## Limitations
- Only the Qwen3 dense architecture (no MoE, no rope scaling / sliding window).
- Pure PyTorch eager; no custom kernels, no CUDA graphs. Batch-1 decode is launch-overhead bound.
- Paged attention gathers each sequence's K/V into a padded tensor every layer (no fused paged-attention kernel
  on MPS): extra memory traffic proportional to batch × longest context.
- fp16 results depend slightly on batch composition (not batch-invariant); greedy outputs can differ from a
  sequential run at near-tie tokens. fp32 is exact.
