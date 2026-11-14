# ember architecture

## Layers

```
 HTTP clients (OpenAI SDK, curl, benchmarks/loadgen.py)
        │  POST /v1/completions, /v1/chat/completions (JSON or SSE)
        ▼
┌───────────────────────────── asyncio event loop (uvicorn) ─────────────────────────────┐
│ ember/server/api.py        FastAPI routes, SSE framing, error mapping (400/422/429/504)  │
│ ember/server/async_engine  generate(): tokenize → inbox.put(add) → await own queue       │
│                            → incremental detokenize → stop strings → yield deltas       │
│                            finally: abort on disconnect / timeout / stop string          │
│ ember/server/metrics.py    Prometheus histograms + gauges   GET /metrics                 │
└───────────────────────────────┬───────────────────────────────▲─────────────────────────┘
          queue.Queue (thread-safe) │ ("add"|"abort", stream)      │ loop.call_soon_threadsafe
                                    ▼                              │ (StepOutput per token)
┌────────────────────────────── engine thread (only owner of the engine) ─────────────────┐
│ ember/engine.py  LLMEngine.step():                                                       │
│    1. Scheduler.schedule()        who runs, how many tokens each (token budget)          │
│    2. ModelRunner.execute()       one packed forward over all scheduled tokens           │
│    3. sample()                    greedy / temperature / top-k / top-p, per-seq RNG       │
│    4. update                      num_computed += n, append token, cache full blocks,     │
│                                   stop on EOS / max_tokens, free blocks                  │
│ ember/scheduler.py     waiting deque, running list, decode-first, chunked prefill,       │
│                        preempt-newest + recompute, static policy for baselines           │
│ ember/block_manager.py block pool, free list (OrderedDict = LRU), block tables,          │
│                        ref counts, chained block hashes for prefix caching               │
└───────────────────────────────┬──────────────────────────────────────────────────────────┘
                                ▼
┌────────────────────────────── model (MPS / CPU) ────────────────────────────────────────┐
│ ember/model_runner.py   input_ids/positions for the packed batch + PagedAttentionMetadata│
│ ember/model.py          Qwen3: embed → 28 × [RMSNorm → GQA attn (q/k norm, RoPE) → +     │
│                         RMSNorm → SwiGLU → +] → RMSNorm → tied LM head (last rows only)  │
│ ember/paged_attention   per layer: index_copy_ new K/V into slots; index_select blocks;  │
│                         padded batched SDPA with causal+length mask                      │
│ ember/quant.py          optional W8A16 Int8Linear (torch._weight_int8pack_mm)            │
│ ember/loader.py         safetensors → fused qkv / gate_up, meta-device init              │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## One request, end to end

1. `POST /v1/chat/completions` → pydantic validation → chat template (jinja2) → token ids.
2. `AsyncLLMEngine.generate` checks back-pressure (`inflight < max_inflight`, else 429), creates a `_Stream`
   (an `asyncio.Queue` + event-loop handle), puts `("add", stream)` in the inbox.
3. Engine thread drains the inbox → `LLMEngine.add_request` → `Sequence` in `scheduler.waiting`.
4. Next `schedule()`: running sequences first; then this one is admitted. `match_prefix` attaches cached blocks
   for the shared prefix (`num_computed_tokens` jumps forward); the remaining prompt tokens are scheduled, possibly
   in chunks across several steps.
5. `ModelRunner.execute` packs every scheduled token, builds slot mapping / block ids / mask once, runs the model;
   only rows that finish a sequence's known tokens are projected to logits.
6. Sampler picks a token; the engine appends it, registers newly full blocks in the prefix cache, checks EOS /
   `max_tokens`, and emits a `StepOutput`. The engine thread routes it to the stream with `call_soon_threadsafe`.
7. `generate` detokenizes incrementally, holds back text that might be the start of a stop string, yields deltas;
   the route frames them as SSE `data:` chunks.
8. On finish / disconnect / timeout the `finally` block records metrics + a JSON log line and, if the sequence is
   still alive, sends `("abort", stream)` so its KV blocks are released on the next engine iteration.

## Key invariants (all tested)

| invariant | where enforced | test |
|---|---|---|
| `block.ref_count` == number of block tables containing it | `BlockManager` | `test_block_manager.py` state machine |
| free list == blocks with ref_count 0, no duplicates | `BlockManager._release/_take_free_block` | same + `test_prefix_cache.py` machine |
| a sequence holds exactly ⌈computed / block_size⌉ blocks | `allocate_slots` | both state machines, `test_step_invariants_under_random_load` |
| cached hash ↔ block mapping is a bijection | `cache_full_blocks`, eviction | `test_prefix_cache_state_machine` |
| a block is shareable only after its K/V are written | engine calls `cache_full_blocks` after the step | design (see docstring) |
| engine output == single-sequence greedy (fp32) | whole pipeline | `test_scheduler.py`, real-model tests |
| aborted / disconnected requests free their blocks | `AsyncLLMEngine.generate` finally | `test_server.py` disconnect + timeout tests |

## Design trade-offs

- **Packed tokens + padded attention.** Linear layers run on `[T, H]` with no padding; only attention scatters into
  `[B, max_q]`. Mixed prefill/decode batches cost padding only inside SDPA.
- **No fused paged-attention kernel.** K/V are gathered per layer into a padded tensor. On MPS there is no
  PyTorch paged kernel; the gather is the main overhead at large batch × long context.
- **Recompute, not swap.** Apple silicon has unified memory, so "swapping to host" frees nothing.
- **Decode-first + chunked prefill.** Protects ITL of running requests; long prompts take several steps.
- **Prefix cache = free list.** Cached blocks cost nothing: they sit in the free list and are evicted LRU only when
  a block is actually needed. Empty blocks are handed out first.
- **One engine thread, one GIL.** Tokenization/detokenization/HTTP run on the event loop and compete with the
  engine thread for the GIL; production systems split API server and engine into processes (vLLM uses ZMQ).
