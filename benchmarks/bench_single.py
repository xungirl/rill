"""Stage 1: single-request latency/throughput, MPS vs CPU, per dtype.

For one request (fixed prompt, greedy, fixed number of new tokens) we measure
  prefill_ms  - time for the prompt forward + first token (= TTFT, no queueing)
  decode_tok_s - generated tokens after the first / time spent generating them
Each config: 1 warmup + `--repeats` timed runs. Load average recorded.

Run: uv run python -m benchmarks.bench_single [--repeats 3] [--new-tokens 128]
"""

from __future__ import annotations

import argparse
import time

import torch

from benchmarks.common import load1, summarize, sync, wait_for_quiet, write_result
from rill.attention import ContiguousKVCache
from rill.loader import load_model, resolve_model_dir
from rill.model import Qwen3Model
from rill.tokenizer import Tokenizer

PROMPT = "Write a short story about a lighthouse keeper who discovers a message in a bottle."
CONFIGS = [("mps", torch.bfloat16), ("mps", torch.float16), ("cpu", torch.bfloat16), ("cpu", torch.float32)]


@torch.inference_mode()
def run_once(model: Qwen3Model, prompt_ids: list[int], new_tokens: int, device: str) -> tuple[float, float]:
    cfg = model.cfg
    dtype = model.embed_tokens.weight.dtype
    cache = ContiguousKVCache(
        cfg.num_hidden_layers, len(prompt_ids) + new_tokens, cfg.num_key_value_heads, cfg.head_dim, dtype, device
    )
    sync(device)
    t0 = time.perf_counter()
    tokens = torch.tensor(prompt_ids, device=device)
    t_first = 0.0
    for i in range(new_tokens):
        pos = torch.arange(cache.start_pos, cache.start_pos + len(tokens), device=device)
        hidden, _ = model(tokens, pos, cache)
        cache.start_pos += len(tokens)
        next_id = int(model.compute_logits(hidden[-1:]).argmax(-1).item())  # .item() syncs
        if i == 0:
            t_first = time.perf_counter()
        tokens = torch.tensor([next_id], device=device)
    t_end = time.perf_counter()
    return (t_first - t0) * 1000, (new_tokens - 1) / (t_end - t_first)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=128)
    args = ap.parse_args()
    tok = Tokenizer(resolve_model_dir())
    prompt_ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": PROMPT}]))
    rows = []
    for device, dtype in CONFIGS:
        model = load_model(device=device, dtype=dtype)
        run_once(model, prompt_ids, 16, device)  # warmup (MPS shader compilation, allocator)
        gate = wait_for_quiet()
        runs = []
        for _ in range(args.repeats):
            before = load1()
            ttft, tps = run_once(model, prompt_ids, args.new_tokens, device)
            runs.append({"prefill_ms": ttft, "decode_tok_s": tps, "load1_before": before, "load1_after": load1()})
            print(f"{device}/{dtype}: prefill {ttft:.1f} ms, decode {tps:.1f} tok/s, load1 {before:.2f}")
        rows.append(
            {
                "device": device,
                "dtype": str(dtype),
                "gate": gate,
                "runs": runs,
                "decode_tok_s": summarize([r["decode_tok_s"] for r in runs]),
                "prefill_ms": summarize([r["prefill_ms"] for r in runs]),
            }
        )
        del model
        if device == "mps":
            torch.mps.empty_cache()
    write_result(
        "stage1_single_request",
        {
            "prompt_tokens": len(prompt_ids),
            "new_tokens": args.new_tokens,
            "repeats": args.repeats,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
