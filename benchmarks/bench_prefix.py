"""Stage 4: prefix caching on a shared-system-prompt workload.

Workload: one 587-token system prompt (benchmarks/data/system_prompt.txt) shared by
32 requests whose user turn is one of the first 32 lines of prompts.txt, rendered
with the chat template (~610 prompt tokens each). 32 forced output tokens per
request (ignore_eos). MPS fp16. A fresh engine per run, so every run starts cold.

  (a) latency: requests one at a time (max_num_seqs=1); TTFT per request.
      With the cache on, request 1 is a miss and requests 2..32 hit.
  (b) throughput: all 32 submitted at t=0, continuous batching, max_num_seqs=16.
Correctness: greedy outputs compared request by request, cache on vs off.

Run: uv run python -m benchmarks.bench_prefix [--repeats 3]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.common import load1, percentile, summarize, wait_for_quiet, write_result
from ember.engine import EngineConfig, LLMEngine
from ember.loader import load_model, resolve_model_dir
from ember.model import Qwen3Model
from ember.sampling import SamplingParams
from ember.scheduler import SchedulerConfig
from ember.tokenizer import Tokenizer

DATA = Path(__file__).parent / "data"
NUM_REQUESTS = 32
OUTPUT_TOKENS = 32


def workload() -> list[list[int]]:
    tok = Tokenizer(resolve_model_dir())
    system = (DATA / "system_prompt.txt").read_text().strip()
    users = (DATA / "prompts.txt").read_text().splitlines()[:NUM_REQUESTS]
    return [
        tok.encode(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}]))
        for u in users
    ]


def run(
    model: Qwen3Model, prompts: list[list[int]], caching: bool, max_num_seqs: int, one_at_a_time: bool
) -> dict[str, Any]:
    cfg = EngineConfig(
        enable_prefix_caching=caching,
        scheduler=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=2048),
    )
    engine = LLMEngine(model, cfg)
    sp = SamplingParams(max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    load_before = load1()
    ttft: list[float] = []
    outputs: list[list[int]] = []
    t0 = time.monotonic()
    if one_at_a_time:
        for p in prompts:
            start = time.monotonic()
            sid = engine.add_request(p, sp)
            first = None
            while engine.has_unfinished():
                for o in engine.step():
                    if o.new_token_id is not None and first is None:
                        first = time.monotonic() - start
            assert first is not None
            ttft.append(first)
            outputs.append(engine.seqs.pop(sid).output_token_ids)
    else:
        ids = [engine.add_request(p, sp) for p in prompts]
        firsts: dict[int, float] = {}
        while engine.has_unfinished():
            for o in engine.step():
                if o.new_token_id is not None and o.seq_id not in firsts:
                    firsts[o.seq_id] = time.monotonic() - t0
        ttft = [firsts[i] for i in ids]
        outputs = [engine.seqs.pop(i).output_token_ids for i in ids]
    wall = time.monotonic() - t0
    sched = engine.scheduler
    return {
        "wall_s": wall,
        "output_tok_s": len(prompts) * OUTPUT_TOKENS / wall,
        "ttft_mean_s": statistics.fmean(ttft),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_mean_after_first_s": statistics.fmean(ttft[1:]),
        "ttft_first_s": ttft[0],
        "prefill_tokens_computed": engine.stats.prefill_tokens,
        "hit_rate": sched.prefix_hit_tokens / max(1, sched.prefix_query_tokens),
        "load1_before": load_before,
        "load1_after": load1(),
        "outputs": outputs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    model = load_model(device="mps", dtype=torch.float16)
    prompts = workload()
    run(model, prompts[:4], True, 4, False)  # warmup
    results: dict[str, Any] = {}
    for scenario, max_seqs, sequential in (("latency_one_at_a_time", 1, True), ("throughput_batch16", 16, False)):
        for caching in (False, True):
            key = f"{scenario}/cache_{'on' if caching else 'off'}"
            gate = wait_for_quiet()
            runs = [run(model, prompts, caching, max_seqs, sequential) for _ in range(args.repeats)]
            outs = [r.pop("outputs") for r in runs]
            results[key] = {
                "gate": gate,
                "runs": runs,
                "summary": {k: summarize([r[k] for r in runs]) for k in runs[0] if isinstance(runs[0][k], float)},
                "outputs": outs[0],
                "identical_across_repeats": all(o == outs[0] for o in outs),
            }
            s = results[key]["summary"]
            print(
                f"{key:40s} ttft mean {s['ttft_mean_s']['mean'] * 1000:7.1f} ms  "
                f"tok/s {s['output_tok_s']['mean']:6.1f}  hit {runs[0]['hit_rate']:.2f}  "
                f"prefill computed {runs[0]['prefill_tokens_computed']}"
            )
    for scenario in ("latency_one_at_a_time", "throughput_batch16"):
        on, off = results[f"{scenario}/cache_on"], results[f"{scenario}/cache_off"]
        same = sum(a == b for a, b in zip(on["outputs"], off["outputs"], strict=True))
        results[f"{scenario}/outputs_identical_on_vs_off"] = f"{same}/{len(prompts)}"
        print(scenario, "outputs identical cache on vs off:", f"{same}/{len(prompts)}")
    for v in results.values():
        if isinstance(v, dict):
            v.pop("outputs", None)
    write_result(
        "stage4_prefix_cache",
        {
            "num_requests": len(prompts),
            "prompt_tokens_mean": statistics.fmean(len(p) for p in prompts),
            "output_tokens": OUTPUT_TOKENS,
            "repeats": args.repeats,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
