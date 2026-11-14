"""Engine decode throughput vs batch size (all sequences decoding in lock-step).

B identical 21-token prompts, 128 forced output tokens each (ignore_eos), one warmup
pass per B. Reports output tok/s and ms per engine step. MPS fp16.

Run: uv run python -m benchmarks.bench_decode_batch [--tag NAME] [--repeats 3]
"""

from __future__ import annotations

import argparse
import time

import torch

from benchmarks.common import load1, summarize, wait_for_quiet, write_result
from ember.engine import EngineConfig, LLMEngine
from ember.loader import load_model
from ember.sampling import SamplingParams
from ember.scheduler import SchedulerConfig

BATCHES = [1, 4, 8, 16, 32, 64]
NEW_TOKENS = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage3_decode_batch")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    model = load_model(device="mps", dtype=torch.float16)
    prompt = [785, 6722, 315, 9625, 374] * 4 + [13]
    rows = []
    for b in BATCHES:
        cfg = EngineConfig(scheduler=SchedulerConfig(max_num_seqs=b, max_num_batched_tokens=4096))
        engine = LLMEngine(model, cfg)
        engine.generate([prompt] * b, SamplingParams(max_tokens=8, ignore_eos=True))  # warmup
        gate = wait_for_quiet()
        runs = []
        for _ in range(args.repeats):
            before = load1()
            t = time.perf_counter()
            steps0 = engine.stats.steps
            engine.generate([prompt] * b, SamplingParams(max_tokens=NEW_TOKENS, ignore_eos=True))
            dt = time.perf_counter() - t
            steps = engine.stats.steps - steps0
            runs.append({"tok_s": b * NEW_TOKENS / dt, "ms_per_step": dt / steps * 1000, "load1_before": before})
        row = {
            "batch": b,
            "gate": gate,
            "runs": runs,
            "tok_s": summarize([r["tok_s"] for r in runs]),
            "ms_per_step": summarize([r["ms_per_step"] for r in runs]),
        }
        print(f"B={b:3d} {row['tok_s']['mean']:7.1f} tok/s  {row['ms_per_step']['mean']:6.1f} ms/step")
        rows.append(row)
    write_result(args.tag, {"new_tokens": NEW_TOKENS, "prompt_tokens": len(prompt), "rows": rows})


if __name__ == "__main__":
    main()
