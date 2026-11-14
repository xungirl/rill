"""Stage 3: sequential vs static batching vs continuous batching (offline, same engine).

Workload (fixed seed): the first 48 chat prompts of benchmarks/data/prompts.txt, all
submitted at t=0. Each request's output length is drawn uniformly from [16, 256]
(seed 0) and forced with ignore_eos, so every mode generates exactly the same
6,000-odd tokens and only scheduling differs. MPS fp16, 2 GiB KV cache.

Modes (one LLMEngine each, same model weights):
  sequential      max_num_seqs=1 (one request at a time)
  static-B        policy="static": admit B, run until ALL B finish, then admit next B
  continuous-B    iteration-level scheduling, up to B running
plus one memory-tight continuous run (small KV pool) to exercise preemption.

Per request: TTFT = first token time - t0 (includes queueing), e2e = finish - t0,
TPOT = (e2e - TTFT) / (output_tokens - 1).

Run: uv run python -m benchmarks.bench_batching [--repeats 3] [--quick]
"""

from __future__ import annotations

import argparse
import random
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

PROMPTS = Path(__file__).parent / "data" / "prompts.txt"
NUM_REQUESTS = 48


def workload() -> tuple[list[list[int]], list[SamplingParams]]:
    tok = Tokenizer(resolve_model_dir())
    lines = PROMPTS.read_text().splitlines()[:NUM_REQUESTS]
    prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}])) for p in lines]
    rng = random.Random(0)
    params = [SamplingParams(max_tokens=rng.randint(16, 256), ignore_eos=True) for _ in prompts]
    return prompts, params


def run(model: Qwen3Model, cfg: EngineConfig, prompts: list[list[int]], params: list[SamplingParams]) -> dict[str, Any]:
    engine = LLMEngine(model, cfg)
    load_before = load1()
    t0 = time.monotonic()
    ids = [engine.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
    first: dict[int, float] = {}
    done: dict[int, float] = {}
    max_running = 0
    while engine.has_unfinished():
        outs = engine.step()
        now = time.monotonic()
        max_running = max(max_running, len(engine.scheduler.running) + sum(o.finished for o in outs))
        for o in outs:
            if o.new_token_id is not None and o.seq_id not in first:
                first[o.seq_id] = now - t0
            if o.finished:
                done[o.seq_id] = now - t0
    wall = time.monotonic() - t0
    outputs = [engine.seqs[i].output_token_ids for i in ids]
    n_out = sum(sp.max_tokens for sp in params)
    ttft = [first[i] for i in ids]
    e2e = [done[i] for i in ids]
    tpot = [(done[i] - first[i]) / (sp.max_tokens - 1) for i, sp in zip(ids, params, strict=True)]
    return {
        "wall_s": wall,
        "output_tok_s": n_out / wall,
        "req_s": len(ids) / wall,
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p99_s": percentile(ttft, 99),
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p99_s": percentile(e2e, 99),
        "tpot_mean_ms": 1000 * statistics.fmean(tpot),
        "steps": engine.stats.steps,
        "preemptions": engine.scheduler.num_preemptions,
        "max_running": max_running,
        "kv_usage_peak": max(engine.stats.kv_usage_samples),
        "kv_usage_mean": statistics.fmean(engine.stats.kv_usage_samples),
        "load1_before": load_before,
        "load1_after": load1(),
        "outputs": outputs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 repeat, skip sequential")
    args = ap.parse_args()
    repeats = 1 if args.quick else args.repeats
    model = load_model(device="mps", dtype=torch.float16)
    prompts, params = workload()

    def sched(**kw: Any) -> EngineConfig:
        return EngineConfig(scheduler=SchedulerConfig(max_num_batched_tokens=2048, **kw))

    modes: list[tuple[str, EngineConfig]] = [] if args.quick else [("sequential", sched(max_num_seqs=1))]
    for b in (8, 16, 32):
        modes.append((f"static-{b}", sched(max_num_seqs=b, policy="static")))
        modes.append((f"continuous-{b}", sched(max_num_seqs=b)))
    # 160 blocks x 16 = 2,560 token slots for up to 32 sequences that grow to ~300 tokens -> must preempt
    tight = EngineConfig(num_blocks=160, scheduler=SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=2048))
    modes.append(("continuous-32-tight-kv", tight))

    run(model, sched(max_num_seqs=8), prompts[:8], params[:8])  # warmup
    results = []
    reference: list[list[int]] | None = None  # outputs of the first mode, for batch-invariance check
    for name, cfg in modes:
        gate = wait_for_quiet()
        runs = [run(model, cfg, prompts, params) for _ in range(repeats)]
        outs = [r.pop("outputs") for r in runs]
        if reference is None:
            reference = outs[0]
        same_as_ref = [sum(a == b for a, b in zip(o, reference, strict=True)) for o in outs]
        same_across_repeats = all(o == outs[0] for o in outs)
        agg = {k: summarize([r[k] for r in runs]) for k in runs[0] if isinstance(runs[0][k], float)}
        results.append(
            {
                "mode": name,
                "gate": gate,
                "runs": runs,
                "summary": agg,
                "outputs_identical_to_first_mode": same_as_ref,
                "outputs_identical_across_repeats": same_across_repeats,
            }
        )
        s = agg
        print(
            f"{name:24s} {s['output_tok_s']['mean']:7.1f} tok/s (cv {s['output_tok_s']['cv_pct']:.1f}%)  "
            f"ttft p50 {s['ttft_p50_s']['mean']:.2f}s  e2e p99 {s['e2e_p99_s']['mean']:.2f}s  "
            f"tpot {s['tpot_mean_ms']['mean']:.1f}ms  preempt {runs[0]['preemptions']}  same {same_as_ref}"
        )
    write_result(
        "stage3_batching",
        {
            "num_requests": len(prompts),
            "total_output_tokens": sum(p.max_tokens for p in params),
            "prompt_tokens_total": sum(len(p) for p in prompts),
            "output_len_dist": "uniform[16,256], seed 0, ignore_eos",
            "repeats": repeats,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
