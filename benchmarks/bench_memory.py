"""Stage 2: KV-cache memory utilization, paged vs naive per-request preallocation.

1. Generate real outputs: all 64 prompts in benchmarks/data/prompts.txt run as
   one static batch through the paged ModelRunner (greedy, max_tokens=256,
   stop at EOS). This yields real (prompt_len, output_len) pairs and, at every
   step, the real number of allocated KV blocks.
2. Compare, for the same sequences, how many KV slots each policy reserves:
     naive_max_model_len : every request reserves max_model_len slots (HF-style static cache)
     naive_request_max   : every request reserves prompt_len + max_tokens slots
     paged (bs = 8/16/32): ceil(current_len / bs) * bs slots
   utilization = tokens actually stored / slots reserved, averaged over steps.
3. Translate to capacity: how many such requests fit into a fixed KV budget.

Run: uv run python -m benchmarks.bench_memory
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from benchmarks.common import load1, write_result
from rill.block_manager import BlockManager
from rill.config import ModelConfig
from rill.loader import load_model, resolve_model_dir
from rill.model_runner import ModelRunner, ScheduledSeq
from rill.paged_attention import PagedKVCache
from rill.sampling import SamplingParams, sample
from rill.sequence import Sequence
from rill.tokenizer import Tokenizer

PROMPTS = Path(__file__).parent / "data" / "prompts.txt"
MAX_TOKENS = 256
MAX_MODEL_LEN = 2048
BLOCK_SIZE = 16
BUDGET_GIB = 2.0


def ceil_to(n: int, k: int) -> int:
    return -(-n // k) * k


@torch.inference_mode()
def run_static_batch(device: str, dtype: torch.dtype) -> tuple[list[Sequence], list[list[tuple[int, int]]], int]:
    """Returns finished sequences, per-step (seq_id, current_len) of live sequences, peak blocks."""
    tok = Tokenizer(resolve_model_dir())
    model = load_model(device=device, dtype=dtype)
    eos = set(model.cfg.eos_token_ids)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}]) for p in PROMPTS.read_text().splitlines()]
    seqs = [Sequence(i, tok.encode(p), SamplingParams(max_tokens=MAX_TOKENS)) for i, p in enumerate(prompts)]
    # enough blocks for every sequence to reach its maximum length (so nothing is ever preempted)
    num_blocks = sum(ceil_to(s.num_prompt_tokens + MAX_TOKENS, BLOCK_SIZE) // BLOCK_SIZE for s in seqs)
    bm = BlockManager(num_blocks, BLOCK_SIZE)
    runner = ModelRunner(model, PagedKVCache(model.cfg, num_blocks, BLOCK_SIZE, dtype, device))
    running = list(seqs)
    trace: list[list[tuple[int, int]]] = []
    peak_blocks = 0
    while running:
        batch = []
        for s in running:
            n = s.num_tokens - s.num_computed_tokens
            bm.allocate_slots(s, n)
            batch.append(ScheduledSeq(s, n))
        peak_blocks = max(peak_blocks, bm.num_used_blocks)
        trace.append([(s.seq_id, s.num_tokens) for s in running])  # all tokens get a slot this step
        next_ids = sample(runner.execute(batch), [s.sampling for s in running])
        for s, b, t in zip(running, batch, next_ids, strict=True):
            s.num_computed_tokens += b.num_new_tokens
            s.append_token(t)
            if t in eos or s.num_output_tokens >= MAX_TOKENS:
                s.finish_reason = "stop" if t in eos else "length"
                bm.free(s)
        running = [s for s in running if s.finish_reason is None]
    assert bm.num_free_blocks == num_blocks
    return seqs, trace, peak_blocks


def main() -> None:
    device, dtype = "mps", torch.float16
    load_before = load1()
    seqs, trace, peak_blocks = run_static_batch(device, dtype)
    prompt_lens = [s.num_prompt_tokens for s in seqs]
    out_lens = [s.num_output_tokens for s in seqs]

    def reserved(policy: str, lengths: list[int], ids: list[int]) -> int:
        if policy == "naive_max_model_len":
            return MAX_MODEL_LEN * len(lengths)
        if policy == "naive_request_max":
            return sum(prompt_lens[i] + MAX_TOKENS for i in ids)
        bs = int(policy.split("_bs")[1])
        return sum(ceil_to(n, bs) for n in lengths)

    policies = ["naive_max_model_len", "naive_request_max", "paged_bs8", "paged_bs16", "paged_bs32"]
    # step-averaged utilization: each trace entry lists current lengths of running seqs (in id order)
    util: dict[str, float] = {}
    for pol in policies:
        ratios = []
        for step in trace:
            ids, lengths = [i for i, _ in step], [n for _, n in step]
            ratios.append(sum(lengths) / reserved(pol, lengths, ids))
        util[pol] = statistics.fmean(ratios)
    final_lens = [p + o for p, o in zip(prompt_lens, out_lens, strict=True)]
    all_ids = list(range(len(seqs)))
    final_util = {pol: sum(final_lens) / reserved(pol, final_lens, all_ids) for pol in policies}

    cfg = ModelConfig.from_pretrained_dir(resolve_model_dir())
    budget_tokens = int(BUDGET_GIB * 2**30) // cfg.kv_bytes_per_token(2)
    capacity = {pol: budget_tokens * len(seqs) // reserved(pol, final_lens, all_ids) for pol in policies}

    result = {
        "device": device,
        "dtype": str(dtype),
        "num_requests": len(seqs),
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "block_size_used_in_run": BLOCK_SIZE,
        "prompt_len": {"mean": statistics.fmean(prompt_lens), "min": min(prompt_lens), "max": max(prompt_lens)},
        "output_len": {"mean": statistics.fmean(out_lens), "min": min(out_lens), "max": max(out_lens)},
        "finished_by_eos": sum(s.finish_reason == "stop" for s in seqs),
        "steps": len(trace),
        "peak_blocks_allocated_in_run": peak_blocks,
        "step_avg_utilization": util,
        "final_length_utilization": final_util,
        "kv_bytes_per_token": cfg.kv_bytes_per_token(2),
        "budget_gib": BUDGET_GIB,
        "budget_tokens": budget_tokens,
        "requests_that_fit_in_budget": capacity,
        "load1_before": load_before,
        "load1_after": load1(),
    }
    for k in ("output_len", "step_avg_utilization", "final_length_utilization", "requests_that_fit_in_budget"):
        print(k, result[k])
    write_result("stage2_memory", result)


if __name__ == "__main__":
    main()
