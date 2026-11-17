"""Stage 7: accuracy and speed of weight-only int8 vs fp16 (and sampler overhead).

Perplexity corpus: Python's own stdlib documentation, `pydoc_data.topics.topics`
(616 KB of English technical prose shipped with every CPython; topics joined in
sorted-key order). We score the first N_WINDOWS non-overlapping windows of
WINDOW tokens: ppl = exp(mean next-token cross-entropy). This is a fixed,
local, reproducible text - not a standard LM benchmark, so compare configs
against each other, not against published numbers.

Configs: fp32 CPU (reference), fp16 MPS, int8 (W8A16) MPS.
Speed: engine decode tok/s at batch 1 and 32 (same protocol as bench_decode_batch),
greedy vs temperature=0.8/top_p=0.95 sampling, fp16 vs int8.

Run: uv run python -m benchmarks.bench_quality
"""

from __future__ import annotations

import math
import pydoc_data.topics
import time
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.common import load1, summarize, wait_for_quiet, write_result
from rill.attention import NoCacheAttention
from rill.engine import EngineConfig, LLMEngine
from rill.loader import load_model, resolve_model_dir
from rill.model import Qwen3Model
from rill.quant import quantize_int8, weight_bytes
from rill.sampling import SamplingParams
from rill.scheduler import SchedulerConfig
from rill.tokenizer import Tokenizer

WINDOW = 1024
N_WINDOWS = 16


def corpus_windows() -> list[list[int]]:
    topics: dict[str, str] = pydoc_data.topics.topics
    text = "\n\n".join(topics[k] for k in sorted(topics))
    ids = Tokenizer(resolve_model_dir()).encode(text)
    return [ids[i * WINDOW : (i + 1) * WINDOW] for i in range(N_WINDOWS)]


@torch.inference_mode()
def perplexity(model: Qwen3Model, windows: list[list[int]]) -> dict[str, float]:
    dev = model.embed_tokens.weight.device
    nll, n = 0.0, 0
    for w in windows:
        ids = torch.tensor(w, device=dev)
        hidden, _ = model(ids, torch.arange(len(w), device=dev), NoCacheAttention())
        logits = model.compute_logits(hidden[:-1]).float()
        nll += float(F.cross_entropy(logits, ids[1:], reduction="sum"))
        n += len(w) - 1
    return {"ppl": math.exp(nll / n), "mean_nll": nll / n, "tokens_scored": n}


def decode_speed(model: Qwen3Model, batch: int, sampling: SamplingParams, repeats: int = 3) -> dict[str, Any]:
    prompt = [785, 6722, 315, 9625, 374] * 4 + [13]
    engine = LLMEngine(model, EngineConfig(scheduler=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=4096)))
    engine.generate([prompt] * batch, SamplingParams(max_tokens=8, ignore_eos=True))  # warmup
    gate = wait_for_quiet()
    runs = []
    for _ in range(repeats):
        before = load1()
        t = time.perf_counter()
        engine.generate([prompt] * batch, sampling)
        runs.append({"tok_s": batch * sampling.max_tokens / (time.perf_counter() - t), "load1_before": before})
    return {"gate": gate, "runs": runs, "tok_s": summarize([r["tok_s"] for r in runs])}


def main() -> None:
    windows = corpus_windows()
    out: dict[str, Any] = {"window": WINDOW, "n_windows": N_WINDOWS, "corpus": "pydoc_data.topics (CPython 3.15)"}
    ref = load_model(device="cpu", dtype=torch.float32)
    out["ppl_fp32_cpu"] = perplexity(ref, windows)
    print("fp32 cpu", out["ppl_fp32_cpu"])
    del ref
    fp16 = load_model(device="mps", dtype=torch.float16)
    out["ppl_fp16_mps"] = perplexity(fp16, windows)
    out["weight_bytes_fp16"] = weight_bytes(fp16)
    print("fp16 mps", out["ppl_fp16_mps"])
    int8 = quantize_int8(load_model(device="mps", dtype=torch.float16))
    torch.mps.empty_cache()
    out["ppl_int8_mps"] = perplexity(int8, windows)
    out["weight_bytes_int8"] = weight_bytes(int8)
    print("int8 mps", out["ppl_int8_mps"], out["weight_bytes_fp16"], out["weight_bytes_int8"])

    greedy = SamplingParams(max_tokens=128, ignore_eos=True)
    sampled = SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95, ignore_eos=True)
    speed: dict[str, Any] = {}
    for name, model in (("fp16", fp16), ("int8", int8)):
        for b in (1, 32):
            for sname, sp in (("greedy", greedy), ("top_p", sampled)):
                if name == "int8" and sname == "top_p":
                    continue
                key = f"{name}/B{b}/{sname}"
                speed[key] = decode_speed(model, b, sp)
                print(key, f"{speed[key]['tok_s']['mean']:.1f} tok/s (cv {speed[key]['tok_s']['cv_pct']:.1f}%)")
    out["decode_speed"] = speed
    write_result("stage7_quality_speed", out)


if __name__ == "__main__":
    main()
