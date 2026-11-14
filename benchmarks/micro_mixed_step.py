"""Micro-benchmark: one mixed engine step = 1 prefill chunk + 31 decoding sequences.

Compares attention over ONE padded group (decode rows padded to the chunk length:
the original Stage 2/3 behaviour) with SEPARATE decode/prefill groups (current).
Full model forward on MPS fp16, Qwen3-0.6B, decode contexts of 300 tokens.

Run: uv run python -m benchmarks.micro_mixed_step
"""

from __future__ import annotations

import time
from typing import Any

import torch

from benchmarks.common import load1, summarize, write_result
from ember.loader import load_model
from ember.paged_attention import PagedAttention, PagedAttentionMetadata, PagedKVCache, _build_group

BS = 16
N_DECODE = 31
DECODE_CTX = 300


def main() -> None:
    model = load_model(device="mps", dtype=torch.float16)
    cache = PagedKVCache(model.cfg, 2048, BS, torch.float16, "mps")
    rows: dict[str, Any] = {}
    for chunk in (128, 512, 1024):
        tables, q_lens, ctxs = [], [], []
        nxt = 0
        for _ in range(N_DECODE):
            n = -(-DECODE_CTX // BS)
            tables.append(list(range(nxt, nxt + n)))
            nxt += n
            q_lens.append(1)
            ctxs.append(DECODE_CTX)
        n = -(-chunk // BS)
        tables.append(list(range(nxt, nxt + n)))
        q_lens.append(chunk)
        ctxs.append(chunk)
        split = PagedAttentionMetadata.build(tables, q_lens, ctxs, BS, "mps")
        starts = [sum(q_lens[:i]) for i in range(len(q_lens))]
        one = _build_group(list(range(len(q_lens))), tables, q_lens, ctxs, starts, BS, "mps")
        single = PagedAttentionMetadata(split.slot_mapping, [one], split.num_tokens)
        ids = torch.randint(0, 1000, (split.num_tokens,), device="mps")
        pos = torch.tensor([p for c, q in zip(ctxs, q_lens, strict=True) for p in range(c - q, c)], device="mps")
        res: dict[str, Any] = {}
        with torch.inference_mode():
            outs = {}
            for name, meta in (("single_group", single), ("split_groups", split)):
                model(ids, pos, PagedAttention(cache, meta))  # warmup
                times = []
                for _ in range(5):
                    torch.mps.synchronize()
                    t = time.perf_counter()
                    h, _ = model(ids, pos, PagedAttention(cache, meta))
                    torch.mps.synchronize()
                    times.append((time.perf_counter() - t) * 1000)
                outs[name] = h.float()
                res[name] = {"forward_ms": summarize(times), "runs_ms": times}
            a, b = outs["single_group"], outs["split_groups"]
            res["max_abs_diff"] = float((a - b).abs().max())
            res["max_abs_value"] = float(b.abs().max())
            res["relative_l2_diff"] = float((a - b).norm() / b.norm())
        rows[f"chunk_{chunk}"] = res
        print(
            f"chunk {chunk:5d}: single {res['single_group']['forward_ms']['mean']:7.1f} ms  "
            f"split {res['split_groups']['forward_ms']['mean']:7.1f} ms  rel diff {res['relative_l2_diff']:.2e}"
        )
    write_result(
        "micro_mixed_step", {"decode_rows": N_DECODE, "decode_ctx": DECODE_CTX, "rows": rows, "load1": load1()}
    )


if __name__ == "__main__":
    main()
