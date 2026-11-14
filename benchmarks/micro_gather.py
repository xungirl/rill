"""Micro-benchmark: ways to gather paged K/V on MPS (motivates PagedAttention._gather).

Shape mirrors one decode step of Qwen3-0.6B at batch 32 with ~160-token contexts:
pool [1170*16 slots, 8 kv heads, 128] fp16, 32 sequences x 10 blocks. 56 gathers = K and V for 28 layers.

Run: uv run python -m benchmarks.micro_gather
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from benchmarks.common import load1, summarize, write_result

NB, BS, KV, D = 1170, 16, 8, 128
B, MB = 32, 10


def timed(f: Callable[[], torch.Tensor], n: int = 56) -> float:
    f()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.mps.synchronize()
    return (time.perf_counter() - t) * 1000


def main() -> None:
    g = torch.Generator(device="cpu").manual_seed(0)
    pool = torch.randn(NB * BS, KV, D, generator=g).to("mps", torch.float16)
    bt = torch.randint(0, NB, (B, MB), generator=g).to("mps")
    slots = (bt.unsqueeze(-1) * BS + torch.arange(BS, device="mps")).flatten(1)
    variants: dict[str, Callable[[], torch.Tensor]] = {
        "advanced_index_slots": lambda: pool[slots],
        "advanced_index_blocks": lambda: pool.view(NB, BS, KV, D)[bt].view(B, MB * BS, KV, D),
        "index_select_slots": lambda: pool.index_select(0, slots.flatten()).view(B, MB * BS, KV, D),
        "index_select_blocks": lambda: pool.view(NB, -1).index_select(0, bt.flatten()).view(B, MB * BS, KV, D),
    }
    ref = variants["advanced_index_slots"]()
    rows: dict[str, dict[str, Any]] = {}
    for name, f in variants.items():
        assert torch.equal(f(), ref)
        runs = [timed(f) for _ in range(3)]
        rows[name] = {"ms_per_56_gathers": summarize(runs), "runs": runs}
        print(name, f"{rows[name]['ms_per_56_gathers']['mean']:.1f} ms")
    write_result(
        "micro_gather",
        {"shape": {"slots": NB * BS, "kv": KV, "d": D, "B": B, "blocks_per_seq": MB}, "rows": rows, "load1": load1()},
    )


if __name__ == "__main__":
    main()
