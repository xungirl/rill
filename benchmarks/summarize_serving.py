"""Aggregate results/stage6_serving_*.json across repeats into Markdown tables.

Each cell = mean over repeats (± sample stdev of the repeat means where shown).
Run: uv run python -m benchmarks.summarize_serving results/stage6_serving_ember.json [...]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _ms(x: float) -> str:
    return f"{x * 1000:.0f} ms"


def _s(x: float) -> str:
    return f"{x:.2f} s"


def table(path: str) -> str:
    data = json.loads(Path(path).read_text())
    by_key: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for rep in data["repeats"]:
        for p in rep:
            key = (p["mode"], p.get("rate", p.get("concurrency")))
            by_key[key].append(p)
    lines = [
        f"**{data['backend']}** (`{path}`, {len(data['repeats'])} repeats)",
        "",
        "| load | output tok/s (mean ± sd) | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 "
        "| e2e p50 | e2e p99 | errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (mode, x), pts in sorted(by_key.items(), key=lambda kv: (kv[0][0] != "open", kv[0][1])):

        def m(f: Any, pts: list[dict[str, Any]] = pts) -> float:
            return statistics.fmean(f(p) for p in pts)

        tok = [p["output_tok_s"] for p in pts]
        sd = statistics.stdev(tok) if len(tok) > 1 else 0.0
        label = f"λ={x:g} req/s" if mode == "open" else f"closed, {x} users"
        lines.append(
            f"| {label} | {statistics.fmean(tok):.1f} ± {sd:.1f} | {m(lambda p: p['req_s']):.2f} "
            f"| {_s(m(lambda p: p['ttft_s']['p50']))} | {_s(m(lambda p: p['ttft_s']['p99']))} "
            f"| {_ms(m(lambda p: p['tpot_s']['p50']))} | {_ms(m(lambda p: p['tpot_s']['p99']))} "
            f"| {_s(m(lambda p: p['e2e_s']['p50']))} | {_s(m(lambda p: p['e2e_s']['p99']))} "
            f"| {sum(p['errors'] for p in pts)} |"
        )
    loads = [p["gate"]["load1_at_start"] for rep in data["repeats"] for p in rep]
    lines.append("")
    lines.append(f"load1 at the start of each point: {min(loads):.1f}–{max(loads):.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(table(path))
        print()
