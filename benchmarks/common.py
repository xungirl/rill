"""Helpers shared by every benchmark script: load gating, machine info, result files.

Another process may be using the CPU while we benchmark. Every result records
the 1-minute load average before and after the run, and `wait_for_quiet`
blocks (bounded) until load1 <= threshold before a key run starts.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"
LOAD_LIMIT = 4.0


def load1() -> float:
    return os.getloadavg()[0]


def wait_for_quiet(limit: float = LOAD_LIMIT, max_wait_s: float = 600.0, poll_s: float = 10.0) -> dict[str, Any]:
    """Wait until load1 <= limit (or give up after max_wait_s). Returns a record of what happened."""
    start = time.time()
    while load1() > limit and time.time() - start < max_wait_s:
        time.sleep(poll_s)
    return {"load1_at_start": round(load1(), 2), "waited_s": round(time.time() - start, 1), "gate": limit}


def machine_info() -> dict[str, Any]:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "?"

    return {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "ncpu": os.cpu_count(),
        "mem_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1) if sysctl("hw.memsize").isdigit() else "?",
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def summarize(xs: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "cv_pct": 100 * statistics.stdev(xs) / statistics.fmean(xs) if len(xs) > 1 and any(xs) else 0.0,
    }


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile (same definition as numpy's default)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = {"machine": machine_info(), **payload}
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return path
