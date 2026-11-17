"""Stage 6 orchestrator: start a server, sweep open-loop arrival rates (+ closed loop), stop it.

Backends (all on localhost, same machine):
  rill   uv run python -m rill.server        (this project, MPS fp16)
  hf      uv run python -m benchmarks.hf_server (transformers.generate, one request at a time)
  ollama  an already-running `ollama serve` (llama.cpp Metal backend, GGUF weights)

For each repeat and each rate, a fresh seeded workload from benchmarks/loadgen.py is
sent; the number of requests per point is max(min_requests, rate * seconds_per_point)
so low rates do not take forever and high rates still reach steady state.

Run: uv run python -m benchmarks.bench_serving --backend rill --rates 0.5,1,2,3,4,6 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from typing import Any

import httpx

from benchmarks import loadgen
from benchmarks.common import load1, wait_for_quiet, write_result
from benchmarks.loadgen import closed_loop, make_workload, open_loop, summarize_run

SERVERS = {
    "rill": [sys.executable, "-m", "rill.server", "--port", "{port}"],
    "hf": [sys.executable, "-m", "benchmarks.hf_server", "--port", "{port}"],
}


def wait_ready(url: str, path: str, timeout_s: float = 300) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            if httpx.get(url + path, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"server at {url} not ready after {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rill", "hf", "ollama"], required=True)
    ap.add_argument("--rates", default="0.5,1,2,3,4,6")
    ap.add_argument("--concurrency", default="")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-requests", type=int, default=30)
    ap.add_argument("--seconds-per-point", type=float, default=40)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--server-args", default="", help="extra args for the rill server")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    port = 11434 if args.backend == "ollama" else args.port
    url = f"http://127.0.0.1:{port}"
    model = args.model or ("qwen3:0.6b" if args.backend == "ollama" else "Qwen/Qwen3-0.6B")

    loadgen.TTFT_ON_ANY_CHUNK = args.backend == "ollama"
    proc = None
    if args.backend in SERVERS:
        cmd = [c.format(port=port) for c in SERVERS[args.backend]] + args.server_args.split()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(
            url, "/api/tags" if args.backend == "ollama" else ("/health" if args.backend == "rill" else "/docs")
        )
        warm = make_workload(4, seed=999)
        asyncio.run(open_loop(url, model, warm, 100.0, 999))  # warmup: compile shaders, load model
        rates = [float(x) for x in args.rates.split(",") if x]
        concs = [int(x) for x in args.concurrency.split(",") if x]
        repeats: list[list[dict[str, Any]]] = []
        for rep in range(args.repeats):
            points = []
            for rate in rates:
                n = max(args.min_requests, int(rate * args.seconds_per_point))
                specs = make_workload(n, seed=rep)
                gate = wait_for_quiet()
                res, wall = asyncio.run(open_loop(url, model, specs, rate, seed=rep))
                p = {"mode": "open", "rate": rate, "gate": gate, **summarize_run(res, wall), "load1_after": load1()}
                points.append(p)
                print(
                    f"[{args.backend} rep{rep}] rate {rate:5.2f}: {p['output_tok_s']:6.1f} tok/s {p['req_s']:.2f} req/s"
                    f"  ttft p50/p99 {p['ttft_s']['p50']:.2f}/{p['ttft_s']['p99']:.2f}s"
                    f"  tpot p50 {p['tpot_s']['p50'] * 1000:.0f}ms  e2e p99 {p['e2e_s']['p99']:.1f}s"
                    f"  err {p['errors']}  load1 {gate['load1_at_start']}",
                    flush=True,
                )
            for c in concs:
                specs = make_workload(max(args.min_requests, 2 * c), seed=rep)
                gate = wait_for_quiet()
                res, wall = asyncio.run(closed_loop(url, model, specs, c))
                p = {
                    "mode": "closed",
                    "concurrency": c,
                    "gate": gate,
                    **summarize_run(res, wall),
                    "load1_after": load1(),
                }
                points.append(p)
                print(
                    f"[{args.backend} rep{rep}] closed c={c}: {p['output_tok_s']:6.1f} tok/s"
                    f"  ttft p50 {p['ttft_s']['p50']:.2f}s  tpot p50 {p['tpot_s']['p50'] * 1000:.0f}ms"
                    f"  err {p['errors']}",
                    flush=True,
                )
            repeats.append(points)
        write_result(
            args.tag or f"stage6_serving_{args.backend}",
            {"backend": args.backend, "url": url, "model": model, "server_args": args.server_args, "repeats": repeats},
        )
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
