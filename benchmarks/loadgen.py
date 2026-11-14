"""Load generator for any OpenAI-compatible /v1/completions endpoint (SSE streaming).

Two arrival models:
  open-loop  : requests arrive as a Poisson process at `rate` req/s regardless of
               how the server is doing (inter-arrival ~ Exponential(1/rate)). This
               is how real traffic behaves and is what exposes queueing.
  closed-loop: `concurrency` users, each sends its next request as soon as the
               previous one finishes. Load adapts to server speed (no overload).

Workload (synthetic, seeded; ShareGPT was not used to avoid a download): prompt
and output lengths are log-normal, clipped:
    prompt_tokens ~ LogNormal(mu=ln 160, sigma=0.6) clipped to [16, 1024]
    output_tokens ~ LogNormal(mu=ln 100, sigma=0.6) clipped to [16, 400]
Prompt text is produced by decoding random token ids from the Qwen3 vocabulary
(ids 1000..30000) so it tokenizes to roughly the target length; the server's
reported usage gives the exact counts. `ignore_eos` is sent so the output length
is honoured (servers that don't support it may stop early; we always use the
actual token count from `usage`).

Per request we record: TTFT (send -> first non-empty text chunk), e2e, output
tokens; TPOT = (e2e - TTFT) / (output_tokens - 1).

Run: uv run python -m benchmarks.loadgen --url http://127.0.0.1:8000 --rates 1,2,4 --num-requests 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from benchmarks.common import load1, percentile, write_result
from ember.loader import resolve_model_dir
from ember.tokenizer import Tokenizer


@dataclass
class RequestSpec:
    prompt: str
    max_tokens: int


@dataclass
class RequestResult:
    ok: bool
    status: int
    ttft_s: float
    e2e_s: float
    prompt_tokens: int
    output_tokens: int
    start_s: float  # relative to benchmark start
    error: str = ""


def make_workload(n: int, seed: int) -> list[RequestSpec]:
    rng = random.Random(seed)
    tok = Tokenizer(resolve_model_dir())
    specs = []
    for _ in range(n):
        p = int(min(1024, max(16, rng.lognormvariate(math.log(160), 0.6))))
        o = int(min(400, max(16, rng.lognormvariate(math.log(100), 0.6))))
        text = tok.decode([rng.randrange(1000, 30000) for _ in range(p)])
        specs.append(RequestSpec(text, o))
    return specs


async def send(client: httpx.AsyncClient, url: str, model: str, spec: RequestSpec, t0: float) -> RequestResult:
    body = {
        "model": model,
        "prompt": spec.prompt,
        "max_tokens": spec.max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = math.nan
    usage: dict[str, int] = {}
    chunks = 0
    try:
        async with client.stream("POST", url + "/v1/completions", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                return RequestResult(
                    False, r.status_code, math.nan, time.perf_counter() - start, 0, 0, start - t0, r.text[:200]
                )
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("text"):
                        chunks += 1
                        if math.isnan(ttft):
                            ttft = time.perf_counter() - start
    except httpx.HTTPError as e:
        return RequestResult(False, 0, math.nan, time.perf_counter() - start, 0, 0, start - t0, repr(e)[:200])
    e2e = time.perf_counter() - start
    out = usage.get("completion_tokens", chunks)
    return RequestResult(True, 200, ttft, e2e, usage.get("prompt_tokens", 0), out, start - t0)


async def open_loop(
    url: str, model: str, specs: list[RequestSpec], rate: float, seed: int
) -> tuple[list[RequestResult], float]:
    rng = random.Random(seed + 1)
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=limits) as client:
        t0 = time.perf_counter()
        tasks = []
        next_at = 0.0
        for spec in specs:
            delay = t0 + next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(client, url, model, spec, t0)))
            next_at += rng.expovariate(rate)
        results = await asyncio.gather(*tasks)
        return list(results), time.perf_counter() - t0


async def closed_loop(
    url: str, model: str, specs: list[RequestSpec], concurrency: int
) -> tuple[list[RequestResult], float]:
    queue: asyncio.Queue[RequestSpec] = asyncio.Queue()
    for s in specs:
        queue.put_nowait(s)
    results: list[RequestResult] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        t0 = time.perf_counter()

        async def user() -> None:
            while not queue.empty():
                results.append(await send(client, url, model, queue.get_nowait(), t0))

        await asyncio.gather(*(user() for _ in range(concurrency)))
        return results, time.perf_counter() - t0


def summarize_run(results: list[RequestResult], wall: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if not math.isnan(r.ttft_s)]
    e2e = [r.e2e_s for r in ok]
    tpot = [
        (r.e2e_s - r.ttft_s) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1 and not math.isnan(r.ttft_s)
    ]
    out_tokens = sum(r.output_tokens for r in ok)

    def pct(xs: list[float]) -> dict[str, float]:
        return {
            "p50": percentile(xs, 50),
            "p90": percentile(xs, 90),
            "p99": percentile(xs, 99),
            "mean": statistics.fmean(xs) if xs else math.nan,
        }

    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "error_statuses": sorted({r.status for r in results if not r.ok}),
        "wall_s": wall,
        "output_tok_s": out_tokens / wall,
        "req_s": len(ok) / wall,
        "output_tokens": out_tokens,
        "prompt_tokens": sum(r.prompt_tokens for r in ok),
        "ttft_s": pct(ttft),
        "tpot_s": pct(tpot),
        "e2e_s": pct(e2e),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rates", default="", help="comma-separated open-loop arrival rates (req/s)")
    ap.add_argument("--concurrency", default="", help="comma-separated closed-loop user counts")
    ap.add_argument("--num-requests", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="loadgen")
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()
    specs = make_workload(args.num_requests, args.seed)
    points: list[dict[str, Any]] = []
    for rate in [float(x) for x in args.rates.split(",") if x]:
        before = load1()
        res, wall = asyncio.run(open_loop(args.url, args.model, specs, rate, args.seed))
        point = {
            "mode": "open",
            "rate": rate,
            **summarize_run(res, wall),
            "load1_before": before,
            "load1_after": load1(),
        }
        if args.save_raw:
            point["raw"] = [asdict(r) for r in res]
        points.append(point)
        print(
            f"open rate={rate:5.2f}  {point['output_tok_s']:7.1f} tok/s  {point['req_s']:.2f} req/s  "
            f"ttft p50 {point['ttft_s']['p50']:.2f}s p99 {point['ttft_s']['p99']:.2f}s  "
            f"tpot p50 {point['tpot_s']['p50'] * 1000:.0f}ms  err {point['errors']}  load1 {before:.1f}",
            flush=True,
        )
    for c in [int(x) for x in args.concurrency.split(",") if x]:
        before = load1()
        res, wall = asyncio.run(closed_loop(args.url, args.model, specs, c))
        point = {
            "mode": "closed",
            "concurrency": c,
            **summarize_run(res, wall),
            "load1_before": before,
            "load1_after": load1(),
        }
        points.append(point)
        print(
            f"closed c={c:3d}  {point['output_tok_s']:7.1f} tok/s  {point['req_s']:.2f} req/s  "
            f"ttft p50 {point['ttft_s']['p50']:.2f}s  tpot p50 {point['tpot_s']['p50'] * 1000:.0f}ms  "
            f"err {point['errors']}",
            flush=True,
        )
    write_result(
        args.tag,
        {"url": args.url, "model": args.model, "num_requests": args.num_requests, "seed": args.seed, "points": points},
    )


if __name__ == "__main__":
    main()
