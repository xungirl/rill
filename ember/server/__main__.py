"""Launch the server: uv run python -m ember.server --device mps --port 8000"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
import uvicorn

from ember.engine import EngineConfig, LLMEngine
from ember.loader import DEFAULT_MODEL, load_model, resolve_model_dir
from ember.quant import quantize_int8
from ember.scheduler import SchedulerConfig
from ember.server.api import create_app
from ember.server.async_engine import AsyncLLMEngine
from ember.tokenizer import Tokenizer

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ember OpenAI-compatible server")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", choices=DTYPES, default=None, help="default: float16 on mps, float32 on cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--kv-cache-gib", type=float, default=2.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--max-inflight", type=int, default=128, help="accepted requests before 429")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--int8", action="store_true", help="weight-only int8 for the decoder linears")
    args = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    dtype = DTYPES[args.dtype] if args.dtype else (torch.float16 if args.device == "mps" else torch.float32)
    model = load_model(args.model, device=args.device, dtype=dtype)
    if args.int8:
        model = quantize_int8(model)
    engine = LLMEngine(
        model,
        EngineConfig(
            block_size=args.block_size,
            kv_cache_gib=args.kv_cache_gib,
            enable_prefix_caching=not args.no_prefix_caching,
            scheduler=SchedulerConfig(
                max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens
            ),
        ),
    )
    async_engine = AsyncLLMEngine(engine, Tokenizer(resolve_model_dir(args.model)), max_inflight=args.max_inflight)
    app = create_app(async_engine, args.model, args.request_timeout, args.max_model_len)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
