"""Baseline: a naive OpenAI-style server around Hugging Face `transformers.generate`.

What a quick prototype would do: one global model, an asyncio.Lock so requests
run one at a time (generate() has no request-level batching), streaming via
TextIteratorStreamer from a worker thread. Same model weights and dtype as
rill (Qwen3-0.6B, fp16, MPS). Only /v1/completions with streaming, greedy.

Run: uv run python -m benchmarks.hf_server --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from rill.loader import resolve_model_dir
from rill.server.protocol import CompletionRequest, new_id, now


def create_app(device: str) -> FastAPI:
    path = str(resolve_model_dir())
    tok: Any = AutoTokenizer.from_pretrained(path)
    model: Any = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
    model = model.to(device).eval()
    lock = asyncio.Lock()
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> StreamingResponse:
        prompt = req.prompt if isinstance(req.prompt, str) else tok.decode(req.prompt)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        async def body() -> AsyncIterator[str]:
            async with lock:  # one request at a time
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                kwargs = {"input_ids": ids, "max_new_tokens": req.max_tokens, "do_sample": False, "streamer": streamer}
                if req.ignore_eos:
                    kwargs["min_new_tokens"] = req.max_tokens
                out: dict[str, Any] = {}

                def work() -> None:
                    with torch.inference_mode():
                        out["seq"] = model.generate(**kwargs)

                thread = threading.Thread(target=work)
                thread.start()
                rid = new_id("cmpl")
                it = iter(streamer)
                while True:
                    text = await asyncio.to_thread(next, it, None)
                    if text is None:
                        break
                    if text:
                        chunk = {
                            "id": rid,
                            "object": "text_completion",
                            "created": now(),
                            "choices": [{"index": 0, "text": text, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                thread.join()
                n_out = int(out["seq"].shape[1] - ids.shape[1])
                usage = {
                    "prompt_tokens": int(ids.shape[1]),
                    "completion_tokens": n_out,
                    "total_tokens": int(ids.shape[1]) + n_out,
                }
                yield f"data: {json.dumps({'id': rid, 'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    uvicorn.run(create_app(args.device), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
