"""OpenAI-compatible HTTP API: /v1/completions, /v1/chat/completions (+ SSE streaming).

Each handler turns the request into prompt token ids + SamplingParams, then
consumes `AsyncLLMEngine.generate`. Streaming responses are Server-Sent Events:
`data: {json}\\n\\n` per chunk, terminated by `data: [DONE]`.

Client disconnect: for streams, Starlette cancels the response generator when
the client goes away; for non-streaming requests a watcher task polls
`request.is_disconnected()` and cancels the generation task. Either way
`AsyncLLMEngine.generate`'s `finally` sends an abort, so the sequence's KV blocks
are freed within one engine step.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ember.sampling import SamplingParams
from ember.server.async_engine import (
    AsyncLLMEngine,
    EngineDeadError,
    QueueFullError,
    RequestOutput,
    RequestTimeoutError,
)
from ember.server.protocol import ChatCompletionRequest, CompletionRequest, Usage, new_id, now


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)


def _map_error(e: BaseException) -> JSONResponse:
    if isinstance(e, QueueFullError):
        return _error(429, str(e), "rate_limit_exceeded")
    if isinstance(e, RequestTimeoutError):
        return _error(504, str(e), "timeout")
    if isinstance(e, EngineDeadError):
        return _error(503, str(e), "engine_dead")
    if isinstance(e, ValueError):
        return _error(400, str(e), "invalid_request_error")
    raise e


def _sse(obj: dict[str, Any] | str) -> str:
    return f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n"


async def _watch_disconnect(request: Request, task: asyncio.Task[Any], poll_s: float = 0.1) -> None:
    while not task.done():
        if await request.is_disconnected():
            task.cancel()
            return
        await asyncio.sleep(poll_s)


def create_app(
    engine: AsyncLLMEngine,
    model_name: str,
    request_timeout_s: float | None = 600.0,
    max_model_len: int = 4096,
) -> FastAPI:
    tok = engine.tokenizer

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        engine.start()  # engine thread lives exactly as long as the app
        yield
        engine.shutdown()

    app = FastAPI(title="ember", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    def sampling_for(req: CompletionRequest | ChatCompletionRequest, prompt_len: int) -> SamplingParams:
        if prompt_len + req.max_tokens > max_model_len:
            raise ValueError(f"prompt ({prompt_len}) + max_tokens ({req.max_tokens}) exceeds {max_model_len}")
        return SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
            ignore_eos=req.ignore_eos,
        )

    async def run(
        request: Request,
        req: CompletionRequest | ChatCompletionRequest,
        prompt_ids: list[int],
        render: Callable[[str, RequestOutput | None, bool], dict[str, Any]],
        render_final: Callable[[str, Usage, str | None], dict[str, Any]],
    ) -> Response:
        try:
            sampling = sampling_for(req, len(prompt_ids))
        except ValueError as e:
            return _map_error(e)
        rid = new_id("cmpl" if isinstance(req, CompletionRequest) else "chatcmpl")
        gen = engine.generate(rid, prompt_ids, sampling, req.stop_list(), request_timeout_s)

        if req.stream:
            # Pull the first item before answering so 429/400 become real HTTP errors.
            try:
                first = await anext(gen)
            except StopAsyncIteration:  # pragma: no cover - generate always yields a final item
                first = None
            except (QueueFullError, RequestTimeoutError, EngineDeadError, ValueError) as e:
                return _map_error(e)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _stream_body(request, rid, gen, first, render, render_final, include_usage, len(prompt_ids)),
                media_type="text/event-stream",
            )

        async def collect() -> tuple[str, RequestOutput | None]:
            text, last = [], None
            async for out in gen:
                text.append(out.text)
                last = out
            return "".join(text), last

        task = asyncio.create_task(collect())
        watcher = asyncio.create_task(_watch_disconnect(request, task))
        try:
            text, last = await task
        except asyncio.CancelledError:
            return _error(499, "client disconnected", "cancelled")
        except (QueueFullError, RequestTimeoutError, EngineDeadError, ValueError) as e:
            return _map_error(e)
        finally:
            watcher.cancel()
        assert last is not None
        usage = Usage(
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens,
            total_tokens=last.prompt_tokens + last.completion_tokens,
        )
        body = render(rid, RequestOutput(text, True, last.finish_reason, 0, 0), False)
        body["usage"] = usage.model_dump()
        return JSONResponse(body)

    async def _stream_body(
        request: Request,
        rid: str,
        gen: AsyncGenerator[RequestOutput],
        first: RequestOutput | None,
        render: Callable[[str, RequestOutput | None, bool], dict[str, Any]],
        render_final: Callable[[str, Usage, str | None], dict[str, Any]],
        include_usage: bool,
        prompt_len: int,
    ) -> AsyncIterator[str]:
        out_q: asyncio.Queue[RequestOutput | BaseException | None] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for out in gen:
                    await out_q.put(out)
                await out_q.put(None)
            except asyncio.CancelledError:
                await gen.aclose()  # runs generate()'s finally -> abort in the engine
                raise
            except BaseException as e:
                await out_q.put(e)

        yield _sse(render(rid, None, True))  # role / header chunk
        last = first
        if first is not None:
            yield _sse(render(rid, first, True))
        # No disconnect watcher here: Starlette's StreamingResponse already listens for
        # http.disconnect and cancels this generator, which cancels pump() below.
        task = asyncio.create_task(pump()) if not (first and first.finished) else None
        try:
            while task is not None:
                item = await out_q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    yield _sse({"error": {"message": str(item), "type": type(item).__name__}})
                    break
                last = item
                yield _sse(render(rid, item, True))
            if include_usage and last is not None:
                usage = Usage(
                    prompt_tokens=prompt_len,
                    completion_tokens=last.completion_tokens,
                    total_tokens=prompt_len + last.completion_tokens,
                )
                yield _sse(render_final(rid, usage, last.finish_reason))
            yield _sse("[DONE]")
        finally:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            elif task is None:
                await gen.aclose()

    # ---------------------------------------------------------------- routes
    @app.post("/v1/completions", response_model=None)
    async def completions(req: CompletionRequest, request: Request) -> Response:
        prompt_ids = req.prompt if isinstance(req.prompt, list) else tok.encode(req.prompt)
        if not prompt_ids:
            return _error(400, "prompt must not be empty", "invalid_request_error")

        def render(rid: str, out: RequestOutput | None, chunk: bool) -> dict[str, Any]:
            choice = {
                "index": 0,
                "text": "" if out is None else out.text,
                "logprobs": None,
                "finish_reason": None if out is None else out.finish_reason,
            }
            return {"id": rid, "object": "text_completion", "created": now(), "model": model_name, "choices": [choice]}

        def render_final(rid: str, usage: Usage, reason: str | None) -> dict[str, Any]:
            body = {"id": rid, "object": "text_completion", "created": now(), "model": model_name, "choices": []}
            return {**body, "usage": usage.model_dump()}

        return await run(request, req, list(prompt_ids), render, render_final)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Response:
        prompt = tok.apply_chat_template([m.model_dump() for m in req.messages], enable_thinking=False)
        prompt_ids = tok.encode(prompt)

        def render(rid: str, out: RequestOutput | None, chunk: bool) -> dict[str, Any]:
            base = {"id": rid, "created": now(), "model": model_name}
            reason = None if out is None else out.finish_reason
            if not chunk:
                message = {"role": "assistant", "content": "" if out is None else out.text}
                choice = {"index": 0, "message": message, "finish_reason": reason}
                return {**base, "object": "chat.completion", "choices": [choice]}
            delta = {"role": "assistant", "content": ""} if out is None else {"content": out.text}
            choice = {"index": 0, "delta": delta, "finish_reason": reason}
            return {**base, "object": "chat.completion.chunk", "choices": [choice]}

        def render_final(rid: str, usage: Usage, reason: str | None) -> dict[str, Any]:
            base = {"id": rid, "object": "chat.completion.chunk", "created": now(), "model": model_name}
            return {**base, "choices": [], "usage": usage.model_dump()}

        return await run(request, req, prompt_ids, render, render_final)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "ember"}]}

    @app.get("/health", response_model=None)
    async def health() -> Response:
        if engine.dead is not None:
            return _error(503, str(engine.dead), "engine_dead")
        return JSONResponse({"status": "ok", "inflight": engine.inflight})

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(engine.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
