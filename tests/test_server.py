"""HTTP API tests against a real uvicorn server (tiny random model, CPU, localhost).

A real socket server (not an in-process ASGI transport) is used so that a
client closing its connection is a real TCP disconnect.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from rill.config import tiny_config
from rill.engine import EngineConfig, LLMEngine
from rill.loader import random_model
from rill.scheduler import SchedulerConfig
from rill.server.api import create_app
from rill.server.async_engine import AsyncLLMEngine, _stop_holdback
from rill.tokenizer import Tokenizer
from tests.tiny_tokenizer import IM_END, write_tiny_tokenizer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def serve(tmp: Path, max_inflight: int = 64, timeout_s: float | None = 60.0) -> Iterator[tuple[str, AsyncLLMEngine]]:
    cfg = tiny_config()
    cfg = type(cfg)(**{**cfg.__dict__, "eos_token_ids": (IM_END,)})
    engine = LLMEngine(
        random_model(cfg, seed=0),
        EngineConfig(block_size=16, num_blocks=2048, scheduler=SchedulerConfig(max_num_seqs=8)),
    )
    aengine = AsyncLLMEngine(engine, Tokenizer(write_tiny_tokenizer(tmp / "tok")), max_inflight=max_inflight)
    app = create_app(aengine, "tiny", request_timeout_s=timeout_s, max_model_len=30000)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url + "/health").status_code == 200:
                break
        time.sleep(0.02)
    try:
        yield url, aengine
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, AsyncLLMEngine]]:
    with serve(tmp_path_factory.mktemp("srv")) as s:
        yield s


def sse_events(resp: httpx.Response) -> list[Any]:
    events: list[Any] = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def wait_until(cond: Any, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


GREEDY = {"temperature": 0, "max_tokens": 12}


def test_health_models_metrics(server: tuple[str, AsyncLLMEngine]) -> None:
    url, _ = server
    assert httpx.get(url + "/health").json()["status"] == "ok"
    assert httpx.get(url + "/v1/models").json()["data"][0]["id"] == "tiny"
    assert "ember_ttft_seconds" in httpx.get(url + "/metrics").text


def test_completion_and_streaming_agree(server: tuple[str, AsyncLLMEngine]) -> None:
    url, _ = server
    body = {"prompt": "hello world", **GREEDY, "ignore_eos": True}
    r = httpx.post(url + "/v1/completions", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "text_completion"
    assert data["usage"]["completion_tokens"] == 12
    assert data["choices"][0]["finish_reason"] == "length"
    text = data["choices"][0]["text"]
    assert text  # random model emits visible tokens

    with httpx.stream(
        "POST", url + "/v1/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}}
    ) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = sse_events(resp)
    assert events[-1] == "[DONE]"
    usage = events[-2]["usage"]
    assert usage["completion_tokens"] == 12 and usage["prompt_tokens"] == data["usage"]["prompt_tokens"]
    chunks = [e for e in events[:-2] if e["choices"]]
    assert "".join(c["choices"][0]["text"] for c in chunks) == text
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_token_id_prompt(server: tuple[str, AsyncLLMEngine]) -> None:
    url, aengine = server
    ids = aengine.tokenizer.encode("hello world")
    a = httpx.post(url + "/v1/completions", json={"prompt": ids, **GREEDY, "ignore_eos": True}).json()
    b = httpx.post(url + "/v1/completions", json={"prompt": "hello world", **GREEDY, "ignore_eos": True}).json()
    assert a["choices"][0]["text"] == b["choices"][0]["text"]


def test_chat_completion_and_stream(server: tuple[str, AsyncLLMEngine]) -> None:
    url, _ = server
    body = {"messages": [{"role": "user", "content": "hi"}], **GREEDY, "ignore_eos": True}
    data = httpx.post(url + "/v1/chat/completions", json=body).json()
    assert data["object"] == "chat.completion"
    msg = data["choices"][0]["message"]
    assert msg["role"] == "assistant"
    with httpx.stream("POST", url + "/v1/chat/completions", json={**body, "stream": True}) as resp:
        events = sse_events(resp)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(e["object"] == "chat.completion.chunk" for e in events[:-1])
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:-1])
    assert content == msg["content"]


def test_stop_sequence_truncates_and_frees(server: tuple[str, AsyncLLMEngine]) -> None:
    url, aengine = server
    body = {"prompt": "hello world", "temperature": 0, "max_tokens": 30, "ignore_eos": True}
    full = httpx.post(url + "/v1/completions", json=body).json()["choices"][0]["text"]
    assert len(full) > 13, full  # precondition: the random model produced visible text
    stop = full[10:13]
    expected = full[: full.index(stop)]
    got = httpx.post(url + "/v1/completions", json={**body, "stop": [stop, "\x00never"]}).json()
    assert got["choices"][0]["text"] == expected
    assert got["choices"][0]["finish_reason"] == "stop"
    with httpx.stream("POST", url + "/v1/completions", json={**body, "stop": stop, "stream": True}) as resp:
        events = sse_events(resp)
    streamed = "".join(e["choices"][0]["text"] for e in events if e != "[DONE]")
    assert streamed == expected  # the stop string never leaks into the stream, even partially
    assert wait_until(lambda: not aengine.engine.has_unfinished())


def test_stop_holdback() -> None:
    assert _stop_holdback("hello wor", ["world"]) == 3
    assert _stop_holdback("hello", ["xyz"]) == 0
    assert _stop_holdback("ab", ["abc", "bq"]) == 2


def test_client_disconnect_aborts_sequence(server: tuple[str, AsyncLLMEngine]) -> None:
    url, aengine = server
    body = {"prompt": "long", "temperature": 0, "max_tokens": 20000, "ignore_eos": True, "stream": True}
    with httpx.stream("POST", url + "/v1/completions", json=body) as resp:
        it = resp.iter_lines()
        got = [next(it) for _ in range(6)]
        assert any(line.startswith("data: ") for line in got)
        assert aengine.engine.has_unfinished()
    # connection closed mid-stream: the engine must drop the sequence and free its blocks
    assert wait_until(lambda: not aengine.engine.has_unfinished())
    assert wait_until(
        lambda: aengine.engine.block_manager.num_used_blocks == 0 or aengine.engine.block_manager.enable_prefix_caching
    )
    assert 'ember_requests_total{finish_reason="abort"}' in httpx.get(url + "/metrics").text


def test_validation_errors(server: tuple[str, AsyncLLMEngine]) -> None:
    url, _ = server
    assert httpx.post(url + "/v1/completions", json={"prompt": "x", "top_k": 0}).status_code == 400
    assert httpx.post(url + "/v1/completions", json={"prompt": "x", "temperature": 3.0}).status_code == 422
    assert httpx.post(url + "/v1/completions", json={"prompt": "x", "max_tokens": 0}).status_code == 422
    r = httpx.post(url + "/v1/completions", json={"prompt": "x", "temperature": 0, "max_tokens": 40000})
    assert r.status_code == 400 and "exceeds" in r.json()["error"]["message"]
    assert httpx.post(url + "/v1/completions", json={"prompt": [], "temperature": 0}).status_code == 400


def test_backpressure_returns_429(tmp_path: Path) -> None:
    with serve(tmp_path, max_inflight=2) as (url, aengine):
        body = {"prompt": "x", "temperature": 0, "max_tokens": 20000, "ignore_eos": True, "stream": True}
        with (
            httpx.stream("POST", url + "/v1/completions", json=body) as a,
            httpx.stream("POST", url + "/v1/completions", json=body) as b,
        ):
            # keep the iterators referenced: a discarded iter_lines() closes the stream (= disconnect)
            lines_a, lines_b = a.iter_lines(), b.iter_lines()
            next(lines_a)
            next(lines_b)
            r = httpx.post(url + "/v1/completions", json={**body, "stream": False, "max_tokens": 5})
            assert r.status_code == 429
            assert r.json()["error"]["type"] == "rate_limit_exceeded"
        assert wait_until(lambda: aengine.inflight == 0)
        ok = httpx.post(url + "/v1/completions", json={**body, "stream": False, "max_tokens": 5})
        assert ok.status_code == 200  # capacity is released after the streams close
        assert "ember_rejected_total 1.0" in httpx.get(url + "/metrics").text


def test_request_timeout_returns_504_and_frees(tmp_path: Path) -> None:
    with serve(tmp_path, timeout_s=0.3) as (url, aengine):
        body = {"prompt": "x", "temperature": 0, "max_tokens": 20000, "ignore_eos": True}
        r = httpx.post(url + "/v1/completions", json=body, timeout=30)
        assert r.status_code == 504
        assert wait_until(lambda: not aengine.engine.has_unfinished())


def test_seeded_sampling_is_reproducible_over_http(server: tuple[str, AsyncLLMEngine]) -> None:
    url, _ = server
    body = {"prompt": "hello world", "temperature": 1.0, "top_p": 0.9, "max_tokens": 24, "ignore_eos": True}

    def text(seed: int) -> str:
        r = httpx.post(url + "/v1/completions", json={**body, "seed": seed})
        assert r.status_code == 200
        return str(r.json()["choices"][0]["text"])

    assert text(7) == text(7)
    assert len({text(s) for s in range(5)}) > 1  # different seeds explore different samples
