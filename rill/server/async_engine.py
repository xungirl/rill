"""Bridges the synchronous LLMEngine to asyncio.

Threading model
---------------
* ONE engine thread owns the LLMEngine and is the only code that touches it,
  so the engine needs no locks. It loops: drain the inbox -> engine.step() ->
  route each StepOutput to its request's asyncio.Queue.
* HTTP handlers run on the asyncio event loop. They never block on the GPU:
  they put ("add" | "abort", stream) commands into a thread-safe inbox and
  `await` their own queue, which the engine thread fills with
  `loop.call_soon_threadsafe` (the only safe way to touch asyncio objects from
  another thread).
* Per-request work that needs text (detokenization, stop strings) runs on the
  event loop, keeping the engine thread's step loop tight.

Back-pressure: at most `max_inflight` accepted requests; beyond that
`generate` raises QueueFullError (HTTP 429) instead of queueing unboundedly.
Cancellation: leaving `generate` early (client disconnect, timeout, stop
string) sends an abort so the sequence's KV blocks are freed immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from rill.engine import LLMEngine, StepOutput
from rill.sampling import SamplingParams
from rill.server.metrics import Metrics
from rill.tokenizer import IncrementalDetokenizer, Tokenizer

log = logging.getLogger("rill.request")


class QueueFullError(RuntimeError):
    pass


class RequestTimeoutError(RuntimeError):
    pass


class EngineDeadError(RuntimeError):
    pass


@dataclass
class RequestOutput:
    text: str  # new text since the previous output
    finished: bool
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


@dataclass(eq=False)
class _Stream:
    """One request's mailbox between the engine thread and the event loop."""

    request_id: str
    prompt_ids: list[int]
    sampling: SamplingParams
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[StepOutput | BaseException] = field(default_factory=asyncio.Queue)
    seq_id: int | None = None
    cancelled: bool = False

    def push(self, item: StepOutput | BaseException) -> None:  # called from the engine thread
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


def _stop_holdback(text: str, stops: list[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a stop string.
    That much text must not be streamed yet: the next token might complete the stop."""
    best = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), best, -1):
            if text.endswith(s[:k]):
                best = k
                break
    return best


class AsyncLLMEngine:
    def __init__(
        self,
        engine: LLMEngine,
        tokenizer: Tokenizer,
        metrics: Metrics | None = None,
        max_inflight: int = 64,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.metrics = metrics or Metrics()
        self.max_inflight = max_inflight
        self.inflight = 0  # only touched on the event loop
        self.dead: BaseException | None = None
        self._inbox: queue.Queue[tuple[str, _Stream]] = queue.Queue()
        self._by_seq: dict[int, _Stream] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rill-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------------------------------------------------------- engine thread
    def _loop(self) -> None:
        while self._running:
            self._drain_inbox(block=not self.engine.has_unfinished())
            if not self.engine.has_unfinished():
                continue
            try:
                t = time.perf_counter()
                outputs = self.engine.step()
                self.metrics.step_time.observe(time.perf_counter() - t)
            except Exception as e:  # pragma: no cover - defensive: fail every request, stop serving
                logging.getLogger("rill").exception("engine step failed")
                self.dead = e
                for s in self._by_seq.values():
                    s.push(EngineDeadError(str(e)))
                self._by_seq.clear()
                return
            for out in outputs:
                stream = self._by_seq.get(out.seq_id)
                if stream is None:
                    continue
                stream.push(out)
                if out.finished:
                    del self._by_seq[out.seq_id]
                    self.engine.seqs.pop(out.seq_id, None)
            self._update_gauges()

    def _drain_inbox(self, block: bool) -> None:
        try:
            cmd, stream = self._inbox.get(timeout=0.05) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            if cmd == "add" and not stream.cancelled:
                try:
                    stream.seq_id = self.engine.add_request(stream.prompt_ids, stream.sampling)
                    self._by_seq[stream.seq_id] = stream
                except ValueError as e:
                    stream.push(e)
            elif cmd == "abort" and stream.seq_id is not None and stream.seq_id in self._by_seq:
                self.engine.abort(stream.seq_id)
                del self._by_seq[stream.seq_id]
            try:
                cmd, stream = self._inbox.get_nowait()
            except queue.Empty:
                break
        self._update_gauges()

    def _update_gauges(self) -> None:
        sched = self.engine.scheduler
        self.metrics.waiting.set(len(sched.waiting))
        self.metrics.running.set(len(sched.running))
        self.metrics.kv_usage.set(self.engine.block_manager.usage())
        self.metrics.preemptions.set(sched.num_preemptions)
        if sched.prefix_query_tokens:
            self.metrics.prefix_hit_rate.set(sched.prefix_hit_tokens / sched.prefix_query_tokens)

    # ------------------------------------------------------------ event loop
    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling: SamplingParams,
        stop: list[str] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncGenerator[RequestOutput]:
        """Stream RequestOutputs for one request. Raises QueueFullError,
        RequestTimeoutError, EngineDeadError or ValueError (bad request)."""
        if self.dead is not None:
            raise EngineDeadError(str(self.dead))
        if self.inflight >= self.max_inflight:
            self.metrics.rejected.inc()
            raise QueueFullError(f"{self.inflight} requests in flight (limit {self.max_inflight})")
        stops = [s for s in (stop or []) if s]
        stream = _Stream(request_id, prompt_ids, sampling, asyncio.get_running_loop())
        self.inflight += 1
        self.metrics.inflight.inc()
        self.metrics.prompt_tokens.inc(len(prompt_ids))
        detok = IncrementalDetokenizer(self.tokenizer)
        start = time.monotonic()
        deadline = None if timeout_s is None else start + timeout_s
        first_at: float | None = None
        last_at = start
        n_out = 0
        emitted = 0  # chars of detok.text already sent
        finish_reason: str | None = None
        status = "ok"
        self._inbox.put(("add", stream))
        try:
            while finish_reason is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestTimeoutError(f"request exceeded {timeout_s}s")
                try:
                    item = await asyncio.wait_for(stream.queue.get(), remaining)
                except TimeoutError:
                    raise RequestTimeoutError(f"request exceeded {timeout_s}s") from None
                if isinstance(item, BaseException):
                    raise item
                if item.new_token_id is None:
                    continue  # a prefill chunk; nothing to emit yet
                now = time.monotonic()
                n_out += 1
                self.metrics.generation_tokens.inc()
                if first_at is None:
                    first_at = now
                    self.metrics.ttft.observe(now - start)
                else:
                    self.metrics.itl.observe(now - last_at)
                last_at = now
                is_eos = item.finish_reason == "stop"  # engine stops on EOS: don't print it
                if not is_eos:
                    detok.push(item.new_token_id)
                finish_reason = item.finish_reason
                text = detok.text
                hit = min((i for i in (text.find(s, max(0, emitted - len(s))) for s in stops) if i >= 0), default=-1)
                if hit >= 0:
                    text = text[:hit]
                    finish_reason = "stop"
                    stream.cancelled = True
                    self._inbox.put(("abort", stream))  # stop string: free the sequence now
                    end = len(text)
                elif finish_reason is None:
                    end = len(text) - _stop_holdback(text, stops)
                else:
                    end = len(text)
                delta, emitted = text[emitted:end], max(emitted, end)
                if delta or finish_reason is not None:
                    yield RequestOutput(delta, finish_reason is not None, finish_reason, len(prompt_ids), n_out)
        except BaseException as e:
            status = type(e).__name__
            raise
        finally:
            if finish_reason is None:  # cancelled, timed out or failed: free the KV blocks
                stream.cancelled = True
                self._inbox.put(("abort", stream))
                finish_reason = "abort"
            self.inflight -= 1
            self.metrics.inflight.dec()
            e2e = time.monotonic() - start
            self.metrics.e2e.observe(e2e)
            self.metrics.requests.labels(finish_reason).inc()
            if first_at is not None and n_out > 1:
                self.metrics.tpot.observe((last_at - first_at) / (n_out - 1))
            log.info(
                json.dumps(
                    {
                        "event": "request_done",
                        "request_id": request_id,
                        "status": status,
                        "finish_reason": finish_reason,
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": n_out,
                        "ttft_s": None if first_at is None else round(first_at - start, 4),
                        "e2e_s": round(e2e, 4),
                    }
                )
            )
