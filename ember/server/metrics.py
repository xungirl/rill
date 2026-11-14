"""Prometheus metrics. One registry per server instance (tests build several apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Bucket edges chosen for a laptop: TTFT spans ~20 ms (idle) to tens of seconds (saturated).
_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)
_TOKEN_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.ttft = Histogram("ember_ttft_seconds", "Time to first token", buckets=_LATENCY_BUCKETS, registry=r)
        self.itl = Histogram("ember_itl_seconds", "Inter-token latency", buckets=_TOKEN_BUCKETS, registry=r)
        self.tpot = Histogram(
            "ember_tpot_seconds", "Mean time per output token after the first", buckets=_TOKEN_BUCKETS, registry=r
        )
        self.e2e = Histogram("ember_e2e_seconds", "Request latency", buckets=_LATENCY_BUCKETS, registry=r)
        self.requests = Counter("ember_requests_total", "Finished requests", ["finish_reason"], registry=r)
        self.rejected = Counter("ember_rejected_total", "Requests rejected (429)", registry=r)
        self.prompt_tokens = Counter("ember_prompt_tokens_total", "Prompt tokens", registry=r)
        self.generation_tokens = Counter("ember_generation_tokens_total", "Generated tokens", registry=r)
        self.waiting = Gauge("ember_queue_depth", "Sequences waiting for admission", registry=r)
        self.running = Gauge("ember_running_sequences", "Sequences in the running batch", registry=r)
        self.inflight = Gauge("ember_inflight_requests", "Requests accepted and not yet finished", registry=r)
        self.kv_usage = Gauge("ember_kv_cache_usage", "Fraction of KV blocks in use", registry=r)
        self.prefix_hit_rate = Gauge("ember_prefix_cache_hit_rate", "Cumulative prefix-cache hit rate", registry=r)
        self.preemptions = Gauge("ember_preemptions", "Cumulative preemptions", registry=r)
        self.step_time = Histogram(
            "ember_engine_step_seconds", "Engine step duration", buckets=_TOKEN_BUCKETS, registry=r
        )
