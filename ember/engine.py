"""LLMEngine: the synchronous core loop. schedule -> execute -> sample -> update.

The engine is single-threaded and knows nothing about HTTP. The async server
(ember/server) runs `step()` in a dedicated thread and feeds it requests.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import torch

from ember.block_manager import BlockManager
from ember.model import Qwen3Model
from ember.model_runner import ModelRunner
from ember.paged_attention import PagedKVCache
from ember.sampling import SamplingParams, sample
from ember.scheduler import Scheduler, SchedulerConfig
from ember.sequence import Sequence


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int | None = None  # if None, derived from kv_cache_gib
    kv_cache_gib: float = 2.0
    enable_prefix_caching: bool = True
    seed: int = 0  # RNG for requests that sample without their own seed
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class StepOutput:
    """What happened to one sequence during one engine step."""

    seq_id: int
    new_token_id: int | None  # None if the step was a mid-prompt prefill chunk
    finished: bool
    finish_reason: str | None


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    kv_usage_samples: list[float] = field(default_factory=list)


class LLMEngine:
    def __init__(self, model: Qwen3Model, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.model = model
        cfg = model.cfg
        dtype = model.embed_tokens.weight.dtype
        device = model.embed_tokens.weight.device
        num_blocks = self.config.num_blocks or PagedKVCache.num_blocks_for_budget(
            cfg, int(self.config.kv_cache_gib * 2**30), self.config.block_size, dtype
        )
        self.cache = PagedKVCache(cfg, num_blocks, self.config.block_size, dtype, device)
        self.block_manager = BlockManager(num_blocks, self.config.block_size, self.config.enable_prefix_caching)
        self.scheduler = Scheduler(self.config.scheduler, self.block_manager)
        self.runner = ModelRunner(model, self.cache)
        self.eos_ids = frozenset(cfg.eos_token_ids)
        self.seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()
        self.stats = EngineStats()
        self._rng = torch.Generator().manual_seed(self.config.seed)

    # ---------------------------------------------------------------- requests
    def add_request(self, prompt_token_ids: list[int], sampling: SamplingParams | None = None) -> int:
        seq = Sequence(next(self._ids), list(prompt_token_ids), sampling or SamplingParams())
        seed = seq.sampling.seed
        seq.generator = torch.Generator().manual_seed(seed) if seed is not None else self._rng
        self.scheduler.add(seq)
        self.seqs[seq.seq_id] = seq
        return seq.seq_id

    def abort(self, seq_id: int) -> None:
        seq = self.seqs.pop(seq_id, None)
        if seq is not None and not seq.is_finished:
            self.scheduler.finish(seq, "abort")

    def has_unfinished(self) -> bool:
        return self.scheduler.num_unfinished > 0

    # -------------------------------------------------------------------- step
    def step(self) -> list[StepOutput]:
        sched = self.scheduler.schedule()
        if not sched.batch:
            return []
        logits = self.runner.execute(sched.batch)
        sampling_rows = [s for s in sched.batch if s.completes_sequence]
        token_ids = (
            sample(logits, [s.seq.sampling for s in sampling_rows], [s.seq.generator for s in sampling_rows])
            if sampling_rows
            else []
        )
        tokens = iter(token_ids)

        now = time.monotonic()
        outputs: list[StepOutput] = []
        for item in sched.batch:
            seq = item.seq
            completes = item.completes_sequence
            seq.num_computed_tokens += item.num_new_tokens
            self.block_manager.cache_full_blocks(seq)  # K/V now written -> shareable
            if seq.num_output_tokens == 0:
                self.stats.prefill_tokens += item.num_new_tokens
            else:
                self.stats.decode_tokens += item.num_new_tokens
            if not completes:
                outputs.append(StepOutput(seq.seq_id, None, False, None))
                continue
            tok = next(tokens)
            seq.append_token(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            reason = self._stop_reason(seq, tok)
            if reason is not None:
                self.scheduler.finish(seq, reason)
            outputs.append(StepOutput(seq.seq_id, tok, reason is not None, reason))
        self.stats.steps += 1
        self.stats.kv_usage_samples.append(self.block_manager.usage())
        return outputs

    def _stop_reason(self, seq: Sequence, tok: int) -> str | None:
        if tok in self.eos_ids and not seq.sampling.ignore_eos:
            return "stop"
        if seq.num_output_tokens >= seq.sampling.max_tokens:
            return "length"
        return None

    # ------------------------------------------------------------ offline API
    @torch.inference_mode()
    def generate(self, prompts: list[list[int]], sampling: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        """Run a whole list of prompts to completion; returns output token ids in input order."""
        params = sampling if isinstance(sampling, list) else [sampling] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        while self.has_unfinished():
            self.step()
        return [self.seqs.pop(i).output_token_ids for i in ids]
