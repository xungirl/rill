"""Iteration-level ("continuous") batching scheduler.

Every engine step the scheduler decides, from scratch, which sequences run and
how many of their tokens are computed. Policy (similar to vLLM's v1 scheduler):

  1. RUNNING sequences go first, in arrival order. A decoding sequence needs
     1 token; a sequence still in prefill (chunked) needs the rest of its
     prompt, capped by the remaining token budget. Serving decodes first keeps
     inter-token latency stable while new prompts arrive.
  2. If a running sequence needs a new KV block and none is free, the most
     recently arrived running sequence is PREEMPTED: its blocks are freed and
     it goes back to the front of WAITING with num_computed_tokens = 0, to be
     recomputed later (prompt + tokens generated so far). Recompute costs
     FLOPs but needs no CPU swap space - on a laptop GPU, compute is the
     cheaper resource.
  3. Remaining token budget admits WAITING sequences FCFS. On admission the
     prompt's full blocks are looked up in the prefix cache; hits become
     already-computed tokens, so only the uncached suffix is prefilled. (max_num_seqs caps
     concurrency). With chunked prefill a long prompt is split across steps
     instead of monopolising one step, which bounds the stall it causes for
     everyone else's decode.

`policy="static"` turns this into classic static batching for comparison:
new sequences are admitted only when the running set is empty, so the batch
runs until its longest member finishes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ember.block_manager import BlockManager
from ember.model_runner import ScheduledSeq
from ember.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 1024
    chunked_prefill: bool = True
    policy: str = "continuous"  # or "static"

    def __post_init__(self) -> None:
        if self.policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {self.policy!r}")
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("limits must be positive")


@dataclass
class SchedulerOutput:
    batch: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.batch)


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        self.prefix_hit_tokens = 0  # prompt tokens served from the prefix cache
        self.prefix_query_tokens = 0  # prompt tokens looked up
        self._static_admitting = False  # static policy: is the current batch still being formed?

    # ----------------------------------------------------------------- queue ops
    def add(self, seq: Sequence) -> None:
        if not self.config.chunked_prefill and seq.num_tokens > self.config.max_num_batched_tokens:
            raise ValueError("prompt longer than max_num_batched_tokens and chunked prefill is off")
        if self.bm.blocks_for(seq.num_tokens + seq.sampling.max_tokens) > self.bm.num_blocks:
            raise ValueError("request can never fit in the KV cache")
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        """Mark seq finished (stop token, length, abort...) and release its blocks."""
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        self.bm.free(seq)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    # ----------------------------------------------------------------- schedule
    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens

        # 1) running sequences (decode, or the next chunk of an unfinished prefill)
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            # make room by preempting from the back (newest first) until seq fits
            while not self.bm.can_allocate(seq, n):
                victim = self.running[-1]
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            if seq.status is not SequenceStatus.RUNNING:
                break  # seq preempted itself: nothing behind it remains either
            self.bm.allocate_slots(seq, n)
            out.batch.append(ScheduledSeq(seq, n))
            budget -= n
            i += 1

        # 2) admit waiting sequences (not in a step that had to preempt: memory is tight)
        static = self.config.policy == "static"
        if static and not self.running:
            self._static_admitting = True  # previous static batch fully drained
        if out.preempted or (static and not self._static_admitting):
            return out
        while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            cached = self.bm.match_prefix(seq)  # shares cached full blocks, if any
            seq.num_computed_tokens = cached
            n = seq.num_tokens - cached
            if n > budget and not self.config.chunked_prefill:
                self._undo_prefix_match(seq)
                break
            n = min(n, budget)
            if not self.bm.can_allocate(seq, n):
                self._undo_prefix_match(seq)
                break
            self.waiting.popleft()
            seq.num_cached_tokens = cached
            self.prefix_hit_tokens += cached
            self.prefix_query_tokens += seq.num_tokens
            self.bm.allocate_slots(seq, n)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.batch.append(ScheduledSeq(seq, n))
            budget -= n
        if static and (len(self.running) >= self.config.max_num_seqs or not self.waiting):
            self._static_admitting = False  # batch is formed; no joins until it drains
        return out

    def _undo_prefix_match(self, seq: Sequence) -> None:
        self.bm.free(seq)
        seq.num_computed_tokens = 0

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)
