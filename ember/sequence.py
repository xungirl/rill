"""Per-request state tracked by the scheduler."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from ember.sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"  # queued, or preempted and waiting to be recomputed
    RUNNING = "running"  # holds KV blocks, scheduled every step
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    """One generation request.

    `token_ids` = prompt + generated tokens so far. `num_computed_tokens` counts
    how many of them already have K/V written in the cache. The scheduler only
    ever reasons about the gap between the two:
        prefill  : num_tokens - num_computed_tokens  > 1  (possibly in chunks)
        decode   : num_tokens - num_computed_tokens == 1  (the token just sampled)
    This single counter is what makes chunked prefill, prefix-cache hits and
    preemption-by-recompute all the same code path.
    """

    seq_id: int
    prompt_token_ids: list[int]
    sampling: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    token_ids: list[int] = field(init=False)
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0  # prompt tokens served from the prefix cache (stats)
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: str | None = None
    num_preemptions: int = 0
    first_token_time: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.token_ids = list(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
