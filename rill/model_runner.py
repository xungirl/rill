"""Turns one scheduled batch into one forward pass over the paged KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rill.model import Qwen3Model
from rill.paged_attention import PagedAttention, PagedAttentionMetadata, PagedKVCache
from rill.sequence import Sequence


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int  # tokens of seq computed this step (prefill chunk size, or 1 for decode)

    @property
    def completes_sequence(self) -> bool:
        """True if after this step every token of seq has K/V -> we must sample the next one."""
        return self.seq.num_computed_tokens + self.num_new_tokens == self.seq.num_tokens


class ModelRunner:
    def __init__(self, model: Qwen3Model, cache: PagedKVCache) -> None:
        self.model = model
        self.cache = cache
        self.device = model.embed_tokens.weight.device

    @torch.inference_mode()
    def execute(self, batch: list[ScheduledSeq]) -> Tensor:
        """Run the model on the new tokens of every scheduled sequence.

        Returns float32 logits [N, V] for the N sequences with
        `completes_sequence` (in batch order). Rows for mid-prompt prefill
        chunks are never projected to the vocabulary.
        """
        input_ids: list[int] = []
        positions: list[int] = []
        last_index: list[int] = []  # packed index of each sampling row's last token
        for s in batch:
            start = s.seq.num_computed_tokens
            end = start + s.num_new_tokens
            input_ids.extend(s.seq.token_ids[start:end])
            positions.extend(range(start, end))
            if s.completes_sequence:
                last_index.append(len(input_ids) - 1)
        meta = PagedAttentionMetadata.build(
            block_tables=[s.seq.block_table for s in batch],
            query_lens=[s.num_new_tokens for s in batch],
            context_lens=[s.seq.num_computed_tokens + s.num_new_tokens for s in batch],
            block_size=self.cache.block_size,
            device=self.device,
        )
        ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
        hidden, _ = self.model(ids_t, pos_t, PagedAttention(self.cache, meta))
        if not last_index:
            return hidden.new_empty(0, self.model.cfg.vocab_size, dtype=torch.float32)
        rows = hidden[torch.tensor(last_index, device=self.device)]
        return self.model.compute_logits(rows).float()
