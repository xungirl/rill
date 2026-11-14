"""The paged KV path must compute exactly what the contiguous / no-cache paths compute."""

from __future__ import annotations

import random
from collections import OrderedDict

import pytest
import torch

from ember.attention import NoCacheAttention
from ember.block_manager import BlockManager
from ember.model import Qwen3Model
from ember.model_runner import ModelRunner, ScheduledSeq
from ember.paged_attention import PagedAttentionMetadata, PagedKVCache
from ember.sampling import SamplingParams
from ember.sequence import Sequence


def reference_logits(model: Qwen3Model, ids: list[int]) -> torch.Tensor:
    h, _ = model(torch.tensor(ids), torch.arange(len(ids)), NoCacheAttention())
    return model.compute_logits(h)


@torch.no_grad()
@pytest.mark.parametrize("block_size", [1, 3, 4, 16])
@pytest.mark.parametrize("chunk", [5, 1000])
def test_mixed_batches_match_full_forward(tiny_model: Qwen3Model, block_size: int, chunk: int) -> None:
    """Several sequences advance together in packed batches mixing prefill
    chunks (of up to `chunk` tokens) and decode tokens; every logits row must
    equal the full causal forward of that sequence."""
    cfg = tiny_model.cfg
    rng = random.Random(block_size * 7 + chunk)
    bm = BlockManager(num_blocks=200, block_size=block_size)
    order = list(range(200))
    random.Random(1).shuffle(order)
    bm.free_block_ids = OrderedDict.fromkeys(order)  # non-contiguous physical blocks
    cache = PagedKVCache(cfg, 200, block_size, torch.float32, "cpu")
    runner = ModelRunner(tiny_model, cache)

    full = [[rng.randrange(cfg.vocab_size) for _ in range(n)] for n in (1, 7, 13, 30)]
    prompt_lens = [1, 4, 9, 17]
    seqs = [Sequence(i, f[:p], SamplingParams()) for i, (f, p) in enumerate(zip(full, prompt_lens, strict=True))]
    got: list[list[torch.Tensor]] = [[] for _ in seqs]

    while any(s.num_computed_tokens < len(f) for s, f in zip(seqs, full, strict=True)):
        batch = []
        for s, f in zip(seqs, full, strict=True):
            if s.num_computed_tokens == len(f):
                continue
            n = min(chunk, s.num_tokens - s.num_computed_tokens)
            bm.allocate_slots(s, n)
            batch.append(ScheduledSeq(s, n))
        logits = runner.execute(batch)
        row = 0
        for sched in batch:
            s = sched.seq
            done = sched.completes_sequence
            s.num_computed_tokens += sched.num_new_tokens
            if done:
                got[s.seq_id].append(logits[row])
                row += 1
                if s.num_tokens < len(full[s.seq_id]):  # "sample" the known next token
                    s.append_token(full[s.seq_id][s.num_tokens])
        assert row == logits.shape[0]

    for s, f, p in zip(seqs, full, prompt_lens, strict=True):
        ref = reference_logits(tiny_model, f)[p - 1 :]  # logits at last prompt token onwards
        torch.testing.assert_close(torch.stack(got[s.seq_id]), ref, atol=1e-5, rtol=1e-5)


def test_metadata_mask_slots_and_groups() -> None:
    # seq0: 2 cached + 2 new tokens (prefill chunk); seq1: 4 cached + 1 decode token; block_size 2
    meta = PagedAttentionMetadata.build(
        block_tables=[[5, 3], [0, 1, 7]], query_lens=[2, 1], context_lens=[4, 5], block_size=2, device="cpu"
    )
    assert meta.slot_mapping.tolist() == [3 * 2 + 0, 3 * 2 + 1, 7 * 2 + 0]
    decode, prefill = meta.groups  # decode rows and prefill rows are attended separately
    assert decode.max_q == 1 and decode.token_index.tolist() == [2]
    assert decode.block_ids.tolist() == [0, 1, 7]
    assert decode.mask[0, 0, 0].tolist() == [True] * 5 + [False]  # decode at pos 4 sees 0..4
    assert prefill.max_q == 2 and prefill.token_index.tolist() == [0, 1]
    assert prefill.block_ids.tolist() == [5, 3]
    assert prefill.mask[0, 0].tolist() == [[True, True, True, False], [True, True, True, True]]


def test_padding_rows_never_fully_masked() -> None:
    meta = PagedAttentionMetadata.build(
        block_tables=[[0, 1], [2]], query_lens=[3, 2], context_lens=[3, 2], block_size=2, device="cpu"
    )
    (prefill,) = meta.groups
    assert prefill.mask.any(dim=-1).all()  # a fully masked row would make softmax NaN


def _paged_greedy(model: Qwen3Model, prompts: list[list[int]], max_tokens: int, block_size: int) -> list[list[int]]:
    """All prompts as one static batch through the paged runner (no scheduler)."""
    bm = BlockManager(num_blocks=512, block_size=block_size)
    dtype = model.embed_tokens.weight.dtype
    runner = ModelRunner(model, PagedKVCache(model.cfg, 512, block_size, dtype, "cpu"))
    seqs = [Sequence(i, p, SamplingParams(max_tokens=max_tokens)) for i, p in enumerate(prompts)]
    for _ in range(max_tokens):
        batch = []
        for s in seqs:
            n = s.num_tokens - s.num_computed_tokens
            bm.allocate_slots(s, n)
            batch.append(ScheduledSeq(s, n))
        next_ids = runner.execute(batch).argmax(-1).tolist()
        for s, b, t in zip(seqs, batch, next_ids, strict=True):
            s.num_computed_tokens += b.num_new_tokens
            s.append_token(t)
    return [s.output_token_ids for s in seqs]


@pytest.mark.real_model
@torch.no_grad()
def test_real_model_paged_batch_equals_contiguous_greedy() -> None:
    from ember.loader import load_model, resolve_model_dir
    from ember.simple_generate import generate_greedy
    from ember.tokenizer import Tokenizer

    tok = Tokenizer(resolve_model_dir())
    model = load_model(device="cpu", dtype=torch.float32)
    texts = ["The capital of France is", "def quicksort(arr):", "List three primary colors:", "Hello"]
    prompts = [tok.encode(t) for t in texts]
    paged = _paged_greedy(model, prompts, max_tokens=20, block_size=16)
    for p, out in zip(prompts, paged, strict=True):
        assert out == generate_greedy(model, p, 20)
