"""Tokenizer + chat template, using only the `tokenizers` library and jinja2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self._tok = HFTokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._chat_template = env.from_string(cfg["chat_template"]) if "chat_template" in cfg else None

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict[str, Any]], enable_thinking: bool = False) -> str:
        """Render OpenAI-style messages with the model's own jinja template."""
        if self._chat_template is None:
            raise ValueError("this tokenizer has no chat template")
        return self._chat_template.render(
            messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking
        )


class IncrementalDetokenizer:
    """Turns a growing list of token ids into a stream of text deltas.

    BPE tokens do not map 1:1 onto characters: a multi-byte UTF-8 character can
    be split across tokens, which decodes to U+FFFD until the rest arrives. We
    re-decode all ids (O(n) per token - fine for outputs of a few thousand
    tokens) and hold output back while it ends in U+FFFD, so streamed text
    never contains half a character.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tok = tokenizer
        self.ids: list[int] = []
        self.text = ""  # everything emitted so far

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        full = self.tok.decode(self.ids)
        if full.endswith("�"):
            return ""
        delta = full[len(self.text) :]
        self.text = full
        return delta
