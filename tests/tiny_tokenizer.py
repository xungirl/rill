"""A 512-token byte-level tokenizer matching tiny_config(), so API tests need no download.

ids 0-255: the 256 byte-level symbols; 256-509: plain added tokens "<x0>".."<x253>"
(so random-model output decodes to visible text); 510/511: <|im_start|>/<|im_end|>.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
IM_END = 511


def write_tiny_tokenizer(directory: Path) -> Path:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_tokens([f"<x{i}>" for i in range(254)])
    tok.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    assert tok.token_to_id("<|im_end|>") == IM_END
    directory.mkdir(parents=True, exist_ok=True)
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": CHAT_TEMPLATE}))
    return directory
