# Derived from tinker-cookbook/tokenizer_utils.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified: retain the Hugging Face text-tokenizer path used by Search-R1 only.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and provenance.
"""Cached tokenizer loading for the project's Qwen text experiments."""

import os
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    Tokenizer = PreTrainedTokenizerBase
else:
    Tokenizer = Any


class SupportsEncodeOrdinary(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def encode_without_special_tokens(tokenizer: SupportsEncodeOrdinary, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    from transformers import AutoTokenizer

    model_name = model_name.split(":")[0]
    kwargs = {}
    if os.path.isdir(model_name) or os.environ.get("HF_TRUST_REMOTE_CODE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        kwargs["trust_remote_code"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)
