"""Tokenizer configuration and caching without model downloads."""

from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from search_r1.env.tokenizer import get_tokenizer


@pytest.mark.parametrize("trust", ["", "true"])
def test_huggingface_load_options_and_cache(monkeypatch, trust):
    get_tokenizer.cache_clear()
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", trust)
    loader = Mock()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    try:
        tokenizer = get_tokenizer("Qwen/Qwen3.5-4B:variant")
        assert get_tokenizer("Qwen/Qwen3.5-4B:variant") is tokenizer
        loader.assert_called_once_with(
            "Qwen/Qwen3.5-4B", **({"trust_remote_code": True} if trust else {})
        )
    finally:
        get_tokenizer.cache_clear()


def test_local_tokenizer_directory_keeps_existing_trust_behavior(monkeypatch, tmp_path):
    get_tokenizer.cache_clear()
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    loader = Mock()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    try:
        get_tokenizer(str(tmp_path))
        loader.assert_called_once_with(str(tmp_path), trust_remote_code=True)
    finally:
        get_tokenizer.cache_clear()
