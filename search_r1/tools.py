from __future__ import annotations

import asyncio
import logging
import re
import string
from dataclasses import dataclass
from functools import reduce
from typing import Annotated

import aiohttp
import chz
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool


def normalize_answer(s: str) -> str:
    """Normalize answer by lowercasing, removing punctuation, articles, and fixing whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    # Apply transformations in order using reduce
    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, func: func(text), transformations, s)


logger = logging.getLogger(__name__)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)


@chz.chz
class RetrievalConfig:
    n_results: int = 3


class BM25Tool:
    """Search tool backed by Search-R1's BM25 retrieval server (``retrieval/retrieval_server.py``).

    Replaces the recipe's ``ChromaTool``. The text returned to the model keeps the
    recipe's format (``Query: ...`` followed by ``Document i:`` blocks); only the
    retrieval backend differs.

    Pickle support: the aiohttp session is not pickleable, so ``__getstate__``
    drops it and it is re-created lazily on first use.
    """

    def __init__(
        self,
        retrieval_url: str,
        retrieval_config: RetrievalConfig = RetrievalConfig(),
        max_retries: int = 10,
        initial_retry_delay: int = 1,
    ):
        self._retrieval_url = retrieval_url.rstrip("/")
        self._retrieval_config = retrieval_config
        self._max_retries = max_retries
        self._initial_retry_delay = initial_retry_delay
        self._session: aiohttp.ClientSession | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_session"] = None
        return state

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _retrieve_with_retry(self, query_list: list[str]) -> list[list[str]]:
        """Return, for each query, the ``contents`` of its top passages.

        ``return_scores`` is always sent as True: with False the upstream server
        unpacks a single return value into two and fails (or, for exactly two
        queries, silently returns the wrong shape).
        """
        payload = {
            "queries": query_list,
            "topk": self._retrieval_config.n_results,
            "return_scores": True,
        }
        for attempt in range(self._max_retries):
            try:
                async with self._get_session().post(
                    f"{self._retrieval_url}/retrieve", json=payload
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                return [[hit["document"]["contents"] for hit in hits] for hits in data["result"]]
            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait_time = self._initial_retry_delay * (1.5**attempt)
                    logger.error(
                        f"Retrieval attempt {attempt + 1}/{self._max_retries} "
                        f"failed: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise e

        raise RuntimeError("All retrieval attempts failed")

    @tool
    async def search(
        self,
        query_list: Annotated[
            list[str],
            "A list of fully-formed semantic queries. The tool will return search results for each query.",
        ],
    ) -> ToolResult:
        """Search Wikipedia for relevant information based on the given query."""
        async with _CONNECTION_SEMAPHORE:
            documents_list = await self._retrieve_with_retry(query_list)

        # Same format as the recipe's ChromaTool.
        message_content = ""
        for query, documents in zip(query_list, documents_list):
            message_content += f"Query: {query}\n"
            for doc_i, doc in enumerate(documents):
                message_content += f"Document {doc_i + 1}:\n"
                message_content += f"{doc}\n"

        return simple_tool_result(message_content)


@dataclass
class TextAnswerReward:
    """Reward function to check text answer against gold answers.

    formula: format_coef * (correct_format - 1) + correct_answer
    """

    gold_answers: list[str]
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the completed episode by checking the final assistant message."""
        # Find the last assistant message
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            return 0.0, {"format": 0.0, "correct": 0.0}

        # Use get_text_content to properly handle thinking models (o1, o3)
        content = get_text_content(final_message)

        correct_format = float(self._extract_answer(content) is not None)
        correct_answer = float(self._check_answer(content))

        reward = self.format_coef * (correct_format - 1) + correct_answer
        return reward, {"format": correct_format, "correct": correct_answer}

    def _extract_answer(self, text: str) -> str | None:
        if "Answer:" not in text:
            return None
        parts = text.split("Answer:")
        if len(parts) != 2:
            return None
        return parts[1].strip()

    def _check_answer(self, text: str) -> bool:
        model_answer = self._extract_answer(text)
        if model_answer is None or len(self.gold_answers) == 0:
            return False
        for gold in self.gold_answers:
            if normalize_answer(model_answer) == normalize_answer(gold):
                return True
        return False
