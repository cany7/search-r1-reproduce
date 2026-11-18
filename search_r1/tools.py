from __future__ import annotations

import asyncio
import json
import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from typing import Annotated, Literal

import aiohttp
import chz
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, ToolCall
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
        return await self.retrieve(query_list)

    async def retrieve(self, query_list: list[str]) -> ToolResult:
        """Retrieve these queries in order and format their passages for the model."""
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


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between two answers after ``normalize_answer`` (SQuAD-style)."""
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    num_same = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def requested_query_list(tool_call: ToolCall) -> object:
    """Keep the original query_list value, including invalid stringified lists."""
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return None
    return arguments.get("query_list") if isinstance(arguments, dict) else None


def extract_search_queries(messages: list[Message]) -> tuple[int, list[str]]:
    """Return (number of search calls the model made, all their queries in order)."""
    num_calls = 0
    queries: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if tool_call.function.name != "search":
                continue
            num_calls += 1
            query_list = requested_query_list(tool_call)
            if isinstance(query_list, list):
                queries.extend(str(q) for q in query_list)
    return num_calls, queries


def count_tool_errors(messages: list[Message]) -> int:
    """Number of tool results that are an error rather than search results.

    A call whose arguments fail validation (e.g. ``query_list`` given as a string)
    is answered with ``{"error": ...}`` and no search runs.
    """
    return sum(
        1
        for message in messages
        if message.get("role") == "tool" and str(message.get("content", "")).startswith('{"error"')
    )


@dataclass
class TextAnswerReward:
    """Reward function to check text answer against gold answers.

    formula: answer_score + format_coef * (correct_format - 1) + no_search_coef * (searched - 1)

    - ``answer_score="f1"`` with ``no_search_coef=0.1`` is the F1+ reward
      (Search-R1++): token-level F1 against the best-matching gold answer, minus
      0.1 when the trajectory never called the search tool, minus 0.1 when there is
      no well-formed answer.
    - ``answer_score="em"`` with ``no_search_coef=0.0`` is the recipe's original
      reward (exact match plus the format term).

    The metrics always report both exact match (``correct``) and ``f1``.
    """

    gold_answers: list[str]
    format_coef: float = 0.1
    answer_score: Literal["f1", "em"] = "f1"
    no_search_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the completed episode by checking the final assistant message."""
        # Find the last assistant message
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            return 0.0, {"format": 0.0, "correct": 0.0, "f1": 0.0, "searched": 0.0}

        # Use get_text_content to properly handle thinking models (o1, o3)
        content = get_text_content(final_message)

        correct_format = float(self._extract_answer(content) is not None)
        correct_answer = float(self._check_answer(content))
        f1 = self._f1(content)
        num_search_calls, _ = extract_search_queries(history)
        searched = float(num_search_calls > 0)

        score = f1 if self.answer_score == "f1" else correct_answer
        reward = (
            score + self.format_coef * (correct_format - 1) + self.no_search_coef * (searched - 1)
        )
        return reward, {
            "format": correct_format,
            "correct": correct_answer,
            "f1": f1,
            "searched": searched,
        }

    def _extract_answer(self, text: str) -> str | None:
        if "Answer:" not in text:
            return None
        parts = text.split("Answer:")
        if len(parts) != 2:
            return None
        return parts[1].strip() or None

    def _check_answer(self, text: str) -> bool:
        model_answer = self._extract_answer(text)
        if model_answer is None:
            return False
        normalized = normalize_answer(model_answer)
        return bool(normalized) and any(
            normalized == normalize_answer(gold) for gold in self.gold_answers
        )

    def _f1(self, text: str) -> float:
        model_answer = self._extract_answer(text)
        if model_answer is None or len(self.gold_answers) == 0:
            return 0.0
        return max(compute_f1(model_answer, gold) for gold in self.gold_answers)
