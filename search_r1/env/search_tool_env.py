"""Search environments: unlimited search and per-turn limits for the second training phase."""

from __future__ import annotations

import json
from typing import Annotated

from search_r1.env import types
from search_r1.env.agent_tool_env import AgentToolMessageEnv
from search_r1.env.message_env import EnvFromMessageEnv, MessageStepResult
from search_r1.env.renderers.base import Message, Renderer, ToolCall, ToolSpec
from search_r1.env.tool_dispatch import tool
from search_r1.env.tool_types import ToolResult
from search_r1.env.tools import (
    BM25Tool,
    RetrievalUnavailable,
    TextAnswerReward,
    requested_query_list,
)

# Transition log key set on the terminal step whose reward was replaced. Its value is the
# reward the shared scorer gave, so rollouts keep both the original and the training reward.
MAX_TURNS_UNANSWERED_LOG = "max_turns_unanswered_original_reward"
MAX_TURNS_STOP = f"{types.STOP_METRIC_PREFIX}{types.StopReason.MAX_TURNS}"


def is_max_turns_unanswered(result: MessageStepResult) -> bool:
    """The episode ended at the turn limit and its final message has no valid answer.

    A fifth turn that answers without a tool call stops as completed, and a valid answer
    next to a tool call keeps format == 1. Parse failures stop as parse_error; generation
    truncation and context overflow end in EnvFromMessageEnv without reaching here.
    """
    return (
        result.episode_done
        and result.metrics.get(MAX_TURNS_STOP) == 1.0
        and result.metrics.get("format") == 0.0
    )


class SearchMessageEnv(AgentToolMessageEnv):
    """Every search call runs all of its queries.

    The tool dispatcher answers any tool exception with an error message to the model.
    A retrieval server that is still down after its retries is not a model error, so
    ``step`` raises it and the rollout fails instead of continuing on that message.
    """

    def __init__(
        self,
        retriever: BM25Tool,
        initial_messages: list[Message],
        reward_fn: TextAnswerReward,
        max_turns: int,
    ):
        self.retriever = retriever
        self._retrieval_failure: RetrievalUnavailable | None = None
        super().__init__([self.search], initial_messages, max_turns, reward_fn)

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
        try:
            return await self.retriever.retrieve(query_list)
        except RetrievalUnavailable as failure:
            self._retrieval_failure = failure
            raise

    async def step(self, message: Message) -> MessageStepResult:
        result = await super().step(message)
        if self._retrieval_failure is not None:
            raise self._retrieval_failure
        return result


class LimitedSearchMessageEnv(SearchMessageEnv):
    """One search per assistant message, with a fresh budget on the next turn."""

    def __init__(
        self,
        retriever: BM25Tool,
        initial_messages: list[Message],
        reward_fn: TextAnswerReward,
        max_turns: int,
        example_id: str,
        max_turns_unanswered_reward: float | None = None,
    ):
        self.example_id = example_id
        # Training only: terminal reward for episodes that hit max_turns without an answer.
        self.max_turns_unanswered_reward = max_turns_unanswered_reward
        self._executed_queries: list[str] | None = None
        super().__init__(retriever, initial_messages, reward_fn, max_turns)

    @tool
    async def search(
        self, query_list: Annotated[list[str], "A list of fully-formed semantic queries."]
    ) -> ToolResult:
        """Search Wikipedia for relevant information based on the given query."""
        # The tool dispatcher validates the entire argument before invoking this method.
        self._executed_queries = query_list[:3]
        return await self.retrieve(self._executed_queries)

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        # Keep the sampled action intact in history. Filter only the dispatch list.
        selected = []
        seen_search = False
        for call in tool_calls:
            if call.function.name == "search":
                if seen_search:
                    continue
                seen_search = True
            selected.append(call)
        return await super()._handle_tool_calls(selected)

    async def step(self, message: Message) -> MessageStepResult:
        self._executed_queries = None
        result = await super().step(message)
        calls = [c for c in message.get("tool_calls", []) if c.function.name == "search"]
        records = []
        requested_count = 0
        for index, call in enumerate(calls):
            requested = requested_query_list(call)
            if isinstance(requested, list):
                requested_count += len(requested)
            if index > 0:
                status = "skipped"
            elif self._executed_queries is None:
                status = "invalid_arguments"
            else:
                # The retriever returned results: an outage has already raised in step().
                status = "dispatched"
            records.append(
                {
                    "requested_queries": requested,
                    "executed_queries": self._executed_queries if status == "dispatched" else [],
                    "status": status,
                }
            )
        # Transition logs accept scalar values and are saved in rollout JSONL.
        result.logs["search_calls"] = json.dumps(records, ensure_ascii=False)
        result.metrics.update(
            search_calls_requested=len(calls),
            search_calls_executed=int(self._executed_queries is not None),
            search_queries_requested=requested_count,
            search_queries_executed=len(self._executed_queries or []),
        )
        if self._turn_count == 1:
            result.logs["example_id"] = self.example_id
            result.metrics["first_turn_queries_requested"] = requested_count
        if self.max_turns_unanswered_reward is not None and is_max_turns_unanswered(result):
            # An absolute terminal value, settled once, in place of the scorer's reward.
            result.logs[MAX_TURNS_UNANSWERED_LOG] = result.reward
            result.reward = self.max_turns_unanswered_reward
        return result


def search_tool_spec(limit_search: bool) -> ToolSpec:
    search = LimitedSearchMessageEnv.search if limit_search else SearchMessageEnv.search
    return search.to_spec()


def build_search_env(
    *,
    renderer: Renderer,
    search_tool: BM25Tool,
    initial_messages: list[Message],
    reward_fn: TextAnswerReward,
    max_turns: int,
    max_trajectory_tokens: int,
    max_generation_tokens: int | None,
    context_overflow_reward: float,
    limit_search: bool,
    example_id: str,
    max_turns_unanswered_reward: float | None = None,
) -> EnvFromMessageEnv:
    if limit_search:
        message_env = LimitedSearchMessageEnv(
            search_tool,
            initial_messages,
            reward_fn,
            max_turns,
            example_id,
            max_turns_unanswered_reward=max_turns_unanswered_reward,
        )
    else:
        if max_turns_unanswered_reward is not None:
            raise ValueError("max_turns_unanswered_reward requires limit_search=True")
        message_env = SearchMessageEnv(search_tool, initial_messages, reward_fn, max_turns)
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=message_env.failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=context_overflow_reward,
    )
