"""Per-turn search limits for the second training phase."""

from __future__ import annotations

import json
from typing import Annotated

from tinker_cookbook.renderers.base import Message, Renderer, ToolCall, ToolSpec
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
from tinker_cookbook.tool_use import AgentToolMessageEnv, ToolResult, tool

from search_r1.tools import BM25Tool, TextAnswerReward, requested_query_list


class LimitedSearchMessageEnv(AgentToolMessageEnv):
    """One search per assistant message, with a fresh budget on the next turn."""

    def __init__(
        self,
        retriever: BM25Tool,
        initial_messages: list[Message],
        reward_fn: TextAnswerReward,
        max_turns: int,
        example_id: str,
    ):
        self.retriever = retriever
        self.example_id = example_id
        self._executed_queries: list[str] | None = None
        super().__init__([self.search], initial_messages, max_turns, reward_fn)

    @tool
    async def search(
        self, query_list: Annotated[list[str], "A list of fully-formed semantic queries."]
    ) -> ToolResult:
        """Search Wikipedia for relevant information based on the given query."""
        # The cookbook validates the entire argument before invoking this method.
        self._executed_queries = query_list[:3]
        return await self.retriever.retrieve(self._executed_queries)

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
                # The queries reached the retriever. Its tool result records any I/O error.
                status = "dispatched"
            records.append(
                {
                    "requested_queries": requested,
                    "executed_queries": self._executed_queries if status == "dispatched" else [],
                    "status": status,
                }
            )
        # Cookbook transition logs accept scalar values and are saved in rollout JSONL.
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
        return result


def search_tool_spec(limit_search: bool) -> ToolSpec:
    search = LimitedSearchMessageEnv.search if limit_search else BM25Tool.search
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
) -> EnvFromMessageEnv:
    if limit_search:
        message_env = LimitedSearchMessageEnv(
            search_tool, initial_messages, reward_fn, max_turns, example_id
        )
    else:
        message_env = AgentToolMessageEnv(
            tools=[search_tool.search],
            initial_messages=initial_messages,
            reward_fn=reward_fn,
            max_turns=max_turns,
        )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=message_env.failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=context_overflow_reward,
    )
