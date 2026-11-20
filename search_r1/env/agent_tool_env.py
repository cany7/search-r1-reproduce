# Derived from tinker-cookbook/tool_use/agent_tool_message_env.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local imports and a focused text-search API.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from search_r1.env import types
from search_r1.env.limits import ParseErrorPolicy, TerminationRewardPolicy
from search_r1.env.message_env import MessageEnv, MessageStepResult
from search_r1.env.renderers.base import (
    PARSE_FAILURE_DETAIL_MAX_CHARS,
    Message,
    ToolCall,
    UnparsedToolCall,
    format_content_as_string,
    get_text_content,
)
from search_r1.env.tool_dispatch import handle_tool_call
from search_r1.env.tool_types import Tool, ToolResult

ToolExecution = Literal["parallel", "sequential", "concurrent_safe"]

RewardResult = tuple[float, dict[str, float]]


RewardFn = Callable[[list[Message]], Awaitable[RewardResult]]


@dataclass
class AgentToolMessageEnv(MessageEnv):
    """Generic tool-use MessageEnv for agents."""

    tools: list[Tool]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: RewardFn
    failed_parse_reward: float = -0.1
    terminate_on_parse_error: bool = True
    max_tool_calls: int | None = None
    parse_error_policy: ParseErrorPolicy | None = None
    tool_execution: ToolExecution = "parallel"
    termination_policy: TerminationRewardPolicy | None = None
    history: list[Message] = field(default_factory=list)

    _turn_count: int = 0
    _consecutive_parse_errors: int = field(default=0, init=False)
    _tool_calls_made: int = field(default=0, init=False)
    _tool_dict: dict[str, Tool] = field(default_factory=dict, init=False)
    _should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}
        if self.tool_execution == "concurrent_safe":
            # The public Tool protocol has no concurrency-safety marker yet,
            # so there is nothing to key the safe/unsafe split off. Reserved
            # until the tool contract grows one; use "sequential" (always
            # safe) or "parallel" (the default) instead.
            raise NotImplementedError(
                'tool_execution="concurrent_safe" is reserved: the Tool protocol has '
                "no concurrency-safety marker yet. Use 'sequential' or 'parallel'."
            )

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    def set_max_tool_calls(self, max_tool_calls: int) -> None:
        """Apply a rollout-level tool-call limit (seam used by the rollout runner)."""
        if self.max_tool_calls is None:
            self.max_tool_calls = max_tool_calls
        else:
            self.max_tool_calls = min(self.max_tool_calls, max_tool_calls)

    def set_parse_error_policy(self, policy: ParseErrorPolicy) -> None:
        """Apply a rollout-level parse-error policy (seam used by the rollout runner)."""
        self.parse_error_policy = policy

    async def observe_truncated_response(self, message: Message) -> list[Message]:
        """Record a truncated assistant response (LENGTH-continue support)."""
        self.history.append(message)
        return self.history

    async def add_messages(self, messages: list[Message]) -> list[Message]:
        """Append externally injected messages (rollout-hook support)."""
        self.history.extend(messages)
        return self.history

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Execute tool calls and append results to history."""
        tool_results: list[ToolResult]
        if self.tool_execution == "sequential":
            tool_results = [await handle_tool_call(self._tool_dict, tc) for tc in tool_calls]
        else:
            tool_results = list(
                await asyncio.gather(*[handle_tool_call(self._tool_dict, tc) for tc in tool_calls])
            )

        all_messages: list[Message] = []

        for tool_result in tool_results:
            # Append messages to history
            for msg in tool_result.messages:
                self.history.append(msg)
                all_messages.append(msg)

            # Check if any tool signals to stop
            if tool_result.should_stop:
                self._should_stop = True

        return all_messages

    async def step(self, message: Message) -> MessageStepResult:
        """Execute any tools and return next messages."""
        self._turn_count += 1
        metrics: dict[str, float] = {}
        logs: types.Logs = {}

        # Append the message to history
        self.history.append(message)

        # Log assistant content (handles both str and multimodal content)
        assistant_text = get_text_content(message)
        if assistant_text:
            logs["assistant_content"] = assistant_text

        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        unparsed_tool_calls: list[UnparsedToolCall] = list(message.get("unparsed_tool_calls") or [])
        max_turns_reached = self._turn_count >= self.max_turns

        if unparsed_tool_calls:
            metrics["parse_error"] = 1.0
            logs["parse_errors"] = "\n".join(tc.error for tc in unparsed_tool_calls)

        if unparsed_tool_calls and not tool_calls:
            # The model tried to call tools but every call was malformed —
            # a content-level parse failure (the framing itself was clean).
            self._consecutive_parse_errors += 1
            if self.parse_error_policy is not None:
                return self._content_parse_error_step(
                    unparsed_tool_calls, metrics, logs, max_turns_reached
                )
            # Default one-shot semantics (no policy): failed_parse_reward,
            # reward_fn skipped, terminate_on_parse_error respected.
            done = self.terminate_on_parse_error or max_turns_reached
            if max_turns_reached:
                metrics["max_turns"] = 1.0
            if done:
                metrics[f"{types.STOP_METRIC_PREFIX}{types.StopReason.PARSE_ERROR}"] = 1.0
            return MessageStepResult(
                reward=self.failed_parse_reward,
                episode_done=done,
                next_messages=self.history,
                metrics=metrics,
                logs=logs,
            )

        # This turn was not a parse failure: the consecutive-error budget
        # (ParseErrorPolicy.max_consecutive) applies to *consecutive* errors.
        self._consecutive_parse_errors = 0

        # Enforce the tool-call budget when configured (default None = no cap):
        # pre-dispatch (a batch arriving with no budget left dispatches nothing)
        # and mid-batch (a batch larger than the remaining budget is cut at the
        # boundary; only the allowed prefix executes). Either way the episode
        # ends with StopReason.MAX_TOOL_CALLS.
        tool_calls_capped = False
        tool_calls_to_run = tool_calls
        if tool_calls and self.max_tool_calls is not None:
            remaining_budget = max(self.max_tool_calls - self._tool_calls_made, 0)
            if len(tool_calls) > remaining_budget:
                tool_calls_capped = True
                tool_calls_to_run = tool_calls[:remaining_budget]
                logs["tool_calls_dropped"] = (
                    f"{len(tool_calls) - remaining_budget} tool call(s) dropped: "
                    f"max_tool_calls={self.max_tool_calls} reached"
                )

        # Execute valid tool calls if present
        if tool_calls_to_run:
            for i, tc in enumerate(tool_calls_to_run):
                logs[f"tool_call_{i}"] = f"{tc.function.name}({tc.function.arguments})"

            tool_result_messages = await self._handle_tool_calls(tool_calls_to_run)
            self._tool_calls_made += len(tool_calls_to_run)

            for i, msg in enumerate(tool_result_messages):
                logs[f"tool_result_{i}"] = format_content_as_string(msg["content"])

        # Determine if episode is done
        no_tool_calls = len(tool_calls) == 0
        done = no_tool_calls or max_turns_reached or self._should_stop or tool_calls_capped

        if max_turns_reached and not no_tool_calls:
            metrics["max_turns"] = 1.0
        if self._should_stop:
            metrics["tool_stopped"] = 1.0
        if tool_calls_capped:
            metrics["max_tool_calls"] = 1.0

        if done:
            if no_tool_calls:
                stop_reason = types.StopReason.COMPLETED
            elif self._should_stop:
                stop_reason = types.StopReason.TOOL_STOPPED
            elif tool_calls_capped:
                stop_reason = types.StopReason.MAX_TOOL_CALLS
            else:
                stop_reason = types.StopReason.MAX_TURNS
            metrics[f"{types.STOP_METRIC_PREFIX}{stop_reason}"] = 1.0

        reward = 0.0
        if done:
            reward, reward_metrics = await self._grade()
            metrics.update(reward_metrics)

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=metrics,
            logs=logs,
        )

    async def _grade(self) -> RewardResult:
        """Run ``reward_fn`` under the termination policy's grader knobs."""
        policy = self.termination_policy
        messages = self.history
        if policy is not None and not policy.pass_all_messages_to_grader:
            messages = self.history[len(self.initial_messages) :]
        timeout = policy.grader_timeout_seconds if policy is not None else None
        async with asyncio.timeout(timeout):
            return await self.reward_fn(messages)

    def _content_parse_error_step(
        self,
        unparsed_tool_calls: list[UnparsedToolCall],
        metrics: dict[str, float],
        logs: types.Logs,
        max_turns_reached: bool,
    ) -> MessageStepResult:
        """Handle a content parse failure under a :class:`ParseErrorPolicy`."""
        policy = self.parse_error_policy
        assert policy is not None
        logs["parse_failure_kind"] = "content"
        if policy.mask_error_turns:
            metrics[types.PARSE_ERROR_MASKED_METRIC_KEY] = 1.0

        if self._consecutive_parse_errors <= policy.max_consecutive and not max_turns_reached:
            details = "\n".join(tc.error for tc in unparsed_tool_calls)
            details = details[:PARSE_FAILURE_DETAIL_MAX_CHARS]
            retry_message: Message = {
                "role": "user",
                "content": policy.retry_message_template.format(details=details),
            }
            self.history.append(retry_message)
            logs["parse_error_retry"] = (
                f"injected corrective message (consecutive parse error "
                f"{self._consecutive_parse_errors}, budget {policy.max_consecutive})"
            )
            return MessageStepResult(
                reward=-policy.penalty_per_error,
                episode_done=False,
                next_messages=self.history,
                metrics=metrics,
                logs=logs,
            )

        if max_turns_reached:
            metrics["max_turns"] = 1.0
        metrics[f"{types.STOP_METRIC_PREFIX}{types.StopReason.PARSE_ERROR}"] = 1.0
        return MessageStepResult(
            reward=policy.terminal_reward,
            episode_done=True,
            next_messages=self.history,
            metrics=metrics,
            logs=logs,
        )
