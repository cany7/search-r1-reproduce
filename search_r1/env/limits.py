# Derived from tinker-cookbook/rl/rollout_limits.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local imports and a focused text-search API.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.

from typing import Any

import chz

from .types import StopReason


def _positive_or_none(obj: Any, name: str) -> None:
    """chz field validator: the value must be strictly positive or ``None``."""
    value = getattr(obj, name)
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be a positive value or None (unlimited), got {value!r}")


def _non_negative(obj: Any, name: str) -> None:
    """chz field validator: the value must be >= 0."""
    value = getattr(obj, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _contains_details_placeholder(obj: Any, name: str) -> None:
    """chz field validator: the template must contain the ``{details}`` placeholder."""
    value = getattr(obj, name)
    if "{details}" not in value:
        raise ValueError(f"{name} must contain the '{{details}}' placeholder, got {value!r}")


DEFAULT_RETRY_MESSAGE_TEMPLATE = "The previous message had a formatting issue.\nError: {details}"


@chz.chz
class ParseErrorPolicy:
    """How a rollout responds to parse failures in the model's output."""

    max_consecutive: int = chz.field(
        default=0,
        validator=_non_negative,
        doc=(
            "Maximum consecutive content parse failures to retry with an injected "
            "corrective message. 0 (the default) means the first parse failure ends "
            "the episode. Structural failures always end the episode regardless."
        ),
    )
    retry_message_template: str = chz.field(
        default=DEFAULT_RETRY_MESSAGE_TEMPLATE,
        validator=_contains_details_placeholder,
        doc=(
            "Template for the corrective user message injected after a recoverable "
            "parse failure; '{details}' is replaced with the error detail (truncated "
            "to ~16KB). The injected tokens count toward the trajectory token budget."
        ),
    )
    penalty_per_error: float = chz.field(
        default=0.0,
        validator=_non_negative,
        doc=(
            "Penalty subtracted on each retried parse-error turn (that turn's reward "
            "is -penalty_per_error). Not applied on the terminal parse-error turn, "
            "which gets terminal_reward instead."
        ),
    )
    terminal_reward: float = chz.field(
        default=0.0,
        doc=(
            "Flat reward for the turn that ends the episode with "
            "StopReason.PARSE_ERROR (a structural failure, or content failures "
            "exceeding max_consecutive)."
        ),
    )
    mask_error_turns: bool = chz.field(
        default=False,
        doc=(
            "Mark parse-error turns (structural or content) with "
            "metrics['parse_error_masked']=1.0; trajectory_to_data then zeroes the "
            "action-token mask and advantages for those turns, so malformed samples "
            "are excluded from the training loss while the rest of the trajectory "
            "trains normally."
        ),
    )


@chz.chz
class RolloutLimits:
    """Budgets that bound a single rollout (one trajectory)."""

    max_turns: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum number of agent turns (sampling calls) per rollout. "
            "Checked before each turn; reaching the cap produces StopReason.MAX_TURNS."
        ),
    )
    max_trajectory_tokens: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum total tokens in the trajectory, counting both observation "
            "(prompt) and action (sampled) tokens. Feeds the per-turn token budget "
            "as (max_trajectory_tokens - observation length); exhausting it produces "
            "StopReason.MAX_TOKENS. An initial prompt that already exceeds this "
            "budget ends the rollout immediately with StopReason.MAX_TOKENS."
        ),
    )
    max_sampled_tokens: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum cumulative sampled (action-only) tokens per rollout; "
            "observation tokens do not count. Checked after each turn, *after* "
            "LENGTH handling; exhausting it produces StopReason.MAX_SAMPLED_TOKENS "
            "(distinct from MAX_TOKENS, which covers the obs+action budget)."
        ),
    )
    max_tool_calls: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum tool calls per rollout. Checked pre-dispatch (before a turn's "
            "tool batch starts) and mid-batch (between calls within the batch); "
            "exhausting it produces StopReason.MAX_TOOL_CALLS."
        ),
    )
    max_turn_tokens: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum sampled tokens for a single turn. A turn truncated at this "
            "cap (sampler stop_reason 'length') does not by itself end the rollout: "
            "the loop continues unless a cumulative budget (max_trajectory_tokens / "
            "max_sampled_tokens) is also exhausted, in which case the corresponding "
            "stop reason is reported."
        ),
    )
    rollout_timeout_seconds: float | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Wall-clock budget for the whole rollout. Checked between turns and "
            "enforced by an outer timeout around the loop; hitting it ends the "
            "episode gracefully with StopReason.ROLLOUT_TIMEOUT (the trajectory is "
            "kept and remains gradeable, unlike a strategy-level timeout which "
            "cancels and retries the rollout)."
        ),
    )
    sampling_turn_timeout_seconds: float | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Wall-clock budget for a single sampling call. Unlike the other limits "
            "this one raises (the sampling call failed; there is no partial turn to "
            "keep), composing with error-tolerant rollout strategies that retry."
        ),
    )


DEFAULT_LIMIT_STOP_REASONS: tuple[str, ...] = (
    StopReason.MAX_TOKENS,
    StopReason.MAX_SAMPLED_TOKENS,
    StopReason.MAX_TURNS,
    StopReason.MAX_TOOL_CALLS,
    StopReason.ROLLOUT_TIMEOUT,
)


@chz.chz
class TerminationRewardPolicy:
    """How rewards interact with the stop reason a trajectory ended with."""

    zero_reward_on_limit: bool = chz.field(
        default=False,
        doc=(
            "When True, a trajectory whose stop_reason is in limit_stop_reasons "
            "has its total reward clamped to min(reward, 0.0) after grading. "
            "Grading still runs (grade-then-clamp); negative rewards pass through "
            "unchanged. Default False: no clamping."
        ),
    )
    limit_stop_reasons: tuple[str, ...] = chz.field(
        default=DEFAULT_LIMIT_STOP_REASONS,
        doc=(
            "Stop reasons treated as limit stops for zero_reward_on_limit. "
            "Defaults to the five budget-imposed reasons: max_tokens, "
            "max_sampled_tokens, max_turns, max_tool_calls, rollout_timeout."
        ),
    )
    skip_grading_on_timeout: bool = chz.field(
        default=False,
        doc=(
            "Skip the (possibly expensive) grader call and use 0.0 when the "
            "reward would be clamped to zero anyway. This requires a triple "
            "condition: (1) the trajectory stopped with rollout_timeout, "
            "(2) zero_reward_on_limit is True, and (3) rollout_timeout is in "
            "limit_stop_reasons — only then is the graded value guaranteed not "
            "to raise the reward above 0.0, so skipping cannot change the "
            "clamped outcome (beyond discarding a possibly-negative grade). "
            "See do_group_rollout for the exact application scope."
        ),
    )
    grader_timeout_seconds: float | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Wall-clock budget for a grading call, enforced with asyncio.timeout "
            "around (a) compute_group_rewards in do_group_rollout and (b) the "
            "env-level reward_fn call in AgentToolMessageEnv when this policy is "
            "installed on the env. A timeout raises TimeoutError (there is no "
            "usable reward), composing with error-tolerant rollout strategies. "
            "None (default) = unbounded."
        ),
    )
    pass_all_messages_to_grader: bool = chz.field(
        default=False,
        doc=(
            "What the env-level grader (AgentToolMessageEnv.reward_fn) sees when "
            "this policy is installed on the env: False (default) passes only the "
            "completion suffix — the messages generated during the rollout, "
            "excluding the initial prompt messages; True passes the full history. "
            "With no policy installed the default behavior (full history) is "
            "unchanged. Group-level graders always receive full trajectories."
        ),
    )
