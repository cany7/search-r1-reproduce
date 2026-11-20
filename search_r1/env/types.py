# Derived from tinker-cookbook/rl/types.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local imports and a focused text-search API.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypedDict

import chz
import tinker

from search_r1.backends.completers import StopCondition, TokensWithLogprobs

Action = list[int]


Observation = tinker.ModelInput


Logprobs = list[float]


Metrics = dict[str, float | int]


Logs = dict[str, str | int | float]


class StopReason(StrEnum):
    """Why a trajectory ended."""

    COMPLETED = "completed"
    """The agent finished on its own (e.g. responded without tool calls)."""
    TOOL_STOPPED = "tool_stopped"
    """A tool signalled that the episode should stop."""
    MAX_TURNS = "max_turns"
    """The turn cap was reached."""
    MAX_TOKENS = "max_tokens"
    """The token budget was exhausted: mid-turn (sampler ``stop_reason='length'``),"""
    MAX_SAMPLED_TOKENS = "max_sampled_tokens"
    """The cumulative sampled-token (action-only) budget was exhausted."""
    MAX_TOOL_CALLS = "max_tool_calls"
    """The trajectory-level tool-call budget was exhausted."""
    CONTEXT_OVERFLOW = "context_overflow"
    """The rendered conversation exceeded the trajectory token budget."""
    PARSE_ERROR = "parse_error"
    """The model output (or its tool calls) failed to parse."""
    ROLLOUT_TIMEOUT = "rollout_timeout"
    """The rollout exceeded its wall-clock time budget."""


STOP_METRIC_PREFIX = "stop/"


PARSE_ERROR_MASKED_METRIC_KEY = "parse_error_masked"


@dataclass(frozen=True)
class InitialObservationOverflow:
    """Sentinel returned by :meth:`Env.initial_observation` instead of an"""

    reward: float
    metrics: Metrics = field(default_factory=dict)
    logs: Logs = field(default_factory=dict)


@dataclass
class StepResult:
    """Result returned by :meth:`Env.step`."""

    reward: float
    """Immediate reward for this step."""
    episode_done: bool
    """Whether the episode has ended."""
    next_observation: Observation
    """Observation for the next step (or final observation if episode_done)."""
    next_stop_condition: StopCondition
    """Stop condition for the next generation."""
    metrics: Metrics = field(default_factory=dict)
    """Numeric values aggregated and reported in training logs (e.g., timing, counts)."""
    logs: Logs = field(default_factory=dict)
    """Diagnostic info for display/debugging tools (not aggregated like metrics)."""


@dataclass
class Transition:
    """A single (observation, action, reward) tuple from a trajectory."""

    ob: Observation
    """Observation the agent saw before taking the action."""
    ac: TokensWithLogprobs
    """Action taken (tokens and their log-probabilities)."""
    reward: float
    """Immediate reward received after taking the action."""
    episode_done: bool
    """Whether this transition ended the episode."""
    metrics: Metrics = field(default_factory=dict)
    """Numeric values aggregated and reported in training logs."""
    logs: Logs = field(default_factory=dict)
    """Diagnostic info for display/debugging tools (not aggregated like metrics)."""


class ActionExtra(TypedDict, total=False):
    """Extra metadata passed alongside an action to :meth:`Env.step`."""

    stop_reason: tinker.StopReason
    """Why sampling stopped — ``"stop"`` (hit a stop sequence) or ``"length"``"""


class Env(ABC):
    """Stateful environment that a single agent interacts with."""

    @abstractmethod
    async def initial_observation(
        self,
    ) -> tuple[Observation, StopCondition] | InitialObservationOverflow:
        """Return the starting observation and stop condition for this episode."""
        pass

    @abstractmethod
    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Advance the environment by one step given the agent's action."""
        pass


@dataclass(frozen=True)
class Trajectory:
    """A complete episode: a sequence of transitions from one agent in one environment."""

    transitions: list[Transition]
    final_ob: Observation
    stop_reason: str | None = None


@dataclass(frozen=True)
class RolloutError:
    """A captured error from a failed trajectory rollout."""

    error_type: str
    """The exception class name, e.g. ``'BadRequestError'``."""
    error_message: str
    """``str(exception)`` — the human-readable error description."""


class EnvGroupBuilder(ABC):
    """Builds a group of environments. The group will be used in the following way:"""

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        """Create the environments for this group."""
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute a final reward for each trajectory that depends on the whole group."""
        return [(0.0, {}) for _ in trajectory_group]

    async def cleanup(self) -> None:
        """Clean up resources created by make_envs()."""
        pass

    def logging_tags(self) -> list[str]:
        """Return tags used to aggregate metrics in training logs."""
        return []


@dataclass
class TrajectoryGroup:
    """A group of trajectories produced by one :class:`EnvGroupBuilder`."""

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

    # Error tracking — populated by do_group_rollout when using error-tolerant strategies.
    # Empty list means no trajectory errors occurred.
    rollout_errors: list[RolloutError] = field(default_factory=list)

    def get_total_rewards(self) -> list[float]:
        """Get the total reward (return) for each trajectory in the group."""
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in zip(self.trajectories_G, self.final_rewards_G, strict=True)
        ]


class RLDataset(ABC):
    """A dataset that produces batches of :class:`EnvGroupBuilder` instances."""

    @abstractmethod
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return a batch of EnvGroupBuilder instances at the given index."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        pass


@chz.chz
class RLDatasetBuilder:
    """Abstract builder that constructs training and optional test RL datasets."""

    @abstractmethod
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """Build the training dataset and an optional test dataset."""
        pass
