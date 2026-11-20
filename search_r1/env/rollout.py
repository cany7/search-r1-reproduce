# Derived from tinker-cookbook/rl/rollout_runner.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified: retain the unconfigured single-episode loop used by Search-R1;
# limits and rewards are enforced by the search environment, without generic hooks.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and provenance.
"""A complete search episode: generate, step the environment, repeat until done."""

import tinker

from search_r1.backends.completers import TokenCompleter, TokensWithLogprobs
from search_r1.env.types import (
    STOP_METRIC_PREFIX,
    Env,
    InitialObservationOverflow,
    Trajectory,
    Transition,
)


def stop_reason_from_metrics(metrics) -> str | None:
    for key, value in metrics.items():
        if key.startswith(STOP_METRIC_PREFIX) and value:
            return key[len(STOP_METRIC_PREFIX) :]
    return None


async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    # This runner deliberately has no hidden model presets or runner-side budgets.
    # Refuse unsupported configuration instead of silently ignoring its limits.
    if getattr(env, "rollout_limits", None) is not None:
        raise ValueError(
            "Search-R1 rollout uses environment limits; runner rollout_limits are unsupported"
        )
    initial = await env.initial_observation()
    if isinstance(initial, InitialObservationOverflow):
        empty = tinker.ModelInput.empty()
        transition = Transition(
            ob=empty,
            ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
            reward=initial.reward,
            episode_done=True,
            metrics=initial.metrics,
            logs=initial.logs,
        )
        return Trajectory([transition], empty, stop_reason_from_metrics(initial.metrics))

    observation, stop = initial
    transitions = []
    while True:
        action = await policy(observation, stop)
        result = await env.step(action.tokens, extra={"stop_reason": action.stop_reason})
        transitions.append(
            Transition(
                ob=observation,
                ac=action,
                reward=result.reward,
                episode_done=result.episode_done,
                metrics=result.metrics,
                logs=result.logs,
            )
        )
        observation, stop = result.next_observation, result.next_stop_condition
        if result.episode_done:
            return Trajectory(transitions, observation, stop_reason_from_metrics(result.metrics))
