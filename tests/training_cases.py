"""Fixed token trajectories for comparing migrated training math with pinned upstream."""

import tinker

from search_r1.backends.completers import TokensWithLogprobs
from search_r1.env.types import Trajectory, TrajectoryGroup, Transition


def transition(ob, ac, reward=0.0, masked=False, sequence=None):
    return Transition(
        ob=tinker.ModelInput.from_ints(ob),
        ac=TokensWithLogprobs(
            ac,
            [-0.25 * (i + 1) for i in range(len(ac))],
            sequence_id=sequence,
            prompt_cache_hit_tokens=min(1, len(ob)),
        ),
        reward=reward,
        episode_done=True,
        metrics={"parse_error_masked": 1.0} if masked else {"format": 1.0},
    )


def trajectory(*steps):
    return Trajectory(list(steps), tinker.ModelInput.empty())


def group(rewards, trajectories=None):
    if trajectories is None:
        trajectories = [trajectory(transition([1, 2], [3, 4])) for _ in rewards]
    return TrajectoryGroup(trajectories, list(rewards), [{} for _ in rewards])


def cases():
    return {
        "centered_partial_rewards": [group([0.1, 0.12, -0.3]), group([1, 0])],
        "constant": [group([0.5, 0.5])],
        "prefix_and_tool_observations": [
            group(
                [1, 0],
                [
                    trajectory(
                        transition([10, 11], [12, 13], sequence="s1"),
                        transition([10, 11, 12, 13, 14, 15], [16, 17], sequence="s2"),
                    ),
                    trajectory(transition([10, 11], [18, 19], sequence="s3")),
                ],
            )
        ],
        "nonprefix_reset": [
            group(
                [1, 0],
                [
                    trajectory(
                        transition([1, 2], [3], sequence="a"), transition([7, 8], [9], sequence="b")
                    ),
                    trajectory(transition([4, 5], [6], sequence="c")),
                ],
            )
        ],
        "masked_turn": [
            group(
                [1, -1],
                [
                    trajectory(
                        transition([1, 2], [3], masked=True, sequence="x"),
                        transition([1, 2, 3, 4], [5], sequence="y"),
                    ),
                    trajectory(transition([1, 2], [6])),
                ],
            )
        ],
        "partial_provenance": [
            group(
                [0.5, 0],
                [
                    trajectory(
                        transition([1, 2], [3], sequence="a"), transition([1, 2, 3, 4], [5])
                    ),
                    trajectory(transition([1, 2], [6], sequence="b")),
                ],
            )
        ],
        "overflow": [
            group(
                [-0.1, 1],
                [trajectory(transition([], [], reward=-0.1)), trajectory(transition([1, 2], [3]))],
            )
        ],
        "all_overflow": [group([0, 0], [trajectory(transition([], [], reward=-0.1))] * 2)],
        "transition_and_final_rewards": [
            group(
                [0.25, -0.25],
                [
                    trajectory(transition([1, 2], [3], reward=0.5)),
                    trajectory(transition([1, 2], [4], reward=-0.5)),
                ],
            )
        ],
    }
