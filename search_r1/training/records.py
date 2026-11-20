# Adapted from tinker-cookbook/rl/rollout_logging.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
import logging
from collections.abc import Sequence
from typing import Any

from search_r1.env.types import TrajectoryGroup
from search_r1.training.utils import safezip

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable form."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            logger.debug("Failed to convert %r via .item(), falling back to str()", type(value))
    return str(value)


def serialize_rollout_summaries(
    *,
    split: str,
    iteration: int,
    trajectory_groups_P: Sequence[TrajectoryGroup],
    taglist_P: Sequence[list[str]],
    sampling_client_steps_P: Sequence[int | None] | None = None,
) -> list[dict[str, Any]]:
    """Serialize trajectory groups into JSON-safe rollout summary records."""
    records: list[dict[str, Any]] = []
    for group_idx, (trajectory_group, tags) in enumerate(safezip(trajectory_groups_P, taglist_P)):
        total_rewards_G = trajectory_group.get_total_rewards()
        sampling_step = (
            sampling_client_steps_P[group_idx] if sampling_client_steps_P is not None else None
        )
        for traj_idx, trajectory in enumerate(trajectory_group.trajectories_G):
            steps = []
            for step_idx, transition in enumerate(trajectory.transitions):
                steps.append(
                    {
                        "step_idx": step_idx,
                        "ob_len": transition.ob.length,
                        "ac_len": len(transition.ac.tokens),
                        "reward": transition.reward,
                        "episode_done": transition.episode_done,
                        "metrics": transition.metrics,
                        "logs": transition.logs,
                    }
                )
            records.append(
                _json_safe(
                    {
                        "schema_version": 1,
                        "split": split,
                        "iteration": iteration,
                        "group_idx": group_idx,
                        "traj_idx": traj_idx,
                        "tags": list(tags),
                        "sampling_client_step": sampling_step,
                        "total_reward": total_rewards_G[traj_idx],
                        "final_reward": trajectory_group.final_rewards_G[traj_idx],
                        "trajectory_metrics": trajectory_group.metrics_G[traj_idx],
                        "steps": steps,
                        "final_ob_len": trajectory.final_ob.length,
                    }
                )
            )
    return records
