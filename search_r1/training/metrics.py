# Adapted from tinker-cookbook/rl/metrics.py; rl/metric_util.py; rl/train.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import tinker
import torch

from search_r1.env.types import TrajectoryGroup
from search_r1.training.usage import CostTracker, TokenUsage
from search_r1.training.utils import all_same, dict_mean, safezip


def _compute_by_group_metrics(trajectory_groups_P: list[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    if n_groups == 0:
        return {
            "by_group/frac_mixed": 0.0,
            "by_group/frac_all_good": 0.0,
            "by_group/frac_all_bad": 0.0,
        }
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def compute_trajectory_metrics(
    trajectory_groups_P: list[TrajectoryGroup], taglist_P: list[list[str]]
) -> dict[str, float]:
    """Compute per-tag and aggregate trajectory metrics for a batch of rollouts."""
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)
    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P) for trajgroups in tag2trajgroups.values()
    )
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups).items()
            }
            out.update(prefixed_metrics)
    out.update(
        {f"env/all/{k}": v for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()}
    )
    return out


def _compute_trajectory_metrics(trajectory_groups_P: list[TrajectoryGroup]) -> dict[str, float]:
    """Compute metrics for the trajectory groups."""
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs_PG for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        transition.ob.length for traj in flat_trajs_PG for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    total_turns = sum(turns_by_trajectory)
    total_episodes = len(flat_trajs_PG)
    total_ac_tokens = sum(ac_tokens_by_turn)
    total_ob_tokens = sum(ob_tokens_by_turn)
    total_cached_ob_tokens = sum(
        transition.ac.prompt_cache_hit_tokens
        for traj in flat_trajs_PG
        for transition in traj.transitions
    )
    metrics = {
        "ac_tokens_per_turn": total_ac_tokens / total_turns if total_turns > 0 else 0.0,
        "ob_tokens_per_turn": total_ob_tokens / total_turns if total_turns > 0 else 0.0,
        "turns_per_episode": total_turns / total_episodes if total_episodes > 0 else 0.0,
        "total_episodes": total_episodes,
        "total_turns": total_turns,
        "total_ac_tokens": total_ac_tokens,
        "total_ob_tokens": total_ob_tokens,
        "total_cached_ob_tokens": total_cached_ob_tokens,
    }
    all_rewards = [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    metrics["reward/total"] = np.mean(all_rewards).item() if all_rewards else 0.0
    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def compute_kl_sample_train(
    data_D: list[tinker.Datum], training_logprobs_D: list[torch.Tensor]
) -> dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs."""
    all_diffs: list[torch.Tensor] = []
    all_sampling_logprobs: list[torch.Tensor] = []
    for datum, training_logprobs in safezip(data_D, training_logprobs_D):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]
        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)
    assert all_diffs
    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()
    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()
    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
    }


def rollout_usage(trajectory_groups: Sequence[TrajectoryGroup]) -> TokenUsage:
    """The sampling the groups' rollouts were billed for."""
    transitions = [
        transition
        for group in trajectory_groups
        for trajectory in group.trajectories_G
        for transition in trajectory.transitions
    ]
    prompt_tokens = sum(t.ob.length for t in transitions)
    cached_tokens = sum(t.ac.prompt_cache_hit_tokens for t in transitions)
    return TokenUsage(
        sampled=sum(len(t.ac.tokens) for t in transitions),
        prefill_cached=cached_tokens,
        prefill_uncached=prompt_tokens - cached_tokens,
    )


def training_usage(data_D: list[tinker.Datum]) -> TokenUsage:
    """The tokens in the datums sent to ``forward_backward`` (shared prefixes across"""
    return TokenUsage(trained=sum(d.model_input.length for d in data_D))


def count_discarded_rollout(
    cost_tracker: CostTracker | None, trajectory_group: TrajectoryGroup
) -> None:
    """Count the sampling of a completed group that is dropped before training."""
    if cost_tracker is not None:
        cost_tracker.record_discarded_sampling(rollout_usage([trajectory_group]))
