# Adapted from tinker-cookbook/rl/data_processing.py; supervised/common.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import dataclasses
import logging

import tinker
import torch
from tinker import TensorData

from search_r1.env.types import PARSE_ERROR_MASKED_METRIC_KEY, Trajectory, TrajectoryGroup
from search_r1.training.utils import all_same, safezip


def create_rightshifted_model_input_and_leftshifted_targets(chunks):
    if any(not isinstance(c, tinker.EncodedTextChunk) for c in chunks):
        raise ValueError("Search-R1 training supports text tokens only")
    tokens = [token for chunk in chunks for token in chunk.tokens]
    if len(tokens) < 2:
        raise ValueError("need at least 2 tokens for input/target split")
    return tinker.ModelInput.from_ints(tokens[:-1]), tokens[1:]


logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """Check if seq1 is a prefix of seq2."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


ProvenanceSpan = tinker.PromptProvenanceSpan | tinker.SampledProvenanceSpan


def _drop_first_position(spans: list[ProvenanceSpan]) -> list[ProvenanceSpan]:
    """The spans for the same tokens with the first position removed."""
    first = spans[0]
    rest = spans[1:]
    if first.length == 1:
        return rest
    return [dataclasses.replace(first, offset=first.offset + 1, length=first.length - 1), *rest]


def _drop_last_position(spans: list[ProvenanceSpan]) -> list[ProvenanceSpan]:
    """The spans for the same tokens with the last position removed."""
    last = spans[-1]
    rest = spans[:-1]
    if last.length == 1:
        return rest
    return [*rest, dataclasses.replace(last, length=last.length - 1)]


def trajectory_to_data(
    traj: Trajectory, traj_advantage: float, include_provenance: bool = True
) -> list[tinker.Datum]:
    """Return one or more Datum objects corresponding to the trajectory."""

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []
        # Where each token of full_sequence came from, while every action so far
        # has a sequence_id; None once one doesn't (or if not include_provenance).
        spans: list[ProvenanceSpan] | None = None

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []
            cls.spans = [] if include_provenance else None

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        datum = tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )
        spans = SequenceAccumulator.spans
        if not spans:
            return datum
        # The input drops the last token of the sequence and the targets drop the first.
        target_spans = _drop_first_position(spans)
        if not any(isinstance(span, tinker.SampledProvenanceSpan) for span in target_spans):
            return datum
        return datum.with_provenance(
            model_input=_drop_last_position(spans), loss_fn_inputs=target_spans
        )

    SequenceAccumulator.clear()
    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        # A transition with no tokens (e.g. a rollout that ended before its first
        # turn) contributes nothing to the datum, so it needs no provenance.
        sequence_id = ac_with_logprobs.sequence_id
        if sequence_id is None and (delta_ob_len > 0 or ac_with_logprobs.tokens):
            SequenceAccumulator.spans = None
        if SequenceAccumulator.spans is not None and sequence_id is not None:
            # The new observation tokens are the end of this request's prompt.
            prompt_offset = _flat_ob_token_len(SequenceAccumulator.full_sequence)
            if delta_ob_len > 0:
                SequenceAccumulator.spans.append(
                    tinker.PromptProvenanceSpan(
                        sequence_id=sequence_id, offset=prompt_offset, length=delta_ob_len
                    )
                )
            if ac_with_logprobs.tokens:
                SequenceAccumulator.spans.append(
                    tinker.SampledProvenanceSpan(
                        sequence_id=sequence_id, length=len(ac_with_logprobs.tokens)
                    )
                )
        # Parse-error turns flagged for masking (ParseErrorPolicy.mask_error_turns)
        # contribute no training signal: their action tokens get mask=0 and
        # advantage=0, while surrounding turns train normally.
        masked = float(transition.metrics.get(PARSE_ERROR_MASKED_METRIC_KEY, 0.0)) >= 1.0
        action_mask = 0.0 if masked else 1.0
        action_advantage = 0.0 if masked else traj_advantage
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [action_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend(
            [0.0] * delta_ob_len + [action_mask] * len(ac_with_logprobs.tokens)
        )

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
    include_provenance: bool = True,
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(
                traj, float(traj_advantage), include_provenance=include_provenance
            )
            data_D.extend(new_data)
            metadata_D.extend([{"group_idx": i_group, "traj_idx": i_traj} for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Filter out trajectory groups where all trajectories received the same reward."""
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
