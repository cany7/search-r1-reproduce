"""Inspect dynamic checkpoints, disk requirements and rollout evidence on the Pod."""

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.cloud.checkpoints.ckpt_prune import state_file
from search_r1.checkpoint_index import full_records, latest, state_at

GIB = 1024**3


def records(path):
    if path.is_file():
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def snapshot(run):
    path = run / "config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


@dataclass(frozen=True)
class Origin:
    step: int = 0
    offset: int = 0
    uri: str | None = None


def origin(run, parent=None, step=None, *, fresh=False):
    """Use the pinned source even after the original parent has been archived."""
    cfg = {} if fresh else snapshot(run)
    if (parent is None) != (step is None):
        raise ValueError("Specify PARENT_RUN and PARENT_STEP together")
    if parent is not None:
        parent = Path(parent).resolve()
        if (
            parent == run.resolve()
            or parent in run.resolve().parents
            or run.resolve() in parent.parents
        ):
            raise ValueError("Expected separate parent and child directories")
        if step < 0 or (cfg and not cfg.get("parent_state_path")):
            raise ValueError("Cannot change an existing run's origin")
        if cfg.get("parent_run") and (Path(cfg["parent_run"]).resolve(), cfg["parent_step"]) != (
            parent,
            step,
        ):
            raise ValueError("Requested parent differs from the pinned origin")
    if pinned := cfg.get("parent_checkpoint"):
        if (pinned["batch"], pinned["state_path"]) != (
            cfg["parent_step"],
            cfg["parent_state_path"],
        ):
            raise ValueError("Inconsistent pinned parent checkpoint")
        return Origin(pinned["batch"], pinned["next_prompt_offset"], pinned["state_path"])
    if parent is None:
        if cfg.get("parent_state_path"):
            raise ValueError("Supply PARENT_RUN and PARENT_STEP for this fork")
        return Origin()
    row = state_at(parent, step)
    if cfg.get("parent_state_path", row["state_path"]) != row["state_path"]:
        raise ValueError("Parent state differs from this fork's recorded parent")
    parent_cfg = snapshot(parent)
    if (
        parent_cfg.get(
            "sampling_mode", "dynamic" if parent_cfg.get("parent_state_path") else "ordinary"
        )
        != "ordinary"
    ):
        raise ValueError("Forking requires an ordinary parent")
    offset = row.get("next_prompt_offset", step * parent_cfg["dataset_builder"]["batch_size"])
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid parent cursor")
    return Origin(step, offset, row["state_path"])


def own_records(output, start):
    cfg = snapshot(output)
    batch = cfg.get("dataset_builder", {}).get("batch_size", 1)
    dynamic = (
        cfg.get("sampling_mode", "dynamic" if cfg.get("parent_state_path") else "ordinary")
        == "dynamic"
    )
    rows = full_records(output)
    for row in rows:
        if row.get("parent_step", start.step) != start.step or (
            cfg.get("parent_run") and row.get("parent_run", cfg["parent_run"]) != cfg["parent_run"]
        ):
            raise ValueError(f"Foreign checkpoint origin in {output}")
        if row.get("owner") == "parent":
            if not dynamic and (row["state_path"], row["batch"]) == (start.uri, start.step):
                continue
            raise ValueError(f"Foreign parent checkpoint in {output}")
        if (
            row.get("owner") not in (None, "child")
            or row.get("parent_state_path") != start.uri
            or row["state_path"] == start.uri
            or (start.uri and row["batch"] <= start.step)
        ):
            raise ValueError(f"Foreign checkpoint in {output}")
        if dynamic:
            if (
                start.uri
                and row["state_path"].partition("/weights/")[0]
                == start.uri.partition("/weights/")[0]
            ):
                raise ValueError(
                    f"Dynamic child references its parent's model namespace in {output}"
                )
            offset = row.get("next_prompt_offset")
            if (
                (start.uri and row.get("dynamic_run") != output.name)
                or type(offset) is not int
                or offset < start.offset + (row["batch"] - start.step) * batch
            ):
                raise ValueError(f"Foreign or incomplete dynamic checkpoint in {output}")
    return [r for r in rows if r.get("owner") != "parent"]


def disk_plan(output, base, targets, save_every, start, batch_size, group_size, state_bytes=None):
    if (
        not targets
        or targets != sorted(set(targets))
        or targets[0] <= start.step
        or min(batch_size, group_size) < 1
        or save_every < 0
    ):
        raise ValueError("Invalid target, batch size, group size or save cadence")
    row = latest(own_records(output, start))
    last = row["batch"] if row else start.step
    offset = row["next_prompt_offset"] if row and "next_prompt_offset" in row else start.offset
    remaining = max(0, targets[-1] - last)
    if not remaining:
        return True
    reference = row["state_path"] if row else start.uri
    if reference:
        state_bytes = state_file(base, reference).stat().st_size
    if state_bytes is None:
        print(
            "No reference archive yet. Set STATE_BYTES to estimate disk before the first save.",
            file=sys.stderr,
        )
        return True
    if state_bytes <= 0:
        raise ValueError("Expected a positive full-state archive size")
    saves = {t for t in targets if t > last}
    if save_every:
        saves.update(range((last // save_every + 1) * save_every, targets[-1] + 1, save_every))
    # Estimate refill volume from the committed cursor. Until the first child save,
    # use one wave per update. Allow 512 KiB per trajectory for logs/DB/adapters.
    prompts = max(batch_size, (offset - start.offset) / max(1, last - start.step))
    episodes = math.ceil(remaining * prompts * group_size)
    required = (1 + len(saves)) * state_bytes + 8 * GIB + episodes * 512 * 1024
    volume = base.resolve()
    while not volume.exists():
        volume = volume.parent
    free = shutil.disk_usage(volume).free
    print(
        f"free={free / GIB:.2f} GiB estimate={required / GIB:.2f} GiB, "
        f"cursor={offset} projected trajectories={episodes}, full saves={sorted(saves)}",
        file=sys.stderr,
    )
    return free >= required


def verify(output, target, start):
    cfg = snapshot(output)
    batch = cfg["dataset_builder"]["batch_size"]
    group_size = cfg["dataset_builder"]["group_size"]
    index = {r["batch"]: r for r in own_records(output, start)}
    if target not in index:
        raise ValueError(f"Missing child checkpoint at update {target}")
    updates = {
        r["progress/completed_updates"]: r
        for r in records(output / "metrics.jsonl")
        if "progress/completed_updates" in r
    }
    offset = start.offset
    for step in range(start.step + 1, target + 1):
        r = updates[step]
        if (
            r["step"] != step - 1
            or r["sampling/start_prompt_offset"] != offset
            or r["sampling/selected_groups"] != batch
        ):
            raise ValueError(f"Invalid update counters at {step}")
        if cfg.get("rollout_json_export", True):
            counts = {"constant": 0, "selected": 0, "surplus": 0}
            path = output / f"iteration_{step - 1:06d}" / "train_rollout_summaries.jsonl"
            for pos, rows in groupby(records(path), key=lambda x: x["prompt_offset"]):
                group = list(rows)
                reason = group[0]["selection"]
                if pos != offset or len(group) != group_size:
                    raise ValueError(f"Invalid group size/offset at update {step}")
                uniform = len({x["total_reward"] for x in group}) == 1
                if (
                    (reason == "constant") != uniform
                    or reason not in counts
                    or any(
                        x["sampling_client_step"] != step - 1 or x["selection"] != reason
                        for x in group
                    )
                ):
                    raise ValueError(f"Inconsistent selection/policy at prompt {pos}")
                counts[reason] += 1
                offset += 1
            if any(counts[k] != r[f"sampling/{k}_groups"] for k in counts):
                raise ValueError(f"Rollout counts disagree at update {step}")
        else:
            offset += r["sampling/attempted_groups"]
        if (
            offset != r["sampling/start_prompt_offset"] + r["sampling/attempted_groups"]
            or offset != r["sampling/next_prompt_offset"]
            or (step in index and index[step]["next_prompt_offset"] != offset)
        ):
            raise ValueError(f"Inconsistent cursor at update {step}")
        if r.get("optim/optimizer_steps", 1) and not all(
            math.isfinite(r[k]) for k in ("optim/kl_sample_train_v2", "optim/entropy")
        ):
            raise ValueError(f"Nonfinite KL or entropy at update {step}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    disk = commands.add_parser("disk-plan")
    check = commands.add_parser("verify")
    for sub in (disk, check):
        sub.add_argument("output", type=Path)
        sub.add_argument("--parent-run", type=Path)
        sub.add_argument("--parent-step", type=int)
    check.add_argument("--target", type=int, required=True)
    disk.add_argument("--ckpt-base", type=Path, required=True)
    disk.add_argument("--targets", nargs="+", type=int, required=True)
    disk.add_argument("--save-every", type=int, required=True)
    disk.add_argument("--batch-size", type=int, required=True)
    disk.add_argument("--group-size", type=int, required=True)
    disk.add_argument("--state-bytes", type=int)
    args = parser.parse_args()
    start = origin(args.output, args.parent_run, args.parent_step)
    if args.command == "verify":
        verify(args.output, args.target, start)
    elif not disk_plan(
        args.output,
        args.ckpt_base,
        args.targets,
        args.save_every,
        start,
        args.batch_size,
        args.group_size,
        args.state_bytes,
    ):
        raise SystemExit("Insufficient disk space for the requested saves")


if __name__ == "__main__":
    main()
