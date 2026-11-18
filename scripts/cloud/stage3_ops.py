"""Pod operations for the stage-two to stage-three handoff (standard library only)."""

import argparse
import json
import math
import re
import shutil
import sys
from itertools import groupby
from pathlib import Path

START = 100
PARENT = Path("/workspace/runs/train/search-r1-q35-64x8-s2")
CKPT_BASE = Path("/workspace/ckpts/skyrl")
CLEANUP_STEPS = {55, 60, 65, 70, 80, 85, 90, 95}
GIB = 1024**3


def records(path):
    if path.is_file():
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def parent_final(parent):
    index = list(records(parent / "checkpoints.jsonl"))
    if not index or (index[-1]["batch"], index[-1]["name"]) != (START, "final"):
        raise ValueError("Stage-two final checkpoint at update 100 is required")
    return index[-1]


def state_file(base, uri):
    match = re.fullmatch(r"tinker://([A-Za-z0-9_-]+)/weights/(final|[0-9]{6})", uri)
    if not match:
        raise ValueError(f"Unexpected SkyRL full-state path: {uri}")
    path = base / match[1] / (match[2] + ".tar.gz")
    if path.resolve() != base.resolve() / match[1] / path.name:
        raise ValueError(f"Unexpected symlink in checkpoint path: {path}")
    return path


def cleanup(parent, base, apply):
    final = parent_final(parent)
    if not state_file(base, final["state_path"]).is_file():
        raise ValueError("Missing step-100 parent archive")
    index = list(records(parent / "checkpoints.jsonl"))
    candidates = {}
    protected = set()
    for r in index:
        uri = r.get("state_path")
        if not uri:
            continue
        step, name = r["batch"], r["name"]
        if step in CLEANUP_STEPS and name == f"{step:06d}" and uri.endswith("/weights/" + name):
            candidates[state_file(base, uri)] = step
        else:
            protected.add(state_file(base, uri))
    if candidates.keys() & protected:
        raise ValueError("A candidate is shared with a protected checkpoint")
    total = 0
    for path, step in sorted(candidates.items(), key=lambda pair: pair[1]):
        if not path.is_file():
            print(f"already absent: step {step} {path}")
            continue
        size = path.stat().st_size
        print(f"{'delete' if apply else 'would delete'}: step {step} {size / GIB:.2f} GiB {path}")
        if apply:
            path.unlink()
        total += size
    print(f"{'released' if apply else 'reclaimable'} {total / GIB:.2f} GiB")


def own_records(output, parent_uri):
    index = list(records(output / "checkpoints.jsonl"))
    for r in index:
        if (
            r.get("stage3_run") != output.name
            or r.get("parent_state_path") != parent_uri
            or r["batch"] <= START
            or not isinstance(r.get("next_prompt_offset"), int)
            or r["state_path"].split("/")[2] == parent_uri.split("/")[2]
        ):
            raise ValueError(f"Foreign or incomplete checkpoint in {output}")
    return index


def disk_plan(parent, output, base, targets, save_every, *, include_smoke=False):
    if not targets or targets != sorted(set(targets)) or targets[0] <= START or save_every < 0:
        raise ValueError("Expected ascending child update targets and a nonnegative save cadence")
    if output.resolve().parent != parent.resolve().parent or output.resolve() == parent.resolve():
        raise ValueError("Expected an independent stage-three directory")
    uri = parent_final(parent)["state_path"]
    state_bytes = state_file(base, uri).stat().st_size
    children = own_records(output, uri)
    last = children[-1]["batch"] if children else START
    saves = {t for t in targets if t > last}
    if save_every:
        saves.update(t for t in range(last + 1, targets[-1] + 1) if t % save_every == 0)
    # A full archive of scratch space plus an estimate for database growth,
    # adapters and run logs. Recheck before each segment, using current df.
    reserve = state_bytes + (8 * GIB) + max(0, targets[-1] - last) * GIB // 4
    smoke_saves = []
    if include_smoke:
        smoke = own_records(output.with_name("search-r1-q35-64x8-s3-smoke"), uri)
        smoke_last = smoke[-1]["batch"] if smoke else START
        smoke_saves = [t for t in (102, 103) if t > smoke_last]
        reserve += max(0, 103 - smoke_last) * GIB // 4
    count = len(saves) + len(smoke_saves)
    required = count * state_bytes + reserve if count else 0
    free = shutil.disk_usage(base).free
    print(
        f"free {free / GIB:.1f} GiB, projected need {required / GIB:.1f} GiB "
        f"including reserve; remaining full saves {sorted(saves)}, additional smoke saves {smoke_saves}",
        file=sys.stderr,
    )
    return free >= required


def verify(parent, output, target):
    """Check the durable state, completed optimizer rows and ordered raw rollout evidence."""
    uri = parent_final(parent)["state_path"]
    index = own_records(output, uri)
    checkpoints = {r["batch"]: r for r in index}
    if target not in checkpoints:
        raise ValueError(f"Missing checkpoint at update {target}")
    updates = {
        r["progress/completed_updates"]: r
        for r in records(output / "metrics.jsonl")
        if "progress/completed_updates" in r
    }
    offset = START * 64
    for step in range(START + 1, target + 1):
        r = updates[step]
        if (
            r["step"] != step - 1
            or r["sampling/start_prompt_offset"] != offset
            or r["sampling/selected_groups"] != 64
        ):
            raise ValueError(f"Invalid update counters at {step}")
        counts = {"constant": 0, "selected": 0, "surplus": 0}
        path = output / f"iteration_{step - 1:06d}" / "train_rollout_summaries.jsonl"
        for pos, group in groupby(records(path), key=lambda x: x["prompt_offset"]):
            rows = list(group)
            if pos != offset or len(rows) != 8:
                raise ValueError(f"Invalid group size/offset at update {step}, expected {offset}")
            reason = rows[0]["selection"]
            uniform = len({x["total_reward"] for x in rows}) == 1
            if (reason == "constant") != uniform or any(
                x["sampling_client_step"] != step - 1 or x["selection"] != reason for x in rows
            ):
                raise ValueError(f"Inconsistent selection/policy at prompt {pos}")
            counts[reason] += 1
            offset += 1
        if (
            any(counts[k] != r[f"sampling/{k}_groups"] for k in counts)
            or sum(counts.values()) != r["sampling/attempted_groups"]
            or r["sampling/next_prompt_offset"] != offset
            or (step in checkpoints and checkpoints[step]["next_prompt_offset"] != offset)
        ):
            raise ValueError(f"Rollouts, metrics and checkpoint disagree at update {step}")
        diagnostics = [r["optim/kl_sample_train_v2"], r["optim/entropy"]]
        if not all(map(math.isfinite, diagnostics)):
            raise ValueError(f"Nonfinite KL or entropy at update {step}")
        print(f"update {step}: {counts}, next offset {offset}, KL/entropy {diagnostics}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("cleanup-stage2")
    clean.add_argument("--apply", action="store_true")
    disk = commands.add_parser("disk-plan")
    disk.add_argument("output", type=Path)
    disk.add_argument("--targets", nargs="+", type=int, required=True)
    disk.add_argument("--save-every", type=int, default=5)
    disk.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include pending smoke archives when planning the formal run",
    )
    check = commands.add_parser("verify")
    check.add_argument("output", type=Path)
    check.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    if args.command == "cleanup-stage2":
        cleanup(PARENT, CKPT_BASE, args.apply)
    elif args.command == "verify":
        verify(PARENT, args.output, args.target)
    elif not disk_plan(
        PARENT,
        args.output,
        CKPT_BASE,
        args.targets,
        args.save_every,
        include_smoke=args.include_smoke,
    ):
        raise SystemExit("Insufficient space for this plan; see the handoff disk options")


if __name__ == "__main__":
    main()
