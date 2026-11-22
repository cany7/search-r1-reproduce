"""Delete full training states that are no longer needed; never touch sampler weights.

Kept:
- the full state at every eval step (needed later to evaluate the chosen checkpoint on the
  test set, or to resume from it);
- the latest full state (needed to resume).
- every parent-owned archive, including aliases/duplicate index entries.

New full-state records must declare owner="child" or owner="parent". If any
full-state record has unknown ownership, skip pruning the entire index: ownership
cannot be inferred safely from batch numbers or model IDs. Sampler-only records are
ignored.

Deleted:
- other full states (the periodic saves between eval steps, kept only as crash insurance);
- duplicate saves of one step (training writes both ``000025`` and ``final`` at the end of a
  segment; the later record, the one the driver and training use, is kept).

usage: ckpt_prune.py <checkpoints.jsonl> <ckpt_base> <eval_steps comma-separated> [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path


def state_file(ckpt_base: Path, state_path: str) -> Path:
    """tinker://<model>/weights/<name> -> <ckpt_base>/<model>/<name>.tar.gz"""
    if not state_path.startswith("tinker://"):
        raise ValueError(f"unexpected state path: {state_path}")
    rest = state_path.removeprefix("tinker://")
    model, _, name = rest.partition("/weights/")
    if any(
        not part or part in (".", "..") or "/" in part or "\\" in part for part in (model, name)
    ):
        raise ValueError(f"unexpected state path: {state_path}")
    result = (ckpt_base / model / f"{name}.tar.gz").resolve()
    if not result.is_relative_to(ckpt_base.resolve()):
        raise ValueError(f"state archive is outside checkpoint root: {state_path}")
    return result


def plan(records: list[dict], eval_steps: set[int]) -> tuple[list[dict], list[dict]]:
    """Split checkpoint records into (keep, delete) by the rules above."""
    records = [r for r in records if r.get("state_path")]
    if not records:
        return [], []
    if any(r.get("owner") not in ("parent", "child") for r in records):
        return records, []
    children = [r for r in records if r["owner"] == "child"]
    if not children:
        return records, []
    canonical: dict[int, dict] = {}
    for r in children:  # Last complete record at a step wins.
        canonical[r["batch"]] = r
    latest_batch = max(canonical)
    keep, delete = [], []
    for r in records:
        if r["owner"] == "parent":
            keep.append(r)
            continue
        is_canonical = canonical[r["batch"]] is r
        if is_canonical and (r["batch"] in eval_steps or r["batch"] == latest_batch):
            keep.append(r)
        else:
            delete.append(r)
    return keep, delete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints_jsonl", type=Path)
    parser.add_argument("ckpt_base", type=Path)
    parser.add_argument("eval_steps", help="comma-separated eval steps, e.g. 25,50,75,100")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eval_steps = {int(s) for s in args.eval_steps.split(",") if s}
    records = [json.loads(line) for line in args.checkpoints_jsonl.open() if line.strip()]
    if any(r.get("state_path") and r.get("owner") not in ("parent", "child") for r in records):
        print("skip automatic pruning: full-state ownership is unknown")
        return 0
    keep, delete = plan(records, eval_steps)
    keep_files = {state_file(args.ckpt_base, r["state_path"]) for r in keep}
    keep_files.update(
        state_file(args.ckpt_base, r["parent_state_path"])
        for r in records
        if r.get("parent_state_path")
    )
    # Resolve and validate the entire plan before deleting any file.
    delete_files = [(r, state_file(args.ckpt_base, r["state_path"])) for r in delete]

    for r in keep:
        print(f"keep   step {r['batch']:>4} {r['name']:<8} {r['state_path']}")
    for r, f in delete_files:
        if f in keep_files:  # never delete a file a kept record points to
            continue
        action = "would delete" if args.dry_run else "delete"
        if not f.exists():
            print(f"gone   step {r['batch']:>4} {r['name']:<8} {f}")
            continue
        size_gb = f.stat().st_size / 1e9
        print(f"{action} step {r['batch']:>4} {r['name']:<8} {f} ({size_gb:.1f} GB)")
        if not args.dry_run:
            f.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
