"""Delete full training states that are no longer needed; never touch sampler weights.

Kept:
- the full state at every eval step (needed later to evaluate the chosen checkpoint on the
  test set, or to resume from it);
- the latest full state (needed to resume).

Deleted:
- other full states (the periodic saves between eval steps, kept only as crash insurance);
- duplicate saves of one step (the cookbook writes both ``000025`` and ``final`` at the end of a
  segment; the later record, the one the driver and the cookbook use, is kept).

usage: ckpt_prune.py <checkpoints.jsonl> <ckpt_base> <eval_steps comma-separated> [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path


def state_file(ckpt_base: Path, state_path: str) -> Path:
    """tinker://<model>/weights/<name> -> <ckpt_base>/<model>/<name>.tar.gz"""
    rest = state_path.removeprefix("tinker://")
    model, _, name = rest.partition("/weights/")
    if not model or not name or "/" in name:
        raise ValueError(f"unexpected state path: {state_path}")
    return ckpt_base / model / f"{name}.tar.gz"


def plan(records: list[dict], eval_steps: set[int]) -> tuple[list[dict], list[dict]]:
    """Split checkpoint records into (keep, delete) by the rules above."""
    records = [r for r in records if r.get("state_path")]
    if not records:
        return [], []
    canonical: dict[int, dict] = {}
    for r in records:  # later records win, matching how the cookbook picks the last one
        canonical[r["batch"]] = r
    latest_batch = records[-1]["batch"]
    keep, delete = [], []
    for r in records:
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
    keep, delete = plan(records, eval_steps)
    keep_files = {state_file(args.ckpt_base, r["state_path"]) for r in keep}

    for r in keep:
        print(f"keep   step {r['batch']:>4} {r['name']:<8} {r['state_path']}")
    for r in delete:
        f = state_file(args.ckpt_base, r["state_path"])
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
