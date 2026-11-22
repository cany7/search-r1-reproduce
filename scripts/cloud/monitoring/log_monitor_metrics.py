"""Log dev-eval results and rollout behavior of a training run to a side W&B run.

The training run's own W&B history is never touched. This script
reads files that training and evaluation have already written and logs them to a separate run
in the same project:

- name ``<run_name>-monitor``, a fixed id derived from the run name, ``resume="allow"``,
  ``group=<run_name>``, ``job_type="monitor"``; every metric uses ``training/step`` as its axis.
- ``dev/*``: one point per dev eval, from ``<eval_root>/<run_name>-stepXXXX/summary.json``
  (and the base-model eval at step 0 with ``--base-eval``).
- ``rollout/*``: one point per training step, from
  ``<log_path>/iteration_XXXXXX/train_rollout_summaries.jsonl``.

Definitions (rollout):
- max_turns_rate: episodes stopped by the turn cap while still calling the tool (the final
  step's ``max_turns`` metric; the same termination as dev/max_turns_rate). An episode that
  answers on its last allowed turn is not counted; turns_frac_<max_turns> counts all episodes
  that used every turn.
- answered_rate: final step's ``format`` == 1 (same definition as the dev answer_rate).
- search_calls_per_episode: ``tool_call_N`` entries in the steps' logs.
- final_seq_len: last step's ob_len + ac_len (the longest sequence of the episode).
- answer_tokens_mean: last step's ac_len over answered episodes.

Idempotent: ``<log_path>/monitor_wandb_state.json`` records the size and mtime of every source
file already logged. Unchanged files are skipped; a file rewritten by a resumed segment is
logged again (the newer point is plotted alongside the old one). The state is only updated
after the W&B run is finished cleanly. A training step is logged only once metrics.jsonl has
a row for it, so a rollout file still being written is not read.

usage: log_monitor_metrics.py --run-name NAME --log-path DIR --eval-root DIR
           [--base-eval DIR] [--project P] [--wandb-dir DIR] [--dry-run]
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

STATE_FILE = "monitor_wandb_state.json"
DEV_FIELDS = (
    "em",
    "f1",
    "answer_rate",
    "em_when_answered",
    "search_calls",
    "no_search_rate",
    "tool_error_rate",
    "turns",
    "sampled_tokens",
)
DEV_SOURCE_FIELDS = ("em", "f1", "answer_rate")


def monitor_run_id(run_name: str) -> str:
    return "monitor-" + hashlib.sha1(run_name.encode()).hexdigest()[:20]


def fingerprint(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def dev_metrics(summary_file: Path) -> dict[str, float] | None:
    """Metrics of one dev eval, or None if it is incomplete or had errors."""
    s = json.loads(summary_file.read_text())
    by_source = s["by_source"]
    overall = by_source["OVERALL"]
    if s.get("run", {}).get("num_errors", 0) != 0 or not overall.get("n"):
        return None
    out = {f"dev/{k}": overall[k] for k in DEV_FIELDS if overall.get(k) is not None}
    out["dev/max_turns_rate"] = overall.get("termination", {}).get("max_turns", 0) / overall["n"]
    out["dev/n"] = overall["n"]
    for source, stats in sorted(by_source.items()):
        if source == "OVERALL":
            continue
        for k in DEV_SOURCE_FIELDS:
            if stats.get(k) is not None:
                out[f"dev/{source}/{k}"] = stats[k]
    return out


def p90(values: list[float]) -> float:
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def rollout_metrics(rows: list[dict], max_turns: int) -> dict[str, float]:
    """Behavior metrics of one training step's rollouts (one row per episode)."""
    n = len(rows)
    if n == 0:
        raise ValueError("no episodes")
    hit_cap = []
    answered = []
    searches = []
    turns = Counter()
    seq_lens = []
    answer_tokens = []
    by_source: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for r in rows:
        steps = r["steps"]
        last = steps[-1]
        final = last.get("metrics", {})
        cap = final.get("max_turns", 0.0) > 0
        ans = final.get("format", 0.0) == 1.0
        hit_cap.append(cap)
        answered.append(ans)
        searches.append(
            sum(1 for s in steps for k in s.get("logs", {}) if k.startswith("tool_call_"))
        )
        turns[len(steps)] += 1
        seq_lens.append(last["ob_len"] + last["ac_len"])
        if ans:
            answer_tokens.append(last["ac_len"])
        for tag in r.get("tags") or ["untagged"]:
            by_source[tag].append((cap, ans))

    out = {
        "rollout/episodes": n,
        "rollout/max_turns_rate": sum(hit_cap) / n,
        "rollout/answered_rate": sum(answered) / n,
        "rollout/search_calls_per_episode": sum(searches) / n,
        "rollout/final_seq_len_mean": sum(seq_lens) / n,
        "rollout/final_seq_len_p90": p90(seq_lens),
        "rollout/final_seq_len_max": max(seq_lens),
    }
    for k in range(1, max_turns + 1):
        out[f"rollout/turns_frac_{k}"] = turns.get(k, 0) / n
    if answer_tokens:
        out["rollout/answer_tokens_mean"] = sum(answer_tokens) / len(answer_tokens)
    for source, flags in sorted(by_source.items()):
        out[f"rollout/{source}/max_turns_rate"] = sum(c for c, _ in flags) / len(flags)
        out[f"rollout/{source}/answered_rate"] = sum(a for _, a in flags) / len(flags)
    return out


def completed_steps(log_path: Path) -> set[int]:
    """Training steps with a metrics.jsonl row, i.e. whose rollouts are fully written."""
    metrics_file = log_path / "metrics.jsonl"
    if not metrics_file.exists():
        return set()
    steps = set()
    for line in metrics_file.open():
        if line.strip():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a row still being appended
            if isinstance(rec.get("step"), int):
                steps.add(rec["step"])
    return steps


def collect(args: argparse.Namespace, state: dict) -> list[tuple[str, int, dict, list[int]]]:
    """Pending points as (kind, step, metrics, source fingerprint), sorted by step."""
    pending = []
    config_file = args.log_path / "config.json"
    max_turns = 5
    if config_file.exists():
        max_turns = json.loads(config_file.read_text())["dataset_builder"].get("max_turns", 5)

    done = completed_steps(args.log_path)
    for d in sorted(args.log_path.glob("iteration_*")):
        f = d / "train_rollout_summaries.jsonl"
        m = re.fullmatch(r"iteration_(\d+)", d.name)
        if not m or not f.exists():
            continue
        step = int(m.group(1))
        if step not in done:
            print(f"skip rollout step {step}: no metrics.jsonl row yet")
            continue
        fp = fingerprint(f)
        if state["rollout"].get(str(step)) == fp:
            continue
        rows = [r for r in load_jsonl(f) if r.get("split", "train") == "train"]
        if rows and "selection" in rows[0]:
            # The historical rollout/* axis is the sampler version. Preserve that
            # meaning; stage-three files also include the groups filtered out.
            selected = [r for r in rows if r["selection"] == "selected"]
            metrics = rollout_metrics(rows, max_turns)
            metrics.update(
                {
                    key.replace("rollout/", "selected_rollout/", 1): value
                    for key, value in rollout_metrics(selected, max_turns).items()
                }
            )
        else:
            metrics = rollout_metrics(rows, max_turns)
        pending.append(("rollout", step, metrics, fp))

    evals: list[tuple[int, Path]] = []
    if args.base_eval is not None:
        evals.append((0, args.base_eval / "summary.json"))
    if explicit := getattr(args, "dev_summary", None):
        step, path = explicit
        evals.append((int(step), Path(path)))
    else:
        pattern = re.compile(re.escape(args.run_name) + r"-step(\d+)")
        for d in sorted(args.eval_root.glob(f"{args.run_name}-step*")):
            m = pattern.fullmatch(d.name)
            if m and (d / "summary.json").exists():
                evals.append((int(m.group(1)), d / "summary.json"))
    for step, f in evals:
        if not f.exists():
            raise SystemExit(f"missing eval summary: {f}")
        fp = fingerprint(f)
        if state["dev"].get(str(step)) == fp:
            continue
        metrics = dev_metrics(f)
        if metrics is None:
            print(f"skip dev eval at step {step}: incomplete or with errors ({f})")
            continue
        pending.append(("dev", step, metrics, fp))

    pending.sort(key=lambda p: (p[1], p[0] != "rollout"))
    return pending


def log_to_wandb(args: argparse.Namespace, pending: list) -> str:
    # This is a side process; it must never resume the training run's history.
    for key in ("WANDB_RUN_ID", "WANDB_RESUME"):
        os.environ.pop(key, None)
    import wandb

    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    identity_file = args.log_path / "wandb_run.json"
    identity = json.loads(identity_file.read_text()) if identity_file.exists() else {}
    run_id = monitor_run_id(args.run_name)
    if run_id == identity.get("id"):
        raise SystemExit(f"monitor run id {run_id} equals the training run id")
    run = wandb.init(
        id=run_id,
        resume="allow",
        project=identity.get("project", args.project),
        entity=identity.get("entity"),
        name=f"{args.run_name}-monitor",
        group=args.run_name,
        job_type="monitor",
        dir=str(args.wandb_dir),
        config={"training_run": args.run_name, "training_wandb_run": identity.get("id")},
    )
    try:
        run.define_metric("training/step")
        run.define_metric("*", step_metric="training/step", step_sync=False)
        for _, step, metrics, _ in pending:
            run.log(metrics | {"training/step": step}, commit=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dev-summary",
        nargs=2,
        metavar=("STEP", "FILE"),
        help="Exact dev result for this driver segment",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-path", type=Path, required=True, help="training log_path")
    parser.add_argument(
        "--eval-root", type=Path, required=True, help="directory holding <run>-stepXXXX evals"
    )
    parser.add_argument(
        "--base-eval", type=Path, help="base-model dev eval directory, logged at step 0"
    )
    parser.add_argument("--project", default="search-r1-reproduce-experiment")
    parser.add_argument(
        "--wandb-dir", type=Path, help="local W&B files (default <log_path>/wandb_monitor)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would be logged")
    args = parser.parse_args()
    args.wandb_dir = args.wandb_dir or args.log_path / "wandb_monitor"
    if not args.log_path.is_dir():
        raise SystemExit(f"no such log path: {args.log_path}")

    if args.dry_run:  # read-only: no lock, no state, no W&B
        return log_pending(args)
    # Never unlink the lock file; the OS releases the lock on exit.
    with (args.log_path / f"{STATE_FILE}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return log_pending(args)


def log_pending(args: argparse.Namespace) -> int:
    state_file = args.log_path / STATE_FILE
    state = {"rollout": {}, "dev": {}}
    if state_file.exists():
        state.update(json.loads(state_file.read_text()))

    pending = collect(args, state)
    for kind, step, metrics, _ in pending:
        print(f"{kind} step {step}: {json.dumps(metrics, sort_keys=True)}")
    print(
        f"{len(pending)} points to log to {args.run_name}-monitor "
        f"(id {monitor_run_id(args.run_name)})"
    )
    if not pending or args.dry_run:
        return 0

    run_id = log_to_wandb(args, pending)
    for kind, step, _, fp in pending:
        state[kind][str(step)] = fp
    state["run_id"] = run_id
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(state_file)
    print(f"logged {len(pending)} points; state in {state_file}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # report and exit non-zero; never affect other processes
        print(f"log_monitor_metrics failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
