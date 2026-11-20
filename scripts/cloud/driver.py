"""Local orchestration for the Pod shell entry points. Child processes own SDK sessions."""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.cloud import stage3_ops as ops
from scripts.cloud.evaluation import CLOUD, evaluate, restart
from scripts.cloud.upload_checkpoint import marker_matches
from search_r1.checkpoint_index import latest, state_at


def overrides(arguments):
    values = {}
    for arg in arguments:
        key, sep, value = arg.partition("=")
        if not sep or not key:
            raise ValueError(f"Expected key=value, got {arg!r}")
        values[key] = value
    return values


def targets(start, total, interval):
    if interval < 1 or total < start:
        raise ValueError("Expected EVAL_EVERY > 0 and TOTAL_STEPS >= the fork step")
    if total == start:
        return []
    return [*range(start + interval, total, interval), total]


def run_path(name, kind="train"):
    return (Path(os.environ["POD_ROOT"]) / "runs" / kind / name).resolve()


def training_args(run, options, *, fresh=False, default_save_every=5):
    saved = {} if fresh else ops.snapshot(run)
    parent = options.pop("parent_run", os.environ.get("PARENT_RUN"))
    step = options.pop("parent_step", os.environ.get("PARENT_STEP"))
    parent = run_path(parent) if parent else None
    step = int(step) if step is not None else None
    start = ops.origin(run, parent, step, fresh=fresh)
    # Inherit the recipe on fork/restart. Explicit CLI overrides take precedence.
    source = saved or (ops.snapshot(parent) if parent else {})
    builder = source.get("dataset_builder", {})
    args = {
        key: source[key]
        for key in (
            "model_name",
            "lora_rank",
            "renderer_name",
            "learning_rate",
            "max_tokens",
            "wandb_project",
            "skyrl_compat",
        )
        if source.get(key) is not None
    }
    if "wandb_project" in source and source["wandb_project"] is None:
        args["wandb_project"] = "None"
    args.update(
        {
            key: builder[key]
            for key in (
                "seed",
                "limit_search",
                "max_turns",
                "prompt_name",
                "answer_score",
                "format_coef",
                "max_trajectory_tokens",
                "no_search_coef",
                "max_turns_unanswered_reward",
                "retrieval_url",
            )
            if builder.get(key) is not None
        }
    )
    if "retrieval_config" in builder:
        args["n_results"] = builder["retrieval_config"]["n_results"]
    args.update(
        batch_size=os.environ.get("BATCH_SIZE", builder.get("batch_size", 64)),
        group_size=os.environ.get("GROUP_SIZE", builder.get("group_size", 8)),
        save_every=os.environ.get("SAVE_EVERY", saved.get("save_every", default_save_every)),
        wandb_name=saved.get("wandb_name") or run.name,
    )
    mode = options.pop(
        "sampling_mode",
        os.environ.get("SAMPLING_MODE")
        or saved.get("sampling_mode")
        or ("dynamic" if saved.get("parent_state_path") else "ordinary"),
    )
    if mode not in ("ordinary", "dynamic"):
        raise ValueError("Expected ordinary or dynamic sampling")
    if saved and mode != saved.get(
        "sampling_mode", "dynamic" if saved.get("parent_state_path") else "ordinary"
    ):
        raise ValueError("Requested sampling mode differs from the saved run")
    if saved:
        if saved.get("session_name"):
            args["session_name"] = saved["session_name"]
        args["max_sampling_rounds"] = saved.get("max_sampling_rounds", 8)
    if mode == "ordinary":
        args["num_substeps"] = source.get("num_substeps", 1)
        if stream := source.get("stream_minibatch_config"):
            args.update(stream_minibatch=True, num_minibatches=stream["num_minibatches"])
    if parent:
        args.update(parent_run=str(parent), parent_step=step)
    args.setdefault("retrieval_url", os.environ["RETRIEVAL_URL"])
    args.update(options)
    return mode, start, args


def segment(run, target, mode, args, start, log):
    module = "search_r1.train_dynamic_sampling" if mode == "dynamic" else "search_r1.train"
    command = ["uv", "run", "--frozen", "python", "-m", module]
    for key, value in (args | {"max_steps": target, "log_path": str(run)}).items():
        command.append(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    with log.open("w") as stream:
        result = subprocess.run(
            command, cwd=os.environ["PROJ"], stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        print(f"Trainer exited {result.returncode} at target {target}; see {log}", file=sys.stderr)
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
    if ops.snapshot(run).get("parent_checkpoint"):
        # Training resolved the actual cursor, including epoch boundaries in old parents.
        pinned = ops.origin(run)
        if (pinned.step, pinned.uri) != (start.step, start.uri):
            raise ValueError("Trainer published a different fork origin")
        start = pinned
    record = latest(ops.own_records(run, start))
    if not record or record["batch"] != target:
        raise ValueError(f"Expected a child checkpoint at {target}")
    return start


def check_disk(run, steps, start, args):
    estimate = os.environ.get("STATE_BYTES")
    if not ops.disk_plan(
        run,
        Path(os.environ["CKPT_BASE"]),
        steps,
        int(args["save_every"]),
        start,
        int(args["batch_size"]),
        int(args["group_size"]),
        int(estimate) if estimate else None,
    ):
        raise RuntimeError("Insufficient space before training")


def monitor_dev(project, run, target, summary, logs):
    monitor = [
        project / ".venv/bin/python",
        CLOUD / "log_monitor_metrics.py",
        "--run-name",
        run.name,
        "--log-path",
        run,
        "--eval-root",
        project / "runs/eval",
        "--dev-summary",
        str(target),
        summary,
    ]
    if base_eval := os.environ.get("BASE_EVAL_DIR"):
        monitor.extend(["--base-eval", base_eval])
    with (logs / f"{run.name}_monitor.log").open("a") as out:
        try:
            subprocess.run(monitor, stdout=out, stderr=subprocess.STDOUT, timeout=600, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            return f"WARNING: monitor at {target}: {error}"
    return None


def start_upload(project, run, target, previous, summary, base, markers, logs):
    command = [
        project / ".venv/bin/python",
        CLOUD / "upload_checkpoint.py",
        "--run-name",
        run.name,
        "--log-path",
        run,
        "--step",
        str(target),
        "--prev-step",
        str(previous),
        "--eval-summary",
        summary,
        "--ckpt-base",
        base,
        "--marker-dir",
        markers,
    ]
    with tempfile.NamedTemporaryFile(
        prefix=f"{run.name}_upload_step{target:04d}.log.", dir=logs, delete=False
    ) as out:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return process, out.name


def dev_driver(name, options):
    project = Path(os.environ["PROJ"])
    logs = Path(os.environ["POD_ROOT"]) / "logs"
    run = run_path(options.pop("log_path", name))
    total = int(options.pop("max_steps", os.environ.get("TOTAL_STEPS", "100")))
    mode, start, args = training_args(run, options)
    if args.setdefault("behavior_if_log_dir_exists", "resume") != "resume":
        raise ValueError("Segmented training requires behavior_if_log_dir_exists=resume")
    fixed = int(os.environ.get("UPLOAD_FROM", start.step))
    # Older ordinary forks could start with a manually seeded, unmarked parent row.
    if fixed != start.step and (
        start.uri or fixed < 0 or state_at(run, fixed).get("owner") is not None
    ):
        raise ValueError("UPLOAD_FROM must match the pinned fork step")
    schedule = targets(fixed, total, int(os.environ.get("EVAL_EVERY", 25)))
    history_file = run / "driver_targets.json"
    history = set(json.loads(history_file.read_bytes())) if history_file.exists() else set()
    schedule = sorted(set(schedule) | {t for t in history if fixed < t <= total})
    log = logs / f"{run.name}_driver.log"

    def report(message):
        print(message, flush=True)
        with log.open("a") as out:
            print(message, file=out)

    report(f"{mode} run {run.name}, origin {fixed}, targets {schedule}")
    processes = []
    previous = fixed
    expected_markers = []
    base = Path(os.environ["CKPT_BASE"])
    markers = run / "uploads"
    upload = os.environ.get("UPLOAD", "1") == "1"
    restarted = False
    for target in schedule:
        row = latest(ops.own_records(run, start))
        current = row["batch"] if row else start.step
        if current < target:
            if mode == "dynamic":
                check_disk(run, [target], start, args)
            report(f"training {current} -> {target}")
            subprocess.run(
                [
                    "curl",
                    "-fsS",
                    "-X",
                    "POST",
                    os.environ["RETRIEVAL_URL"] + "/retrieve",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    '{"queries":["test"],"topk":1,"return_scores":true}',
                ],
                stdout=subprocess.DEVNULL,
                check=True,
            )
            restart(f"{run.name}_server_train_{target}", log, "training")
            restarted = True
            start = segment(
                run, target, mode, args, start, logs / f"{run.name}_train_to_{target}.log"
            )
        state_at(run, target)
        if mode == "dynamic":
            ops.verify(run, target, start)
        if target not in history:
            history.add(target)
            pending = history_file.with_suffix(".pending")
            pending.write_text(json.dumps(sorted(history)) + "\n")
            pending.replace(history_file)
        summary, evaluated = evaluate(run, target, "dev", {}, log)
        restarted |= evaluated
        report(f"dev at {target}: {summary}")
        if warning := monitor_dev(project, run, target, summary, logs):
            report(warning)
        if os.environ.get("PRUNE_STATES", "1") == "1":
            with log.open("a") as out:
                subprocess.run(
                    [
                        sys.executable,
                        CLOUD / "ckpt_prune.py",
                        run / "checkpoints.jsonl",
                        base,
                        ",".join(map(str, sorted(history | set(schedule)))),
                    ],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        marker = markers / f"step{target:04d}.json"
        expected_markers.append((marker, summary, previous))
        if upload and not marker_matches(marker, summary, previous):
            process, upload_log = start_upload(
                project, run, target, previous, summary, base, markers, logs
            )
            processes.append(process)
            report(f"Uploading ({previous}, {target}]; log {upload_log}")
        previous = target
    if restarted:
        subprocess.run(
            ["tmux", "kill-session", "-t", "skyrl"], check=False, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["uv", "run", "--frozen", "--extra", "tinker", "ray", "stop", "--force"],
            cwd=os.environ["SKYRL"],
            check=True,
        )
    statuses = [p.wait() for p in processes]
    if upload and (any(statuses) or any(not marker_matches(*entry) for entry in expected_markers)):
        raise RuntimeError("W&B upload missing or failed; rerun to retry")
    report(f"Completed {run.name} through step {total}")


def smoke(name, batch, total, options):
    logs = Path(os.environ["POD_ROOT"]) / "logs"
    run = run_path(options.pop("log_path", name), "tests")
    split = os.environ.get("RESTART_AT")
    behavior = options.pop("behavior_if_log_dir_exists", "resume" if split else "delete")
    target = int(options.pop("max_steps", total))
    mode, start, args = training_args(
        run,
        {"batch_size": batch, "wandb_name": f"test-{run.name}", **options},
        fresh=behavior == "delete",
        default_save_every=25,
    )
    if behavior not in ("delete", "resume", "raise", "ask"):
        raise ValueError("Invalid log directory behavior")
    if target <= start.step or (split and not start.step < int(split) < target):
        raise ValueError("Expected origin < RESTART_AT < final target")
    steps = [int(split), target] if split else [target]
    for number, step in enumerate(steps):
        policy = behavior if number == 0 else "resume"
        row = latest(ops.own_records(run, start)) if policy == "resume" else None
        if row and row["batch"] >= step:
            state_at(run, step)
        else:
            if mode == "dynamic" and (policy == "resume" or not run.exists()):
                check_disk(run, steps[number:], start, args)
            if (
                number
                or (row and policy == "resume")
                or os.environ.get("RESTART_BEFORE_TRAIN") == "1"
            ):
                restart(f"{run.name}_server_{step}", logs / f"{run.name}_restart.log", "training")
            log = logs / (f"train_{run.name}_{step}.log" if split else f"train_{run.name}.log")
            times = logs / "train_tests.times"
            with times.open("a") as out:
                print(f"START {name} {int(time.time())}", file=out, flush=True)
                try:
                    start = segment(
                        run, step, mode, args | {"behavior_if_log_dir_exists": policy}, start, log
                    )
                except BaseException:
                    print(f"END {name} {int(time.time())} failed", file=out)
                    raise
                print(f"END {name} {int(time.time())} rc=0", file=out)
        if mode == "dynamic":
            ops.verify(run, step, start)


def main():
    action, *args = sys.argv[1:]
    # A single combined runtime serves every run. Keep restarts and standalone
    # evaluations from switching another driver's loaded model.
    lockfile = Path(os.environ["POD_ROOT"]) / "logs/.skyrl-driver.lock"
    with lockfile.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == "train":
            dev_driver(args[0], overrides(args[1:]))
        elif action == "eval":
            run = run_path(args[0])
            summary, _ = evaluate(
                run, args[1], args[2], overrides(args[3:]), lockfile.parent / f"{run.name}_eval.log"
            )
            print(summary)
        elif action == "smoke":
            smoke(*args[:3], overrides(args[3:]))
        else:
            raise ValueError(f"Unknown driver action {action}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
