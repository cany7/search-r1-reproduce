"""Evaluate one fixed set using a precise full state (or the SkyRL base model)."""

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from search_r1.checkpoint_index import state_at

CLOUD = Path(__file__).resolve().parents[1]


def restart(name, log, phase):
    with log.open("a") as stream:
        result = subprocess.run(
            ["bash", CLOUD / "services" / "restart_skyrl.sh", name],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"server restart failed before {phase}; see {log}")


def evaluate(run, step, split, options, log):
    project = Path(os.environ["PROJ"])
    if split not in ("dev", "test"):
        raise ValueError("Expected dev or test")
    data = project / "eval_sets" / f"{split}.jsonl"
    specified = options.get("training_config_file") or os.environ.get("TRAINING_CONFIG_FILE")
    source = Path(specified or run / "config.json").resolve()
    training = None
    if step != "base" or specified or source.exists():
        training = json.loads(source.read_bytes())
    data_bytes = data.read_bytes()
    counts = Counter(
        json.loads(line)["data_source"] for line in data_bytes.splitlines() if line.strip()
    )
    limit = options.get("limit_per_source")
    expected = (
        sum(min(count, int(limit)) for count in counts.values())
        if limit not in (None, "None")
        else sum(counts.values())
    )
    if expected <= 0:
        raise ValueError("Empty evaluation set")
    state = None if step == "base" else state_at(run, int(step))["state_path"]
    args = {"concurrency": os.environ.get("EVAL_CONCURRENCY", "32"), **options}
    for key in (
        "run_name",
        "eval_file",
        "backend",
        "tinker_state_url",
        "tinker_checkpoint_url",
        "skyrl_base_eval",
    ):
        if key in options:
            raise ValueError(f"The evaluation driver selects {key}")
    args.update(
        backend="tinker",
        eval_file=str(data),
        tinker_base_url=os.environ["TINKER_BASE_URL"],
    )
    if training is not None:
        args["training_config_file"] = str(source)
    if state:
        args["tinker_state_url"] = state
    else:
        args["skyrl_base_eval"] = "true"
    digest = hashlib.sha256(data_bytes)
    digest.update(json.dumps([args, training], sort_keys=True).encode())
    label = "base" if state is None else f"step{int(step):04d}"
    name = f"{run.name}-{label}-{split}-{digest.hexdigest()[:16]}"
    output = project / "runs/eval" / name
    summary = output / "summary.json"
    receipt = output / ".complete"

    def complete():
        try:
            stats = json.loads(summary.read_bytes())
            effective = json.loads((output / "config.json").read_bytes())
            return (
                stats["run"]["num_errors"] == 0
                and stats["by_source"]["OVERALL"]["n"] == expected
                and effective["num_items"] == expected
                and effective["training_config"] == training
                and effective.get("tinker_state_url") == state
                and effective["eval_file"] == str(data)
            )
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            return False

    if receipt.is_file() and complete():
        return summary, False
    receipt.unlink(missing_ok=True)
    restart(f"{name}_server", log, "eval")
    args["run_name"] = name
    logfile = Path(os.environ["POD_ROOT"]) / "logs" / f"{name}.log"
    with logfile.open("w") as stream:
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "-m",
                "search_r1.offline_eval",
                *(f"{key}={value}" for key, value in args.items()),
            ],
            cwd=project,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not complete():
        raise ValueError(
            f"Incomplete {split} evaluation: expected {expected} successful items and zero errors; see {logfile}"
        )
    receipt.touch()
    return summary, True
