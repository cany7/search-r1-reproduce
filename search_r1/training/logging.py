# Adapted from tinker-cookbook utils/ml_log.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: Run-scoped W&B event history, JSONL replay and local progress messages.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""Local JSONL and W&B logging for a locked training run."""

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import chz

from search_r1.training.checkpoints import atomic_write, write_json


def rollback_logs(directory, step):
    """An iteration numbered N is committed by a full checkpoint at N+1."""
    path = directory / "metrics.jsonl"
    if path.exists():
        original = path.read_bytes()
        keep = bytearray()
        for line in original.splitlines(keepends=True):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                if line.endswith(b"\n"):
                    raise
                break  # Interrupted append to the uncommitted tail.
            if row["step"] < step:
                keep.extend(line if line.endswith(b"\n") else line + b"\n")
        if original != keep:
            atomic_write(path, keep)
    for path in directory.glob("iteration_*"):
        match = re.fullmatch(r"iteration_(\d+)", path.name)
        if match and int(match[1]) >= step:
            for name in ("train_rollout_summaries.jsonl", "train_rollout_summaries.pending.jsonl"):
                (path / name).unlink(missing_ok=True)


class RunLogger:
    def __init__(self, config, *, fork=False):
        self.directory = Path(config.log_path)
        self.run = None
        if config.wandb_project is None:
            return
        import wandb

        path = self.directory / "wandb_run.json"
        identity = json.loads(path.read_bytes()) if path.exists() else {}
        if path.exists():
            if not isinstance(identity, dict) or not all(
                isinstance(identity.get(k), str) and identity[k]
                for k in ("id", "project", "entity")
            ):
                raise ValueError("Invalid wandb_run.json")
            if (
                identity["project"] != config.wandb_project
                or os.environ.get("WANDB_ENTITY", identity["entity"]) != identity["entity"]
                or (not fork and os.environ.get("WANDB_RUN_ID", identity["id"]) != identity["id"])
            ):
                raise ValueError("W&B identity conflicts with wandb_run.json")
        # A fresh fork must also be independent if the shell exports its parent's run id.
        run_id = identity.get("id") or (uuid4().hex if fork else os.environ.get("WANDB_RUN_ID"))
        self.run = wandb.init(
            project=config.wandb_project,
            entity=identity.get("entity"),
            id=run_id,
            resume="allow" if identity else ("never" if fork else None),
            name=config.wandb_name,
            dir=str(self.directory),
            config=chz.asdict(config),
        )
        try:
            if self.run.settings.mode == "online":
                active = {"id": self.run.id, "entity": self.run.entity, "project": self.run.project}
                if identity and identity != active:
                    raise ValueError("W&B returned an unexpected run identity")
                if not identity:
                    write_json(path, active)
            self.run.define_metric("training/step")
            self.run.define_metric("*", step_metric="training/step", step_sync=False)
        except BaseException:
            self.run.finish(exit_code=1)
            raise

    def log_metrics(self, metrics, step):
        with (self.directory / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(metrics | {"step": step}) + "\n")
            out.flush()
            os.fsync(out.fileno())
        if self.run is not None:
            self.run.log(metrics | {"training/step": step}, commit=True)
        self.message(
            f"Update {metrics.get('progress/completed_updates', step)} "
            f"optimizer steps={metrics.get('optim/optimizer_steps', 0)}"
        )

    def message(self, text):
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {text}"
        with (self.directory / "logs.log").open("a") as out:
            print(line, file=out)
        print(line, file=sys.stderr, flush=True)

    def close(self, *, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
