# Adapted from tinker-cookbook checkpoint_utils.py and cli_utils.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: Locked run directories, exact-step forks and atomic publication of owned checkpoints.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""Resolve a run's immutable starting point and publish resumable local index records."""

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chz

from search_r1.training.config import from_snapshot


def atomic_write(path, data):
    """Single writer (the run lock). Readers see the old or the complete new file."""
    temporary = path.with_name("." + path.name + ".pending")
    with temporary.open("wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def read_config(directory):
    return json.loads((directory / "config.json").read_bytes())


LOCK = ".stage3.lock"  # Interoperate with an already running legacy stage-three process.


@contextmanager
def run_lock(directory, behavior="resume"):
    try:
        directory.mkdir(parents=True)
        existed = False
    except FileExistsError:
        if not directory.is_dir():
            raise
        existed = True
    with (directory / LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Run already locked: {directory}") from exc
        contents = [p for p in directory.iterdir() if p.name != LOCK]
        if existed:
            while behavior == "ask":
                choice = input(f"{directory} exists. Choose delete, resume or exit: ").strip()
                if choice == "exit":
                    raise SystemExit(0)
                if choice in ("delete", "resume"):
                    behavior = choice
            if behavior == "raise":
                raise FileExistsError(directory)
            if behavior == "delete":
                for path in contents:
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        yield


def index_bytes(path):
    """An old append-only writer could leave a partial last line. Keep its complete prefix."""
    if not path.exists():
        return b""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        prefix, separator, tail = data.rpartition(b"\n")
        try:
            json.loads(tail)
        except ValueError:
            return prefix + separator
        data += b"\n"
    return data


def full_records(directory):
    rows = (
        json.loads(line)
        for line in index_bytes(directory / "checkpoints.jsonl").splitlines()
        if line.strip()
    )
    return [
        row
        for row in rows
        if isinstance(row.get("state_path"), str)
        and row["state_path"].strip()
        and type(row.get("batch")) is int
        and row["batch"] >= 0
    ]


def latest(rows):
    # At equal step, prefer the last full record in file order.
    return max(reversed(rows), key=lambda row: row["batch"], default=None)


def commit_record(directory, record):
    path = directory / "checkpoints.jsonl"
    previous = index_bytes(path)
    atomic_write(path, previous + json.dumps(record).encode() + b"\n")


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for builder in dataset.env_group_builders:
        digest.update(json.dumps(builder.datum, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def check_compatible(snapshot, config, digest, *, fork):
    saved = from_snapshot(snapshot)
    for key in ("model_name", "lora_rank", "renderer_name"):
        if getattr(saved, key) != getattr(config, key):
            raise ValueError(f"Incompatible {key}")
    for key in ("seed", "model_name_for_tokenizer"):
        if getattr(saved.dataset_builder, key) != getattr(config.dataset_builder, key):
            raise ValueError(f"Incompatible dataset {key}")
    if snapshot.get("dataset_digest", digest) != digest:
        raise ValueError("Incompatible dataset contents or prompt order")
    if (config.sampling_mode == "ordinary" or not fork) and (
        saved.dataset_builder.batch_size != config.dataset_builder.batch_size
    ):
        raise ValueError("Incompatible batch_size")
    if fork:
        if saved.sampling_mode != "ordinary":
            raise ValueError("Only ordinary parents support forking")
        return
    if saved.sampling_mode != config.sampling_mode:
        raise ValueError("Incompatible sampling_mode for resume")
    # Preserve the rollout and reward settings consumed by offline evaluation.
    saved_builder = chz.asdict(saved.dataset_builder)
    for key, value in chz.asdict(config.dataset_builder).items():
        if key != "retrieval_url" and saved_builder[key] != value:
            raise ValueError(f"Incompatible dataset_builder.{key} for resume")
    for key in ("temperature", "max_tokens", "loss_fn", "loss_fn_config"):
        if getattr(saved, key) != getattr(config, key):
            raise ValueError(f"Incompatible {key} for resume")


def cursor(record, batch_size, prompt_count, *, dynamic):
    step = record["batch"]
    size = prompt_count // batch_size * batch_size
    # Ordinary epochs skip their partial last batch. The saved cursor is the end
    # of the last consumed batch (including at the boundary between two epochs).
    expected = ((step - 1) * batch_size % size + batch_size) if step and size else 0
    offset = record.get("next_prompt_offset", None if dynamic else expected)
    if type(offset) is not int or not 0 <= offset <= prompt_count:
        raise ValueError(f"Invalid or missing next_prompt_offset at step {step}")
    if not dynamic and offset != expected:
        raise ValueError(f"Ordinary next_prompt_offset disagrees with step {step}")
    return offset


@dataclass
class Start:
    record: dict
    offset: int
    fields: dict
    weights_path: str | None

    @property
    def step(self):
        return self.record.get("batch", 0)


def separate_directories(parent, child):
    if parent == child or parent in child.parents or child in parent.parents:
        raise ValueError("Parent and child must have separate run directories")


def child_state(rows, directory, dynamic, parent_uri, parent, step):
    own = []
    registered = False
    for row in rows:
        if row.get("owner") not in (None, "parent", "child"):
            raise ValueError("Unknown checkpoint owner")
        if row.get("parent_state_path") != parent_uri or (
            row.get("stage3_run") is not None and row["stage3_run"] != directory.name
        ):
            raise ValueError("Foreign checkpoint provenance in child index")
        if dynamic and parent_uri and row.get("stage3_run") != directory.name:
            raise ValueError("Missing dynamic child run identity")
        if row.get("parent_run", parent) != parent or row.get("parent_step", step) != step:
            raise ValueError("Foreign parent_run or parent_step in child index")
        if row.get("owner") == "parent":
            if dynamic or row["state_path"] != parent_uri or row["batch"] != step:
                raise ValueError("Foreign parent checkpoint")
            registered = True
        else:
            if parent_uri and (
                row["state_path"] == parent_uri or (step is not None and row["batch"] <= step)
            ):
                raise ValueError("Invalid child checkpoint")
            own.append(row)
    return latest(own), registered


def select_parent(source, step, expected_uri, directory, config, digest, count):
    separate_directories(source, directory)
    parent_settings = read_config(source)
    check_compatible(parent_settings, config, digest, fork=True)
    pinned = latest([r for r in full_records(source) if r["batch"] == step])
    if not pinned or (expected_uri and expected_uri != pinned["state_path"]):
        raise ValueError(f"No matching full parent checkpoint at step {step}")
    batch_size = parent_settings["dataset_builder"]["batch_size"]
    if step > count // batch_size * parent_settings.get("num_epochs", 1):
        raise ValueError("Parent step exceeds the configured dataset epochs")
    if pinned.get("dataset_digest", digest) != digest:
        raise ValueError("Incompatible parent checkpoint dataset")
    return {
        **pinned,
        "next_prompt_offset": cursor(pinned, batch_size, count, dynamic=False),
    }


def prepare_run(config, dataset):
    """Called under the child lock. Pin parent provenance before registering its state."""
    directory = Path(config.log_path).resolve()
    config_file = directory / "config.json"
    saved = read_config(directory) if config_file.exists() else {}
    if not saved and any(p.name not in (LOCK, ".config.json.pending") for p in directory.iterdir()):
        raise ValueError("Refusing to adopt a populated run directory without config.json")
    digest = dataset_digest(dataset)
    count = len(dataset.env_group_builders)
    if saved:
        check_compatible(saved, config, digest, fork=False)
        if config.load_checkpoint_path and config.load_checkpoint_path != saved.get(
            "load_checkpoint_path"
        ):
            raise ValueError("Cannot change weights-only initializer")
    requested = str(Path(config.parent_run).expanduser().resolve()) if config.parent_run else None
    rows = full_records(directory)
    previous = latest(rows) or {}
    bound_parent = saved.get("parent_run") or previous.get("parent_run")
    parent = bound_parent or requested
    step = saved.get("parent_step")
    if step is None:
        step = previous.get("parent_step", config.parent_step)
    pinned = saved.get("parent_checkpoint")
    parent_uri = saved.get("parent_state_path")
    if (
        requested
        and saved
        and (
            (not saved.get("parent_run") and not parent_uri)
            or requested != parent
            or step != config.parent_step
        )
    ):
        raise ValueError("Cannot change the parent of an existing run")
    if (parent or parent_uri) and config.load_checkpoint_path:
        raise ValueError("A fork cannot use weights-only initialization")
    dynamic = config.sampling_mode == "dynamic"
    record, registered = child_state(rows, directory, dynamic, parent_uri, parent, step)
    # Old stage-three snapshots only named the parent's state URI. When a parent
    # is supplied, verify that binding even when resuming a child checkpoint.
    if parent and not pinned and not (record and bound_parent):
        pinned = select_parent(Path(parent), step, parent_uri, directory, config, digest, count)
        parent_uri = pinned["state_path"]
    if pinned and (pinned["state_path"] != parent_uri or pinned["batch"] != step):
        raise ValueError("Pinned parent checkpoint disagrees with run provenance")
    if parent_uri and not (record or pinned):
        raise ValueError("Specify the legacy parent_run and parent_step to recover its cursor")
    start = record or pinned or {}
    if start.get("dataset_digest", digest) != digest:
        raise ValueError("Incompatible checkpoint dataset contents or prompt order")
    offset = (
        cursor(start, config.dataset_builder.batch_size, count, dynamic=dynamic) if start else 0
    )
    if record and dynamic:
        first_step = pinned["batch"] if pinned else 0
        first_offset = pinned["next_prompt_offset"] if pinned else 0
        if (
            offset
            < first_offset + (record["batch"] - first_step) * config.dataset_builder.batch_size
        ):
            raise ValueError("Dynamic next_prompt_offset precedes consumed batches")
    if config.max_steps is not None and start.get("batch", 0) > config.max_steps:
        raise ValueError("Already beyond requested max_steps")
    fields = {"dataset_digest": digest}
    if parent is not None:
        fields.update(parent_run=parent, parent_step=step)
    if parent_uri:
        fields["parent_state_path"] = parent_uri
        if dynamic:
            fields["stage3_run"] = directory.name
    if not saved:
        snapshot = chz.asdict(config) | {"dataset_digest": digest, "parent_run": parent}
        if pinned:
            snapshot.update(parent_checkpoint=pinned, parent_state_path=parent_uri)
        write_json(config_file, snapshot)
    if pinned and not dynamic and not registered and not record:
        commit_record(
            directory,
            pinned | fields | {"owner": "parent", "parent_run": parent, "parent_step": step},
        )
    return Start(
        start, offset, fields, saved.get("load_checkpoint_path") or config.load_checkpoint_path
    )


class CheckpointManager:
    def __init__(self, client, directory, ttl_seconds):
        self.client = client
        self.directory = directory
        self.ttl_seconds = ttl_seconds

    async def _save(self, name, ttl, state):
        full = await self.client.save_state(name, ttl_seconds=ttl)
        if full.path == state.get("parent_state_path") and full.path:
            raise ValueError("SDK returned the parent archive for a child save")
        sampler = await self.client.save_weights_for_sampler(name, ttl_seconds=ttl)
        if not full.path or not sampler.path:
            raise ValueError("SDK returned an incomplete checkpoint")
        commit_record(
            self.directory,
            state
            | {
                "name": name,
                "owner": "child",
                "state_path": full.path,
                "sampler_path": sampler.path,
            },
        )

    async def save_periodic_async(self, step, loop_state):
        await self._save(f"{step:06d}", self.ttl_seconds, loop_state)

    async def save_final_async(self, loop_state):
        await self._save("final", None, loop_state)
