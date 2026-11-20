# Adapted from the local training/checkpoints.py reader.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
"""Read the local full-state index without loading training or SDK dependencies."""

import json


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


def state_at(directory, step):
    record = latest([r for r in full_records(directory) if r["batch"] == step])
    if record is None:
        raise ValueError(f"No full checkpoint at step {step} in {directory}")
    return record
