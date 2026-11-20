"""Exercise real pruning only against archives created inside pytest's tmp_path."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cloud.ckpt_prune import plan, state_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cloud/ckpt_prune.py"


def record(step, model="child", name=None, owner="child"):
    result = {"batch": step, "name": name or f"{step:06d}", "owner": owner}
    result["state_path"] = f"tinker://{model}/weights/{result['name']}"
    if owner is None:
        del result["owner"]
    return result


def prune(tmp_path, records, *extra):
    root = tmp_path / "archives"
    root.mkdir(exist_ok=True)
    paths = set()
    for row in records:
        if row.get("state_path"):
            path = state_file(root, row["state_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test archive")
            paths.add(path)
    index = tmp_path / "checkpoints.jsonl"
    contents = "".join(json.dumps(r) + "\n" for r in records)
    index.write_text(contents)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(index), str(root), "12", *extra],
        capture_output=True,
        text=True,
        check=True,
    )
    assert index.read_text() == contents
    return root, paths, result


def test_parent_and_duplicate_uri_are_protected(tmp_path):
    parent = record(7, model="parent", name="final", owner="parent")
    duplicate = parent | {"owner": "child"}
    rows = [parent, duplicate, record(8), record(12), record(13), record(18)]
    root, _, _ = prune(tmp_path, rows)
    for row in (parent, rows[3], rows[5]):
        assert state_file(root, row["state_path"]).exists()
    for row in (rows[2], rows[4]):
        assert not state_file(root, row["state_path"]).exists()


@pytest.mark.parametrize("owner", [None, "unknown"])
def test_unknown_legacy_parent_skips_entire_index(tmp_path, owner):
    rows = [record(7, model="parent", owner=owner), record(8), record(18)]
    keep, delete = plan(rows, {12})
    assert keep == rows and not delete
    _, paths, result = prune(tmp_path, rows)
    assert all(p.exists() for p in paths)
    assert "skip automatic pruning" in result.stdout


def test_last_complete_record_wins_and_sampler_only_is_ignored(tmp_path):
    rows = [record(8), record(12), record(12, name="final"), record(18)]
    rows.append({"batch": 99, "sampler_path": "tinker://other/sampler_weights/final"})
    root, _, _ = prune(tmp_path, rows)
    assert not state_file(root, rows[0]["state_path"]).exists()
    assert not state_file(root, rows[1]["state_path"]).exists()
    assert state_file(root, rows[2]["state_path"]).exists()
    assert state_file(root, rows[3]["state_path"]).exists()


def test_dry_run_preserves_archives(tmp_path):
    _, paths, result = prune(tmp_path, [record(8), record(18)], "--dry-run")
    assert all(p.exists() for p in paths)
    assert "would delete" in result.stdout


def test_only_parent_or_sampler_records_do_not_delete(tmp_path):
    rows = [record(7, owner="parent"), {"batch": 8, "sampler_path": "sampler"}]
    _, paths, _ = prune(tmp_path, rows)
    assert all(p.exists() for p in paths)
    assert plan(rows[1:], set()) == ([], [])


def test_parent_symlink_alias_protects_actual_archive(tmp_path):
    root = tmp_path / "archives"
    (root / "parent").mkdir(parents=True)
    (root / "alias").symlink_to(root / "parent", target_is_directory=True)
    rows = [record(7, model="parent", owner="parent"), record(7, model="alias"), record(18)]
    root, _, _ = prune(tmp_path, rows)
    assert state_file(root, rows[0]["state_path"]).exists()


@pytest.mark.parametrize(
    "uri", ["other://m/weights/a", "tinker://../weights/a", "tinker://m/weights/../a"]
)
def test_invalid_archive_path_is_rejected(tmp_path, uri):
    with pytest.raises(ValueError):
        state_file(tmp_path, uri)


def test_archive_symlink_must_stay_inside_root(tmp_path):
    root = tmp_path / "archives"
    root.mkdir()
    (root / "model").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        state_file(root, "tinker://model/weights/final")


def test_dynamic_child_only_index_protects_parent_uri_alias(tmp_path):
    root = tmp_path / "archives"
    (root / "parent").mkdir(parents=True)
    (root / "alias").symlink_to(root / "parent", target_is_directory=True)
    rows = [record(8, "alias", "final"), record(12)]
    for r in rows:
        r["parent_state_path"] = "tinker://parent/weights/final"
    prune(tmp_path, rows)
    assert (root / "parent/final.tar.gz").is_file()
