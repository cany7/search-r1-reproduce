"""Forks, historical schema and crash boundaries. Every run lives in tmp_path."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chz
import pytest
from test_training_loop import Builder, config

from search_r1.env.search_env import SearchRLDataset
from search_r1.training import checkpoints, runtime
from search_r1.training.checkpoints import (
    commit_record,
    dataset_digest,
    full_records,
    prepare_run,
    run_lock,
    write_json,
)
from search_r1.training.logging import rollback_logs


@pytest.fixture
def dataset():
    return SearchRLDataset([Builder(i) for i in range(61)], 4)


def write_parent(path, dataset, **options):
    path.mkdir()
    settings = chz.asdict(config(path, **options))
    write_json(path / "config.json", settings | {"dataset_digest": dataset_digest(dataset)})
    for step, path_suffix in [(7, "first"), (11, "newer"), (7, "chosen")]:
        commit_record(
            path,
            {
                "name": path_suffix,
                "batch": step,
                "owner": "child",
                "state_path": f"tinker://parent/weights/{path_suffix}",
                "cumulative_tokens": {"trained": 121},
            },
        )
    commit_record(path, {"batch": 7, "sampler_path": "sampler-only"})
    return path


@pytest.mark.parametrize("mode,bs", [("ordinary", 4), ("dynamic", 3)])
def test_exact_step_registration_idempotence_and_independent_restart(tmp_path, dataset, mode, bs):
    parent = write_parent(tmp_path / "parent", dataset)
    child = tmp_path / "child"
    cfg = config(
        child,
        parent_run=str(parent),
        parent_step=7,
        max_steps=18,
        batch_size=bs,
        sampling_mode=mode,
        group_size=3,
        learning_rate=0.002,
    )
    before = (parent / "checkpoints.jsonl").read_bytes()
    with run_lock(child):
        origin = prepare_run(cfg, dataset)
        assert origin.record["state_path"].endswith("chosen")
        assert (origin.step, origin.offset, origin.record["cumulative_tokens"]) == (
            7,
            28,
            {"trained": 121},
        )
        assert prepare_run(cfg, dataset) == origin
        rows = full_records(child)
        if mode == "ordinary":
            assert len(rows) == 1
            assert (rows[0]["owner"], rows[0]["parent_run"], rows[0]["parent_step"]) == (
                "parent",
                str(parent),
                7,
            )
        else:
            assert not rows
        assert (parent / "checkpoints.jsonl").read_bytes() == before
        parent.rename(tmp_path / "archived")
        assert prepare_run(cfg, dataset) == origin
        snapshot = (child / "config.json").read_bytes()
        child_offset = 32 if mode == "ordinary" else 35
        commit_record(
            child,
            origin.fields
            | {
                "owner": "child",
                "batch": 8,
                "state_path": "child",
                "next_prompt_offset": child_offset,
            },
        )
        commit_record(child, {"batch": 9, "sampler_path": "later"})
        resumed = prepare_run(chz.replace(cfg, parent_run=None, parent_step=None), dataset)
        assert (resumed.step, resumed.offset, resumed.record["state_path"]) == (
            8,
            child_offset,
            "child",
        )
        with pytest.raises(ValueError, match="change the parent"):
            prepare_run(chz.replace(cfg, parent_step=8), dataset)
        with pytest.raises(ValueError, match="change the parent"):
            prepare_run(chz.replace(cfg, parent_run=str(tmp_path / "archived")), dataset)
        assert (child / "config.json").read_bytes() == snapshot


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"parent_step": 8}, "No matching full"),
        ({"seed": 999}, "seed"),
        ({"model_name": "other-model"}, "model_name"),
        ({"lora_rank": 64}, "lora_rank"),
        ({"batch_size": 3}, "batch_size"),
        ({"max_steps": 6}, "beyond requested"),
    ],
)
def test_invalid_forks_never_initialize_child(tmp_path, dataset, change, error):
    parent = write_parent(tmp_path / "parent", dataset)
    child = tmp_path / "child"
    args = {"parent_run": str(parent), "parent_step": 7, "max_steps": 10} | change
    cfg = config(child, **args)
    with run_lock(child), pytest.raises(ValueError, match=error):
        prepare_run(cfg, dataset)
    assert not (child / "config.json").exists()
    assert not full_records(child)


def test_reordered_dataset_or_dynamic_parent_rejected(tmp_path, dataset):
    parent = write_parent(tmp_path / "parent", dataset, sampling_mode="dynamic")
    child = tmp_path / "child"
    cfg = config(child, parent_run=str(parent), parent_step=7, max_steps=8)
    with run_lock(child):
        with pytest.raises(ValueError, match="ordinary parents"):
            prepare_run(cfg, dataset)
        dataset.env_group_builders.reverse()
        with pytest.raises(ValueError, match="prompt order"):
            prepare_run(cfg, dataset)


@pytest.mark.parametrize("offset", [None, -1, True, "32", 2, 100])
def test_legacy_dynamic_requires_an_exact_cursor_and_preserves_files(tmp_path, dataset, offset):
    cfg = config(tmp_path, sampling_mode="dynamic", max_steps=9)
    snapshot = chz.asdict(cfg)
    for key in ("sampling_mode", "parent_run", "parent_step"):
        snapshot.pop(key)
    snapshot["parent_state_path"] = "tinker://parent/weights/final"
    write_json(tmp_path / "config.json", snapshot)
    legacy = {
        "batch": 8,
        "name": "final",
        "state_path": "child",
        "stage3_run": tmp_path.name,
        "parent_state_path": snapshot["parent_state_path"],
    }
    commit_record(tmp_path, legacy | ({} if offset is None else {"next_prompt_offset": offset}))
    paths = [tmp_path / name for name in ("config.json", "checkpoints.jsonl")]
    before = [p.read_bytes() for p in paths]
    with run_lock(tmp_path), pytest.raises(ValueError, match="next_prompt_offset"):
        prepare_run(cfg, dataset)
    assert [p.read_bytes() for p in paths] == before


@pytest.mark.parametrize("mode", ["ordinary", "dynamic"])
def test_existing_legacy_configs_and_unmarked_indexes_read_only(tmp_path, dataset, mode):
    cfg = config(tmp_path, max_steps=12, sampling_mode=mode)
    saved = chz.asdict(cfg)
    for key in ("sampling_mode", "parent_run", "parent_step", "skyrl_compat"):
        saved.pop(key)
    del saved["dataset_builder"]["limit_search"]
    row = {"batch": 11, "state_path": "old", "name": "final"}
    if mode == "dynamic":
        saved["parent_state_path"] = "old-parent"
        row.update(stage3_run=tmp_path.name, parent_state_path="old-parent", next_prompt_offset=57)
    write_json(tmp_path / "config.json", saved)
    commit_record(tmp_path, row)
    paths = [tmp_path / name for name in ("config.json", "checkpoints.jsonl")]
    before = [p.read_bytes() for p in paths]
    with run_lock(tmp_path):
        start = prepare_run(cfg, dataset)
        assert start.record == row and start.offset == (57 if mode == "dynamic" else 44)
    assert [p.read_bytes() for p in paths] == before


def test_crash_between_snapshot_and_parent_registration_repairs_once(
    tmp_path, dataset, monkeypatch
):
    parent = write_parent(tmp_path / "parent", dataset)
    child = tmp_path / "child"
    cfg = config(child, parent_run=str(parent), parent_step=7, max_steps=8)
    with run_lock(child):
        with monkeypatch.context() as patch:

            def crash(*args):
                raise OSError("disk full")

            patch.setattr(checkpoints, "commit_record", crash)
            with pytest.raises(OSError, match="disk full"):
                prepare_run(cfg, dataset)
        snapshot = (child / "config.json").read_bytes()
        commit_record(parent, {"batch": 7, "state_path": "later-duplicate"})
        for _ in range(2):
            start = prepare_run(cfg, dataset)
            assert start.record["state_path"].endswith("chosen")
        assert (child / "config.json").read_bytes() == snapshot
        assert [r["owner"] for r in full_records(child)] == ["parent"]


def test_lock_precedes_logdir_delete_and_releases_on_error(tmp_path):
    keep = tmp_path / "config.json"
    keep.write_text("existing")
    with pytest.raises(RuntimeError, match="interrupt"), run_lock(tmp_path):
        with pytest.raises(ValueError, match="locked"), run_lock(tmp_path, behavior="delete"):
            pytest.fail("Second writer entered")
        assert keep.read_text() == "existing"
        raise RuntimeError("interrupt")
    with run_lock(tmp_path):
        assert keep.exists()


def test_replay_trims_torn_metric_tail_and_only_uncommitted_rollouts(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_bytes(b'{"step":0}\n{"step":1}\n{"step":2}\n{"st')
    for step in range(4):
        path = tmp_path / f"iteration_{step:06}"
        path.mkdir()
        (path / "train_rollout_summaries.jsonl").touch()
        (path / "user-notes").touch()
    (path / "train_rollout_summaries.pending.jsonl").touch()
    rollback_logs(tmp_path, 2)
    assert metrics.read_bytes() == b'{"step":0}\n{"step":1}\n'
    assert len(list(tmp_path.glob("*/train_rollout_summaries.jsonl"))) == 2
    assert len(list(tmp_path.glob("*/user-notes"))) == 4
    assert not list(tmp_path.glob("*/*.pending.jsonl"))


@pytest.mark.parametrize("skyrl", [False, True])
def test_weights_only_is_a_fresh_run_and_survives_initialization_retry(
    tmp_path, dataset, monkeypatch, skyrl
):
    cfg = config(tmp_path, load_checkpoint_path="weights", skyrl_compat=skyrl)
    with run_lock(tmp_path):
        prepare_run(cfg, dataset)
        retry = chz.replace(cfg, load_checkpoint_path=None)
        start = prepare_run(retry, dataset)
        assert start.step == start.offset == 0 and start.weights_path == "weights"
        training = SimpleNamespace(load_state=AsyncMock(), load_state_with_optimizer=AsyncMock())
        service = SimpleNamespace(
            create_lora_training_client=AsyncMock(return_value=training),
            create_training_client_from_state=AsyncMock(return_value=training),
            create_training_client_from_state_with_optimizer=AsyncMock(),
        )
        monkeypatch.setattr(runtime.tinker, "ServiceClient", lambda **kw: nullcontext(service))
        monkeypatch.setattr(runtime, "fetch_model_pricing", lambda model: None)
        monkeypatch.setattr(
            runtime.tinker_backend,
            "get_renderer_name_from_checkpoint_async",
            AsyncMock(return_value=None),
        )
        updates = AsyncMock()
        monkeypatch.setattr(runtime, "run_updates", updates)
        asyncio.run(runtime._train(retry, dataset, start, 2, SimpleNamespace(run=None)))
        assert updates.await_args.args[2:5] == (0, 2, 0)
        if skyrl:
            training.load_state.assert_awaited_once_with("weights")
        else:
            assert service.create_training_client_from_state.await_args.args == ("weights",)
        training.load_state_with_optimizer.assert_not_awaited()
        service.create_training_client_from_state_with_optimizer.assert_not_awaited()


def test_recover_only_unterminated_legacy_index_tail(tmp_path):
    path = tmp_path / "checkpoints.jsonl"
    committed = b'{"batch":1, "state_path": "old", "name":"final"}\n'
    path.write_bytes(committed + b'{"batch":2, "state_path":"unfinished')
    assert [r["batch"] for r in full_records(tmp_path)] == [1]
    assert path.read_bytes().endswith(b"unfinished")
    commit_record(tmp_path, {"batch": 2, "state_path": "complete"})
    assert path.read_bytes().startswith(committed)
    assert [r["batch"] for r in full_records(tmp_path)] == [1, 2]
    with path.open("ab") as out:
        out.write(b"invalid interior record\n")
    with pytest.raises(ValueError):
        full_records(tmp_path)


def test_parent_archive_cannot_be_registered_as_owned_by_dynamic_child(tmp_path):
    client = SimpleNamespace(
        save_state=AsyncMock(return_value=SimpleNamespace(path="parent")),
        save_weights_for_sampler=AsyncMock(),
    )
    manager = checkpoints.CheckpointManager(client, tmp_path, None)
    with pytest.raises(ValueError, match="parent archive"):
        asyncio.run(manager.save_final_async({"batch": 8, "parent_state_path": "parent"}))
    assert not full_records(tmp_path)
    client.save_weights_for_sampler.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [
        {"parent_run": "parent"},
        {"parent_step": 7},
        {"parent_run": " ", "parent_step": 0},
        {"parent_run": "parent", "parent_step": -1},
        {"parent_run": "parent", "parent_step": 7, "load_checkpoint_path": "weights"},
        {"load_checkpoint_path": ""},
    ],
)
def test_invalid_start_options_rejected_before_creating_a_run(tmp_path, options):
    with pytest.raises(ValueError):
        config(tmp_path / "child", **options)
    assert not (tmp_path / "child").exists()
