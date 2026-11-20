"""Exercise local event logging against a fake W&B client, without online runs."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_training_loop import config

from scripts.cloud import upload_checkpoint
from search_r1.training.logging import RunLogger


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = MagicMock()

    def init(**kwargs):
        run = MagicMock()
        run.id = kwargs.get("id") or "training-id"
        run.project = kwargs["project"]
        run.entity = kwargs.get("entity") or "test-entity"
        run.settings.mode = "online"
        return run

    fake.init.side_effect = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for key in ("WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_RESUME"):
        monkeypatch.delenv(key, raising=False)
    return fake


def test_sync_logging_persists_identity_and_records_replayed_steps(tmp_path, fake_wandb):
    cfg = config(tmp_path, wandb_project="test-project")
    logger = RunLogger(cfg)
    logger.log_metrics({"reward": 0.5}, step=24)
    logger.log_metrics({"total": 1}, step=25)
    logger.close()
    first = logger.run
    identity = json.loads((tmp_path / "wandb_run.json").read_text())
    assert identity == {"id": "training-id", "project": "test-project", "entity": "test-entity"}

    resumed = RunLogger(cfg)
    resumed.log_metrics({"reward": 0.6}, step=25)
    resumed.log_metrics({"reward": 0.55}, step=24)
    resumed.close()
    assert fake_wandb.init.call_args.kwargs["id"] == "training-id"
    assert fake_wandb.init.call_args.kwargs["resume"] == "allow"
    calls = first.log.call_args_list + resumed.run.log.call_args_list
    assert [call.args[0]["training/step"] for call in calls] == [
        24,
        25,
        25,
        24,
    ]
    assert all(call.kwargs == {"commit": True} for call in calls)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [24, 25, 25, 24]
    assert all("training/step" not in row for row in rows)


def test_identity_conflict_fails_before_wandb_init_for_conflicting_project(tmp_path, fake_wandb):
    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"id": "saved", "project": "original", "entity": "test-entity"})
    )
    with pytest.raises(ValueError, match="conflicts"):
        RunLogger(config(tmp_path, wandb_project="different"))
    fake_wandb.init.assert_not_called()


def test_wandb_disabled_stays_disabled(tmp_path, fake_wandb):
    logger = RunLogger(config(tmp_path))
    logger.log_metrics({"reward": 1}, step=3)
    logger.close()
    fake_wandb.init.assert_not_called()
    assert not (tmp_path / "wandb_run.json").exists()


def test_forks_ignore_inherited_parent_identity(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "parent")
    monkeypatch.setenv("WANDB_RESUME", "must")
    cfg = config(tmp_path, wandb_project="test-project")
    child = RunLogger(cfg, fork=True)
    assert child.run.id != "parent"
    assert fake_wandb.init.call_args.kwargs["resume"] == "never"
    child.close()
    resumed = RunLogger(cfg, fork=True)
    assert resumed.run.id == child.run.id
    assert fake_wandb.init.call_args.kwargs["resume"] == "allow"
    resumed.close()


def test_upload_does_not_inherit_training_identity(tmp_path, monkeypatch, fake_wandb):
    module = upload_checkpoint
    monkeypatch.setenv("WANDB_RUN_ID", "training-id")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setitem(
        sys.modules, "wandb.sdk.lib.runid", SimpleNamespace(generate_id=lambda: "upload-id")
    )
    module.upload(
        SimpleNamespace(project="test-project", run_name="simulation", step=25),
        [],
        {"training_wandb_run": "training-id"},
        {"id": "training-id", "project": "test-project", "entity": "test-entity"},
    )
    kwargs = fake_wandb.init.call_args.kwargs
    assert kwargs["id"] == "upload-id"
    assert kwargs["resume"] == "never"
    assert kwargs["config"]["training_wandb_run"] == "training-id"
    assert "WANDB_RUN_ID" not in module.os.environ
    assert "WANDB_RESUME" not in module.os.environ
