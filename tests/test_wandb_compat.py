"""Exercise real cookbook logging against a fake W&B client, without online runs."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.utils import ml_log

from scripts.cloud import upload_checkpoint
from search_r1.wandb_compat import persistent_wandb_run


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
    monkeypatch.setattr(ml_log, "wandb", fake)
    monkeypatch.setattr(ml_log, "_wandb_available", True)
    monkeypatch.setattr(ml_log, "_neptune_available", False)
    monkeypatch.setattr(ml_log, "_trackio_available", False)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for key in ("WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_RESUME"):
        monkeypatch.delenv(key, raising=False)
    return fake


def test_sync_logging_persists_identity_and_records_replayed_steps(tmp_path, fake_wandb):
    original = ml_log.WandbLogger
    with persistent_wandb_run():
        logger = ml_log.setup_logging(
            log_dir=tmp_path, wandb_project="test-project", do_configure_logging_module=False
        )
        logger.log_metrics({"reward": 0.5}, step=24)
        logger.log_metrics({"total": 1}, step=25)
        logger.close()
    assert ml_log.WandbLogger is original
    identity = json.loads((tmp_path / "wandb_run.json").read_text())
    assert identity == {"id": "training-id", "project": "test-project", "entity": "test-entity"}

    with persistent_wandb_run():
        resumed = ml_log.setup_logging(
            log_dir=tmp_path, wandb_project="test-project", do_configure_logging_module=False
        )
        resumed.log_metrics({"reward": 0.6}, step=25)
        resumed.log_metrics({"reward": 0.55}, step=24)  # Replay after checkpoint rollback.
        resumed.close()
    assert fake_wandb.init.call_args.kwargs["id"] == "training-id"
    assert fake_wandb.init.call_args.kwargs["resume"] == "allow"
    assert [call.args[0]["training/step"] for call in fake_wandb.log.call_args_list] == [
        24,
        25,
        25,
        24,
    ]
    assert all(call.kwargs == {"commit": True} for call in fake_wandb.log.call_args_list)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [24, 25, 25, 24]
    assert all("training/step" not in row for row in rows)


def test_identity_conflict_fails_before_wandb_init_and_restores_hook(tmp_path, fake_wandb):
    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"id": "saved", "project": "original", "entity": "test-entity"})
    )
    original = ml_log.WandbLogger
    with pytest.raises(ConfigurationError, match="conflicts"):
        with persistent_wandb_run():
            ml_log.WandbLogger(project="different", log_dir=tmp_path)
    fake_wandb.init.assert_not_called()
    assert ml_log.WandbLogger is original


def test_wandb_disabled_stays_disabled(tmp_path, fake_wandb):
    with persistent_wandb_run():
        logger = ml_log.setup_logging(log_dir=tmp_path, do_configure_logging_module=False)
        logger.log_metrics({"reward": 1}, step=3)
        logger.close()
    fake_wandb.init.assert_not_called()
    assert not (tmp_path / "wandb_run.json").exists()


def test_upload_does_not_inherit_training_identity(tmp_path, monkeypatch, fake_wandb):
    module = upload_checkpoint
    monkeypatch.setenv("WANDB_RUN_ID", "training-id")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    fake_wandb.util.generate_id.return_value = "upload-id"
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
