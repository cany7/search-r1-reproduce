import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import tinker

from scripts.cloud import log_monitor_metrics as monitor
from scripts.cloud import stage3_ops as ops
from search_r1 import train_dapo as dapo
from search_r1.backends.completers import TokensWithLogprobs
from search_r1.env.search_env import SearchRLDataset
from search_r1.env.types import TrajectoryGroup
from search_r1.training import sampling

ROOT = Path(__file__).resolve().parents[1]
PARENT_URI = "tinker://model_parent/weights/final"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def parent_run(tmp_path):
    parent = tmp_path / "s2"
    parent.mkdir()
    source = ROOT / "experiments/2026-11-16-a100-validation/training/t64_mb1p/config.json"
    config = json.loads(source.read_text())
    config["dataset_builder"]["limit_search"] = True
    config["wandb_project"] = None
    config["sampling_mode"] = "ordinary"  # New ordinary snapshots must still fork into dynamic mode.
    (parent / "config.json").write_text(json.dumps(config))
    write_jsonl(
        parent / "checkpoints.jsonl", [{"batch": 100, "name": "final", "state_path": PARENT_URI}]
    )
    return parent


def group(rewards):
    ob = tinker.ModelInput.from_ints([1, 2, 3])
    action = TokensWithLogprobs(tokens=[42], maybe_logprobs=[-0.3])
    transition = SimpleNamespace(
        ob=ob, ac=action, metrics={"format": 1}, logs={}, reward=0, episode_done=True
    )
    traj = SimpleNamespace(transitions=[transition], final_ob=ob)
    return TrajectoryGroup(
        trajectories_G=[traj] * len(rewards), metrics_G=[{}] * len(rewards), final_rewards_G=rewards
    )


@pytest.mark.parametrize(
    ("builders", "max_rounds", "attempted"), [(list(range(30)), 2, 8), ([0, 1], 8, 2)]
)
def test_cap_or_dataset_end_never_trains_a_tail(builders, max_rounds, attempted):
    async def wave(items):
        return [group([0.5, 0.5]) for _ in items]

    with pytest.raises(RuntimeError, match=f"{attempted} prompts"):
        asyncio.run(sampling.collect_batch(builders, 0, 4, 2, max_rounds, wave, lambda *args: None))


def test_surplus_and_partial_rewards():
    async def wave(items):
        return [group([0.1, 0.1] if i < 3 else [0.1, 0.12]) for i in items]

    seen = []
    b = asyncio.run(
        sampling.collect_batch(
            list(range(12)), 0, 4, 2, 3, wave, lambda _, __, pos, reason: seen.append((pos, reason))
        )
    )
    assert b.builders == [3, 4, 5, 6]
    assert (b.offset, b.constant, b.surplus, b.rounds) == (8, 3, 1, 2)
    assert seen[-1] == (7, "surplus")


def test_run_resume_and_fresh_formal_run(tmp_path, monkeypatch):
    parent = parent_run(tmp_path)
    root_contents = (parent / "checkpoints.jsonl").read_bytes()
    loaded, updates = [], []
    positions = {}
    archives = {PARENT_URI: 100}

    class Builder:
        def __init__(self, n):
            self.datum = {"id": str(n)}

        def logging_tags(self):
            return ["nq"]

    async def make_dataset(self):
        return SearchRLDataset([Builder(i) for i in range(7400)], 64), None

    async def rollout(sampler, builder, **kwargs):
        n = int(builder.datum["id"])
        positions.setdefault(sampler, []).append(n)
        return group([0] * 8 if n % 2 else [0.1, 0.12] * 4)

    class Client:
        def __init__(self, step, name):
            self.step, self.name = step, name

        async def forward_backward(self, data, loss_fn, loss_fn_config):
            assert len(data) == 64 * 8
            assert loss_fn == "importance_sampling"
            assert all(abs(d.loss_fn_inputs["advantages"].tolist()[-1]) > 0 for d in data)
            return SimpleNamespace(
                loss_fn_outputs=[{"logprobs": d.loss_fn_inputs["logprobs"]} for d in data]
            )

        async def optim_step(self, params):
            assert params.learning_rate == 4e-5
            self.step += 1
            updates.append(self.step)
            return SimpleNamespace(metrics={})

        async def save_weights_and_get_sampling_client(self, **kwargs):
            return self.step

        async def save_state(self, name, **kwargs):
            uri = f"tinker://{self.name}/weights/{name}"
            archives[uri] = self.step
            return SimpleNamespace(path=uri)

        async def save_weights_for_sampler(self, name, **kwargs):
            return SimpleNamespace(path=f"tinker://{self.name}/{name}")

    class Service:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        async def create_training_client_from_state_with_optimizer(self, uri, **kwargs):
            loaded.append(uri)
            return Client(archives[uri], f"model_{len(loaded)}")

    monkeypatch.setattr(dapo, "patch_create_from_state", lambda *args: None)
    monkeypatch.setattr(dapo.tinker, "ServiceClient", Service)
    monkeypatch.setattr(
        dapo.checkpoint_utils, "check_renderer_name_for_checkpoint_async", AsyncMock()
    )
    monkeypatch.setattr(dapo, "fetch_model_pricing", lambda model: None)
    monkeypatch.setattr(dapo.SearchR1DatasetBuilder, "__call__", make_dataset)
    monkeypatch.setattr(sampling, "sample_group", rollout)

    smoke = tmp_path / "s3-smoke"
    formal = tmp_path / "s3"
    for child, target in [(smoke, 102), (smoke, 103), (formal, 101)]:
        config, resume, offset = dapo.load_run(parent, child, target, 1, 8)
        asyncio.run(dapo.run(config, resume, offset))
        ops.verify(parent, child, target)

    assert loaded == [PARENT_URI, "tinker://model_1/weights/final", PARENT_URI]
    assert updates == [101, 102, 103, 101]
    assert positions[100] == list(range(6400, 6528)) * 2
    assert positions[101] == list(range(6528, 6656))
    assert positions[102] == list(range(6656, 6784))
    assert (parent / "checkpoints.jsonl").read_bytes() == root_contents
    ckpts = list(ops.records(smoke / "checkpoints.jsonl"))
    assert [(r["batch"], r["name"]) for r in ckpts] == [
        (101, "000101"),
        (102, "final"),
        (103, "final"),
    ]
    assert ckpts[-1]["discarded_tokens"]["sampled"] == 3 * 64 * 8
    args = SimpleNamespace(log_path=smoke, eval_root=tmp_path, run_name=smoke.name, base_eval=None)
    for _, _, metrics, _ in monitor.collect(args, {"rollout": {}, "dev": {}}):
        assert metrics["rollout/episodes"] == 1024 and metrics["selected_rollout/episodes"] == 512


def test_cleanup_only_unlinks_listed_periodic_files(tmp_path):
    parent = parent_run(tmp_path)
    base = tmp_path / "archives"
    steps = [25, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    rows = [
        {"batch": s, "name": f"{s:06d}", "state_path": f"tinker://model_old/weights/{s:06d}"}
        for s in steps
    ]
    rows.append({"batch": 100, "name": "final", "state_path": PARENT_URI})
    write_jsonl(parent / "checkpoints.jsonl", rows)
    for row in rows:
        path = ops.state_file(base, row["state_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    ops.cleanup(parent, base, False)
    assert len(list(base.glob("*/*.tar.gz"))) == len(rows)
    ops.cleanup(parent, base, True)
    ops.cleanup(parent, base, True)
    for row in rows:
        assert ops.state_file(base, row["state_path"]).exists() == (
            row["batch"] not in ops.CLEANUP_STEPS
        )


def test_foreign_child_index_rejected(tmp_path):
    parent = parent_run(tmp_path)
    child = tmp_path / "s3"
    child.mkdir()
    write_jsonl(child / "checkpoints.jsonl", list(ops.records(parent / "checkpoints.jsonl")))
    with pytest.raises(ValueError):
        ops.own_records(child, PARENT_URI)
    with pytest.raises(ValueError):
        dapo.load_run(parent, child, 112, 5, 8)


def test_disk_plan_uses_real_free_space_and_existing_saves(tmp_path, monkeypatch):
    parent = parent_run(tmp_path)
    child = tmp_path / "s3"
    root = tmp_path / "ckpts"
    p = ops.state_file(root, PARENT_URI)
    p.parent.mkdir(parents=True)
    with p.open("wb") as f:
        f.truncate(1024)
    # GiB is scaled down for this test, keeping every estimate meaningful.
    monkeypatch.setattr(ops, "GIB", 1024)
    monkeypatch.setattr(ops.shutil, "disk_usage", lambda _: SimpleNamespace(free=18 * 1024))
    assert not ops.disk_plan(parent, child, root, [112, 124], 5)
    assert ops.disk_plan(parent, child, root, [112, 124], 0)
    assert not ops.disk_plan(parent, child, root, [112, 124], 0, include_smoke=True)
    with pytest.raises(ValueError):
        ops.disk_plan(parent, parent, root, [112, 124], 0)
