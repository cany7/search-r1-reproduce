"""Local process simulations; never contact services or execute cloud commands."""

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / "cloud"


def executable(path, body):
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


def upload_process(root, entered, attempting):
    """Run the real locking/marker path with a blocked fake upload."""
    from scripts.cloud.checkpoints import upload_checkpoint as module

    root = Path(root)
    original_flock = module.fcntl.flock

    def observed_flock(*args):
        attempting.set()
        return original_flock(*args)

    def fake_upload(*args):
        with (root / "upload_calls").open("a") as calls:
            calls.write(f"{os.getpid()}\n")
        entered.set()
        deadline = time.monotonic() + 10
        while not (root / "release").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return {"artifact": "fake:v0"}

    module.fcntl.flock = observed_flock
    module.collect = lambda args: ([(root / "payload", "state")], {}, {})
    module.upload = fake_upload
    sys.argv = [
        "upload_checkpoint.py",
        "--run-name",
        "simulation",
        "--log-path",
        str(root),
        "--step",
        "25",
        "--prev-step",
        "0",
        "--eval-summary",
        str(root / "summary"),
        "--ckpt-base",
        str(root),
        "--marker-dir",
        str(root / "uploads"),
    ]
    raise SystemExit(module.main())


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_upload_lock_rechecks_marker_and_releases_on_exit(tmp_path, interrupt_first):
    (tmp_path / "payload").write_text("fake checkpoint")
    ctx = multiprocessing.get_context("spawn")
    entered = ctx.Event()
    first_attempt, second_attempt = ctx.Event(), ctx.Event()
    first = ctx.Process(target=upload_process, args=(str(tmp_path), entered, first_attempt))
    second = ctx.Process(target=upload_process, args=(str(tmp_path), entered, second_attempt))
    first.start()
    try:
        assert entered.wait(10)
        second.start()
        assert second_attempt.wait(10)
        if interrupt_first:
            first.terminate()
            first.join(5)
        (tmp_path / "release").touch()
        first.join(5)
        second.join(5)
        assert second.exitcode == 0
        if not interrupt_first:
            assert first.exitcode == 0
        calls = (tmp_path / "upload_calls").read_text().splitlines()
        assert len(calls) == (2 if interrupt_first else 1)
        assert json.loads((tmp_path / "uploads/step0025.json").read_text())["artifact"] == "fake:v0"
        assert not (tmp_path / "uploads/step0025.tmp").exists()
    finally:
        (tmp_path / "release").touch()
        for process in (first, second):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)


@pytest.fixture
def pod(tmp_path):
    # Run the distributed shell files verbatim, from a different working directory.
    project = tmp_path / "synced project"
    cloud = project / "scripts/cloud"
    shutil.copytree(CLOUD, cloud, ignore=shutil.ignore_patterns("__pycache__"))
    (project / "search_r1").mkdir()
    shutil.copy(CLOUD.parents[1] / "search_r1/checkpoint_index.py", project / "search_r1")
    root = tmp_path / "pod volume"
    (root / ".secrets").mkdir(parents=True)
    (root / ".secrets/env.sh").write_text(
        "export RETRIEVAL_URL=http://invalid TINKER_BASE_URL=http://invalid\n"
    )
    (root / "search-r1/SkyRL").mkdir(parents=True)
    (root / "ckpts/skyrl").mkdir(parents=True)
    (project / "eval_sets").mkdir()
    for split, size in [("dev", 3), ("test", 5)]:
        (project / "eval_sets" / f"{split}.jsonl").write_text('{"data_source":"nq"}\n' * size)
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    for name in ("tmux", "pkill", "sleep", "rm", "nvidia-smi", "df", "curl"):
        executable(bin_dir / name, "exit 0")
    (bin_dir / "python3").symlink_to(sys.executable)
    fake = Path(__file__).with_name("cloud_simulator.py")
    for path in (bin_dir / "uv", project / ".venv/bin/python"):
        path.parent.mkdir(parents=True, exist_ok=True)
        executable(path, f'exec "{sys.executable}" "{fake}" "$@"')
    env = {"POD_ROOT": str(root), "PATH": f"{bin_dir}:/usr/bin:/bin", "HEALTH_TIMEOUT_SECONDS": "1"}
    # Model a Pod volume independently of the test machine's free space.
    # Python worker processes must also stay offline even if a double regresses.
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import shutil, socket, types\n"
        "shutil.disk_usage = lambda path: types.SimpleNamespace(free=128 * 1024**3)\n"
        "def forbidden(*args): raise AssertionError('network disabled in pod simulation')\n"
        "socket.socket.connect = socket.socket.connect_ex = forbidden\n"
    )
    env["PYTHONPATH"] = str(guard)
    return root, project, bin_dir, env


def shell(pod, name, args, **env):
    root, project, _, environment = pod
    return subprocess.run(
        ["bash", project / "scripts/cloud" / name, *map(str, args)],
        env=environment | env,
        cwd=root.parent,
        capture_output=True,
        text=True,
        timeout=15,
    )


def calls(root, match):
    path = root / "calls.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if any(match in word for word in row)]


def source_run(root):
    run = root / "runs/train/parent"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps({"dataset_builder": {"batch_size": 2, "group_size": 3, "seed": 31}})
    )
    rows = [
        {"batch": 7, "name": "early", "state_path": "tinker://p/weights/early"},
        {"batch": 10, "name": "newer", "state_path": "tinker://p/weights/newer"},
        {
            "batch": 7,
            "name": "chosen",
            "state_path": "tinker://p/weights/chosen",
            "next_prompt_offset": 14,
        },
        {"batch": 7, "sampler_path": "sampler-only"},
    ]
    (run / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = root / "ckpts/skyrl/p"
    files.mkdir()
    (files / "chosen.tar.gz").write_bytes(b"PARENT")
    return run


@pytest.mark.parametrize("healthy", [True, False])
def test_restart_requires_health_and_launches_synced_start_script(pod, healthy):
    root, project, bin_dir, _ = pod
    executable(bin_dir / "curl", f"exit {0 if healthy else 1}")
    executable(bin_dir / "tmux", f'echo "$*" >> "{root}/tmux-args"')
    result = shell(pod, "services/restart_skyrl.sh", ["simulation"])
    assert result.returncode == (0 if healthy else 1), result.stderr
    assert ("server up:" in result.stdout) == healthy
    command = (root / "tmux-args").read_text()
    assert "scripts/cloud/services/start_skyrl.sh" in command and "/workspace/bin" not in command


@pytest.mark.parametrize(
    "mode,start", [("ordinary", 0), ("ordinary", 7), ("dynamic", 7), ("dynamic", 0)]
)
def test_shell_schedule_and_recovery(pod, mode, start):
    root, project, _, _ = pod
    run = root / "runs/train/simulation"
    options = {
        "TOTAL_STEPS": str(start + 11),
        "EVAL_EVERY": "5",
        "SAMPLING_MODE": mode,
        "BATCH_SIZE": "2",
        "GROUP_SIZE": "3",
    }
    if start:
        source_run(root)
        options.update(PARENT_RUN="parent", PARENT_STEP=str(start))
    result = shell(
        pod,
        "train_with_dev_eval.sh",
        ["simulation", "limit_search=true", "learning_rate=0.0002"],
        **options,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "WARNING: monitor" not in result.stdout
    targets = [start + 5, start + 10, start + 11]
    train = calls(root, "search_r1.train")
    assert [
        next(int(s.split("=")[1]) for s in row if s.startswith("max_steps=")) for row in train
    ] == targets
    module = "search_r1.train_dynamic_sampling" if mode == "dynamic" else "search_r1.train"
    assert all(
        module in row and "learning_rate=0.0002" in row and "limit_search=true" in row
        for row in train
    )
    if start:
        assert all("seed=31" in row and f"parent_step={start}" in row for row in train)
        assert (root / "ckpts/skyrl/p/chosen.tar.gz").read_bytes() == b"PARENT"
    uploads = calls(root, "upload_checkpoint.py")
    assert sorted(
        (int(row[row.index("--prev-step") + 1]), int(row[row.index("--step") + 1]))
        for row in uploads
    ) == list(zip([start, *targets[:-1]], targets))
    evals = calls(root, "search_r1.offline_eval")
    assert len(evals) == 3
    # A completed child must recover its original cadence without a parent CLI argument.
    if start:
        (root / "runs/train/parent").rename(root / "archived")
        options.pop("PARENT_RUN")
        options.pop("PARENT_STEP")
    run_name = next(word[9:] for word in evals[0] if word.startswith("run_name="))
    (project / "runs/eval" / run_name / "summary.json").unlink()
    (run / "uploads" / f"step{targets[1]:04d}.json").unlink()
    result = shell(pod, "train_with_dev_eval.sh", ["simulation"], **options)
    assert result.returncode == 0, result.stderr
    assert len(calls(root, "search_r1.train")) == 3
    assert len(calls(root, "search_r1.offline_eval")) == 4
    assert len(calls(root, "upload_checkpoint.py")) == 4
    result = shell(pod, "train_with_dev_eval.sh", ["simulation"], **options)
    assert result.returncode == 0, result.stderr
    assert (
        len(calls(root, "search_r1.offline_eval")) == len(calls(root, "upload_checkpoint.py")) == 4
    )


@pytest.mark.parametrize("phase", ["training", "eval"])
def test_driver_stops_when_restart_fails(pod, phase):
    root, project, _, _ = pod
    if phase == "eval":
        run = source_run(root)
        name, target = run.name, "7"
    else:
        name, target = "simulation", "1"
    executable(project / "scripts/cloud/services/restart_skyrl.sh", "exit 1")
    result = shell(
        pod, "train_with_dev_eval.sh", [name], TOTAL_STEPS=target, EVAL_EVERY=target, UPLOAD="0"
    )
    assert result.returncode != 0
    assert f"server restart failed before {phase}" in result.stderr
    assert not calls(root, "search_r1.")


def test_missing_upload_retried_with_distinct_logs(pod):
    root, _, _, _ = pod
    for i in (1, 2):
        result = shell(
            pod, "train_with_dev_eval.sh", ["simulation"], TOTAL_STEPS="1", SKIP_MARKER="1"
        )
        assert result.returncode and "W&B upload missing" in result.stderr
        assert len(list((root / "logs").glob("simulation_upload_step0001.log.*"))) == i


def test_eval_distinguishes_weights_sets_snapshots_and_checks_exact_counts(pod):
    root, project, _, _ = pod
    run = source_run(root)
    alt = root / "different config.json"
    alt.write_text('{"dataset_builder":{"limit_search":true}}')
    base = [run, "base", "dev"]
    for args in (
        [run, 7, "dev"],
        [run, 7, "test"],
        [run, 7, "dev", f"training_config_file={alt}"],
        base,
    ):
        result = shell(pod, "eval_checkpoint.sh", args)
        assert result.returncode == 0, result.stderr
    assert len(list((project / "runs/eval").glob("*/.complete"))) == 4
    rows = calls(root, "search_r1.offline_eval")
    assert all("tinker_state_url=tinker://p/weights/chosen" in r for r in rows[:3])
    assert "skyrl_base_eval=true" in rows[-1] and not any(
        "tinker_state_url=" in s for s in rows[-1]
    )
    assert f"training_config_file={alt}" in rows[2]
    for env in ({"SHORT_EVAL": "1"}, {"EVAL_ERRORS": "1"}, {"EVAL_RC": "9"}):
        result = shell(pod, "eval_checkpoint.sh", [run, 10, "test"], **env)
        assert result.returncode != 0
    assert len(list((project / "runs/eval").glob("*/.complete"))) == 4
    assert shell(pod, "eval_checkpoint.sh", [run, 10, "test"]).returncode == 0
    assert shell(pod, "eval_checkpoint.sh", [run, 6, "test"]).returncode != 0


def test_smoke_restarts_and_verifies_dynamic_run(pod):
    root, _, _, _ = pod
    source_run(root)
    result = shell(
        pod,
        "validation/run_train_test.sh",
        ["short", 2, 10, "group_size=3"],
        SAMPLING_MODE="dynamic",
        RESTART_BEFORE_TRAIN="1",
        PARENT_RUN="parent",
        PARENT_STEP="7",
        RESTART_AT="9",
    )
    assert result.returncode == 0, result.stderr
    rows = calls(root, "search_r1.train_dynamic_sampling")
    assert len(rows) == 2 and "max_steps=9" in rows[0] and "max_steps=10" in rows[1]
    assert all("behavior_if_log_dir_exists=resume" in row for row in rows)
    assert (root / "logs/short_restart.log").exists()


@pytest.mark.parametrize("env", [{"NO_SAVE": "1"}, {"FAIL_TRAIN": "1"}])
def test_short_test_does_not_mask_trainer_or_missing_full_state(pod, env):
    root, _, _, _ = pod
    result = shell(pod, "validation/run_train_test.sh", ["short", 2, 1], **env)
    assert result.returncode != 0
    if "FAIL_TRAIN" in env:
        assert result.returncode == 7
    assert "failed" in (root / "logs/train_tests.times").read_text()
    assert "behavior_if_log_dir_exists=delete" in calls(root, "search_r1.train")[0]


def test_tmux_restarts_the_synced_script_with_quoted_config(pod):
    root, _, bin_dir, _ = pod
    # Execute the nested tmux command with the real start script and a fake uv.
    executable(bin_dir / "tmux", 'if [ "$1" = new ]; then bash -c "${@: -1}"; fi')
    executable(
        bin_dir / "uv",
        f'''if [[ "$*" == *skyrl.tinker.api* ]]; then printf '%s' "${{@: -1}}" > "{root}/server-config"; fi''',
    )
    result = shell(
        pod,
        "services/restart_skyrl.sh",
        ["quoted log"],
        EXTRA_CFG=json.dumps({"option": "it's spaced"})[1:-1],
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((root / "server-config").read_text())
    assert config["option"] == "it's spaced"
    assert config["trainer.ckpt_path"] == str(root / "ckpts/skyrl-train")


def test_installer_uses_adjacent_patch_without_network(pod):
    root, project, bin_dir, _ = pod
    repo = root / "search-r1/SkyRL"
    (repo / ".venv/bin").mkdir(parents=True)
    executable(repo / ".venv/bin/python", "exit 0")
    executable(bin_dir / "git", "exit 0")
    executable(bin_dir / "uv", "exit 0")
    # Replace the patcher in the copied checkout with a recording stand-in.
    (project / "scripts/cloud/setup/apply_patch.py").write_text(
        "import pathlib,sys\npathlib.Path(sys.argv[1], 'patch-path').write_text(__file__)\n"
    )
    result = shell(pod, "setup/skyrl_install.sh", [])
    assert result.returncode == 0, result.stderr
    assert (repo / "patch-path").read_text() == str(project / "scripts/cloud/setup/apply_patch.py")


def test_stop_selects_only_python_training_modules(pod):
    root, _, bin_dir, _ = pod
    executable(bin_dir / "pkill", f'printf "%s" "$2" > "{root}/pattern"')
    executable(bin_dir / "pgrep", "exit 1")
    assert shell(pod, "stop_train.sh", []).returncode == 0
    pattern = (root / "pattern").read_text()
    commands = [
        "python -m search_r1.train",
        "/opt/venv/bin/python3.12 -m search_r1.train_dynamic_sampling a=b",
        "python -m search_r1.offline_eval",
        "python -m search_r1.training.loop",
        "python -m search_r1.train_extra",
        "bash -c python -m search_r1.train",
    ]
    # Use POSIX ERE, the same dialect as pkill.
    result = subprocess.run(
        ["grep", "-E", pattern],
        input="\n".join(commands),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == commands[:2]


def test_base_before_first_training_snapshot(pod):
    root, _, _, _ = pod
    result = shell(
        pod, "eval_checkpoint.sh", ["baseline", "base", "test", "base_model=custom/model"]
    )
    assert result.returncode == 0, result.stderr
    args = calls(root, "search_r1.offline_eval")[0]
    assert "skyrl_base_eval=true" in args and "base_model=custom/model" in args
    assert not any(arg.startswith("training_config_file=") for arg in args)


def test_empty_dynamic_child_index_and_midsegment_resume_keep_fork_cadence(pod):
    root, _, _, _ = pod
    parent = source_run(root)
    run = root / "runs/train/partial"
    run.mkdir()
    row = json.loads((parent / "checkpoints.jsonl").read_text().splitlines()[2])
    cfg = {
        "dataset_builder": {"batch_size": 2, "group_size": 3},
        "sampling_mode": "dynamic",
        "parent_step": 7,
        "parent_run": str(parent),
        "parent_state_path": row["state_path"],
        "parent_checkpoint": row,
    }
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "checkpoints.jsonl").touch()
    # Simulate an already committed periodic save before the first scheduled eval.
    result = shell(
        pod,
        "validation/run_train_test.sh",
        ["partial", 2, 9, f"log_path={run}", "behavior_if_log_dir_exists=resume"],
    )
    assert result.returncode == 0, result.stderr
    result = shell(
        pod,
        "train_with_dev_eval.sh",
        ["partial"],
        SAMPLING_MODE="dynamic",
        TOTAL_STEPS="13",
        EVAL_EVERY="5",
    )
    assert result.returncode == 0, result.stderr
    assert "training 9 -> 12" in result.stdout
    uploads = calls(root, "upload_checkpoint.py")
    assert sorted(
        (int(r[r.index("--prev-step") + 1]), int(r[r.index("--step") + 1])) for r in uploads
    ) == [(7, 12), (12, 13)]


def test_real_upload_collection_uses_last_full_state_and_exact_interval(tmp_path):
    from types import SimpleNamespace

    from scripts.cloud.checkpoints import upload_checkpoint as upload

    rows = [
        {
            "batch": 7,
            "owner": "parent",
            "state_path": "tinker://p/weights/start",
            "sampler_path": "tinker://p/start",
        },
        {"batch": 10, "sampler_path": "tinker://c/sampler_weights/ten"},
        {"batch": 12, "state_path": "tinker://c/weights/earlier"},
        {
            "batch": 12,
            "state_path": "tinker://c/weights/chosen",
            "sampler_path": "tinker://c/twelve",
        },
        {"batch": 12, "sampler_path": "tinker://c/sampler-only"},
        {"batch": 17, "state_path": "tinker://c/weights/later"},
    ]
    (tmp_path / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    archive = tmp_path / "archives"
    (archive / "c/sampler_weights").mkdir(parents=True)
    for name in ("chosen", "sampler_weights/ten", "sampler_weights/twelve"):
        (archive / "c" / f"{name}.tar.gz").touch()
    summary = tmp_path / "eval/summary.json"
    summary.parent.mkdir()
    summary.write_text('{"by_source":{"OVERALL":{"n":3}}}')
    summary.with_name("config.json").write_text(
        json.dumps(
            {"tinker_state_url": rows[3]["state_path"], "base_model": "test", "lora_rank": 8}
        )
    )
    files, meta, _ = upload.collect(
        SimpleNamespace(
            log_path=tmp_path,
            prev_step=7,
            step=12,
            ckpt_base=archive,
            eval_summary=summary,
            run_name="test",
        )
    )
    assert meta["state_path"] == rows[3]["state_path"]
    assert [name for _, name in files[:3]] == [
        "state/step0012.tar.gz",
        "sampler/step0010.tar.gz",
        "sampler/step0012.tar.gz",
    ]
    assert len(files) == 5
