"""Local process simulations; never contact services or execute cloud commands."""

import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / "cloud"


def executable(path, body):
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def shell_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    bin_dir = workspace / "bin"
    bin_dir.mkdir(parents=True)
    (workspace / "logs").mkdir()
    (workspace / "SkyRL").mkdir()
    (bin_dir / "env_common.sh").write_text(
        f'export SKYRL="{workspace}/SkyRL"\n'
        f'export PROJ="{workspace}/project"\n'
        f'export RAY_TMPDIR="{workspace}/ray_tmp"\n'
        "export TINKER_BASE_URL=http://invalid RETRIEVAL_URL=http://invalid\n"
    )
    # Replace all machine-affecting commands, even though the script copy also
    # rewrites absolute paths into this isolated temporary directory.
    for name in ("tmux", "pkill", "sleep", "rm", "nvidia-smi", "df", "uv"):
        executable(bin_dir / name, "exit 0")
    return workspace, {"PATH": f"{bin_dir}:/usr/bin:/bin"}


def run_shell_copy(name, workspace, env):
    script = workspace / name
    script.write_text(
        (CLOUD / name)
        .read_text()
        .replace("/workspace", str(workspace))
        .replace("/tmp/ray", str(workspace / "tmp_ray"))
    )
    return subprocess.run(
        ["/bin/bash", str(script), "simulation"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("healthy", [True, False])
def test_restart_requires_health_response(shell_workspace, healthy):
    workspace, env = shell_workspace
    executable(workspace / "bin/curl", f"exit {0 if healthy else 1}")
    # A stale startup log must not satisfy readiness.
    (workspace / "logs/simulation.log").write_text("Uvicorn running\n")
    result = run_shell_copy("restart_skyrl.sh", workspace, env | {"HEALTH_TIMEOUT_SECONDS": "1"})
    assert result.returncode == (0 if healthy else 1), result.stderr
    assert ("server up:" in result.stdout) == healthy
    assert ("timed out" in result.stderr) == (not healthy)


@pytest.mark.parametrize("phase", ["training", "eval"])
def test_driver_stops_when_restart_fails(shell_workspace, phase):
    workspace, env = shell_workspace
    executable(workspace / "bin/curl", "exit 0")
    executable(workspace / "bin/restart_skyrl.sh", "exit 1")
    forbidden = workspace / "unexpected_uv"
    executable(workspace / "bin/uv", f'touch "{forbidden}"; exit 99')
    if phase == "eval":
        run = workspace / "runs/train/simulation"
        run.mkdir(parents=True)
        (run / "checkpoints.jsonl").write_text(
            json.dumps({"batch": 1, "state_path": "tinker://model/weights/final"}) + "\n"
        )
    result = run_shell_copy(
        "train_with_dev_eval.sh",
        workspace,
        env | {"TOTAL_STEPS": "1", "EVAL_EVERY": "1", "UPLOAD": "0"},
    )
    assert result.returncode == 1, result.stderr
    assert f"server restart failed before {phase}" in result.stdout
    assert not forbidden.exists()


def test_driver_rerun_uses_distinct_upload_logs(shell_workspace):
    workspace, env = shell_workspace
    executable(workspace / "bin/curl", "exit 0")
    executable(workspace / "bin/setsid", "echo simulated-upload")
    run = workspace / "runs/train/simulation"
    run.mkdir(parents=True)
    (run / "checkpoints.jsonl").write_text(
        json.dumps({"batch": 1, "state_path": "tinker://model/weights/final"}) + "\n"
    )
    evaluation = workspace / "project/runs/eval/simulation-step0001"
    evaluation.mkdir(parents=True)
    (evaluation / "summary.json").write_text(
        json.dumps({"run": {"num_errors": 0}, "by_source": {"OVERALL": {"n": 1}}})
    )
    env |= {"TOTAL_STEPS": "1", "EVAL_EVERY": "1", "PRUNE_STATES": "0"}
    for attempt in range(2):
        result = run_shell_copy("train_with_dev_eval.sh", workspace, env)
        # The simulated uploader deliberately leaves no completion marker.
        assert result.returncode == 1, result.stderr
        assert "W&B upload missing" in result.stdout
        logs = list((workspace / "logs").glob("simulation_upload_step0001.log.*"))
        assert len(logs) == attempt + 1
        if attempt == 0:
            first_log = logs[0]
            first_log.write_text("previous invocation\n")
    assert first_log.read_text() == "previous invocation\n"


def upload_process(root, entered, attempting):
    """Run the real locking/marker path with a blocked fake upload."""
    from scripts.cloud import upload_checkpoint as module

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


def test_stage3_driver_targets_and_upload_boundaries(shell_workspace):
    workspace, env = shell_workspace
    bin_dir = workspace / "bin"
    project = workspace / "project"
    cloud = project / "scripts/cloud"
    cloud.mkdir(parents=True)
    (cloud / "stage3_ops.py").write_text("pass\n")
    (project / ".venv/bin").mkdir(parents=True)
    executable(project / ".venv/bin/python", "exit 0")
    executable(bin_dir / "restart_skyrl.sh", "exit 0")
    executable(bin_dir / "curl", "exit 0")
    executable(bin_dir / "timeout", "exit 0")
    run = workspace / "runs/train/simulation"
    run.mkdir(parents=True)
    calls = workspace / "train_args"
    executable(
        bin_dir / "uv",
        f'''echo "$*" >> "{calls}"
if [[ "$*" == *search_r1.train_dapo* ]]; then
  step=$(echo "$*" | sed -E 's/.*--target ([0-9]+).*/\\1/')
  echo "{{\\"batch\\":$step,\\"state_path\\":\\"tinker://model/weights/final\\"}}" >> "{run}/checkpoints.jsonl"
elif [[ "$*" == *search_r1.offline_eval* ]]; then
  evaldir="{project}/runs/eval/${{@: -1}}"
  evaldir=${{evaldir/run_name=/}}
  mkdir -p "$evaldir"
  echo '{{"run":{{"num_errors":0}},"by_source":{{"OVERALL":{{"n":1,"em":1,"f1":1,"answer_rate":1}}}}}}' > "$evaldir/summary.json"
fi
exit 0''',
    )
    uploads = workspace / "upload_args"
    executable(
        bin_dir / "setsid",
        f'''echo "$*" >> "{uploads}"
step=$(echo "$*" | sed -E 's/.*--step ([0-9]+).*/\\1/')
mkdir -p "{run}/uploads"
touch "{run}/uploads/$(printf 'step%04d' "$step").json"''',
    )
    result = run_shell_copy("train_with_dev_eval.sh", workspace, env | {"STAGE3_PARENT": "parent"})
    assert result.returncode == 0, result.stdout + result.stderr
    lines = calls.read_text().splitlines()
    assert len([s for s in lines if "search_r1.train_dapo" in s]) == 2
    assert len([s for s in lines if "search_r1.offline_eval" in s]) == 2
    for lo, hi in ((100, 112), (112, 124)):
        assert any(f"--target {hi} " in s for s in lines)
        assert f"--step {hi} --prev-step {lo} " in uploads.read_text()
