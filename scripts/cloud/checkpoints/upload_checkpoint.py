"""Upload one eval step's checkpoint to W&B Artifacts, for keeping it after the pod is gone.

For eval step T of a run the artifact holds:
- the full training state saved at step T (base weights, LoRA and optimizer; loadable with
  offline_eval.py tinker_state_url=... or for resuming, after copying it back under
  --checkpoints-base);
- the sampler (LoRA adapter) weights of every checkpoint in (previous eval step, T];
- the dev-set eval summary/config (including its training-config snapshot) and checkpoint records.

It is one version of the artifact ``<run_name>-checkpoints`` (type ``model``) with alias
``step-<T>``, logged from its own short W&B run (job_type ``checkpoint-upload``, grouped with the
training run) so it never touches the training run while that is being written.

Files are added with policy="immutable" and skip_cache=True: no staging copy of the 10.5 GB state.
A marker file is written on success; rerunning for a step that has one does nothing.
Concurrent invocations for the same run/step wait on a file lock and recheck that marker.

usage: upload_checkpoint.py --run-name NAME --log-path DIR --step T --prev-step P
           --eval-summary FILE --ckpt-base DIR --marker-dir DIR [--project P] [--dry-run]
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.cloud.checkpoints.ckpt_prune import state_file
from search_r1.checkpoint_index import index_bytes

RETRIES = 3


def sampler_file(ckpt_base: Path, sampler_path: str) -> Path:
    """tinker://<model>/<name> -> <ckpt_base>/<model>/sampler_weights/<name>.tar.gz"""
    model, sep, name = sampler_path.removeprefix("tinker://").partition("/")
    name = name.removeprefix("sampler_weights/")
    if not sampler_path.startswith("tinker://") or not sep:
        raise ValueError(f"Unexpected sampler path: {sampler_path}")
    # Validate the same model and archive name used by full-state exports.
    state_file(ckpt_base, f"tinker://{model}/weights/{name}")
    result = (ckpt_base / model / "sampler_weights" / f"{name}.tar.gz").resolve()
    if not result.is_relative_to(ckpt_base.resolve()):
        raise ValueError(f"Sampler archive outside checkpoint root: {result}")
    return result


def collect(args: argparse.Namespace) -> tuple[list[tuple[Path, str]], dict, dict]:
    """Files to upload as (local path, path inside the artifact), the metadata, and the
    training run's W&B identity (project/entity/id, empty if unknown)."""
    records = [
        json.loads(line)
        for line in index_bytes(args.log_path / "checkpoints.jsonl").splitlines()
        if line.strip()
    ]
    canonical: dict[int, dict] = {}
    full = {}
    for r in records:  # later records win, as in the driver and the cookbook
        if type(r.get("batch")) is not int or r.get("owner") == "parent":
            continue
        canonical[r["batch"]] = r
        if isinstance(r.get("state_path"), str) and r["state_path"].strip():
            full[r["batch"]] = r
    canonical.update(full)
    if args.step not in full:
        raise SystemExit(f"no checkpoint with a full state at step {args.step}")
    at_step = canonical[args.step]

    files = [
        (state_file(args.ckpt_base, at_step["state_path"]), f"state/step{args.step:04d}.tar.gz")
    ]
    for batch in sorted(b for b in canonical if args.prev_step < b <= args.step):
        r = canonical[batch]
        if r.get("sampler_path"):
            files.append(
                (sampler_file(args.ckpt_base, r["sampler_path"]), f"sampler/step{batch:04d}.tar.gz")
            )
    files.append((args.eval_summary, f"eval/dev_step{args.step:04d}_summary.json"))
    eval_config_file = args.eval_summary.with_name("config.json")
    files.append((eval_config_file, f"eval/dev_step{args.step:04d}_config.json"))
    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))

    summary = json.loads(args.eval_summary.read_text())["by_source"]["OVERALL"]
    eval_config = json.loads(eval_config_file.read_text())
    if eval_config.get("tinker_state_url") != at_step["state_path"]:
        raise ValueError("Evaluation snapshot and upload state differ")
    identity_file = args.log_path / "wandb_run.json"
    identity = json.loads(identity_file.read_text()) if identity_file.exists() else {}
    skyrl = (
        os.environ.get("SKYRL")
        or Path(os.environ.get("POD_ROOT", "/workspace")) / "search-r1/SkyRL"
    )
    try:
        skyrl_commit = subprocess.run(
            ["git", "-C", skyrl, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        skyrl_commit = "unknown"
    metadata = {
        "run_name": args.run_name,
        "step": args.step,
        "state_path": at_step["state_path"],
        "sampler_path": at_step.get("sampler_path"),
        "checkpoint_records": [
            canonical[b] for b in sorted(canonical) if args.prev_step < b <= args.step
        ],
        "dev_eval": {
            k: summary.get(k) for k in ("n", "em", "f1", "answer_rate", "em_when_answered")
        },
        "training_wandb_run": identity.get("id"),
        "skyrl_commit": skyrl_commit,
        "skyrl_patch": "scripts/cloud/setup/apply_patch.py",
        "base_model": eval_config["base_model"],
        "lora_rank": eval_config["lora_rank"],
    }
    return files, metadata, identity


def upload(args, files, metadata, identity) -> dict:
    # Uploads are separate processes and must never resume the training history.
    for key in ("WANDB_RUN_ID", "WANDB_RESUME"):
        os.environ.pop(key, None)
    import wandb

    volume = Path(os.environ.get("POD_ROOT", "/workspace"))
    os.environ.setdefault("WANDB_DIR", str(volume / "wandb_upload"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(volume / ".cache/wandb"))
    os.environ.setdefault("WANDB_DATA_DIR", str(volume / ".cache/wandb-data"))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)

    try:  # wandb >= 0.2x moved generate_id out of wandb.util
        from wandb.sdk.lib.runid import generate_id
    except ImportError:
        generate_id = wandb.util.generate_id

    run = wandb.init(
        id=generate_id(),
        resume="never",
        project=identity.get("project", args.project),
        entity=identity.get("entity"),
        group=args.run_name,
        job_type="checkpoint-upload",
        name=f"{args.run_name}-upload-step{args.step:04d}",
        config={"step": args.step, "training_wandb_run": metadata["training_wandb_run"]},
    )
    try:
        artifact = wandb.Artifact(f"{args.run_name}-checkpoints", type="model", metadata=metadata)
        for local, name in files:
            artifact.add_file(str(local), name=name, policy="immutable", skip_cache=True)
        run.log_artifact(artifact, aliases=[f"step-{args.step}"])
        artifact.wait()
        info = {"artifact": artifact.qualified_name, "version": artifact.version, "id": artifact.id}
    finally:
        run.finish()
    return info


def marker_matches(marker, summary, previous_step):
    try:
        info = json.loads(marker.read_bytes())
        return (
            isinstance(info, dict)
            and bool(info.get("artifact"))
            and (
                "eval_directory" not in info  # Historical completed uploads.
                or (
                    info["eval_directory"] == summary.parent.name
                    and info["previous_step"] == previous_step
                )
            )
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False


def upload_step(args: argparse.Namespace) -> int:
    """Check the marker and upload; the caller holds the step lock unless dry-running."""
    marker = args.marker_dir / f"step{args.step:04d}.json"
    if marker_matches(marker, args.eval_summary, args.prev_step):
        print(f"already uploaded: {marker.read_text().strip()}")
        return 0

    files, metadata, identity = collect(args)
    total = sum(p.stat().st_size for p, _ in files)
    for local, name in files:
        print(f"{local.stat().st_size / 1e9:8.2f} GB  {name}  <- {local}")
    print(f"{total / 1e9:8.2f} GB  total, artifact {args.run_name}-checkpoints:step-{args.step}")
    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    for attempt in range(1, RETRIES + 1):
        start = time.monotonic()
        try:
            info = upload(args, files, metadata, identity)
        except Exception as e:  # network or service errors: retry, then give up loudly
            print(f"attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}", flush=True)
            if attempt == RETRIES:
                return 1
            time.sleep(60 * attempt)
            continue
        info.update(
            seconds=round(time.monotonic() - start),
            bytes=total,
            eval_directory=args.eval_summary.parent.name,
            previous_step=args.prev_step,
        )
        args.marker_dir.mkdir(parents=True, exist_ok=True)
        pending_marker = marker.with_suffix(".tmp")
        pending_marker.write_text(json.dumps(info) + "\n")
        pending_marker.replace(marker)
        print(f"uploaded in {info['seconds']} s: {info}")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--prev-step", type=int, required=True, help="previous eval step (0 for the first)"
    )
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--ckpt-base", type=Path, required=True)
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument("--project", default="search-r1-reproduce-experiment")
    parser.add_argument(
        "--dry-run", action="store_true", help="list files and metadata, upload nothing"
    )
    args = parser.parse_args()
    if args.dry_run:
        return upload_step(args)

    # marker_dir is scoped to the training run. Never unlink the lock file: other
    # processes may be waiting on its inode. The OS releases the lock on process exit.
    args.marker_dir.mkdir(parents=True, exist_ok=True)
    with (args.marker_dir / f"step{args.step:04d}.lock").open("a") as lock:
        print(f"waiting for upload lock: {lock.name}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        return upload_step(args)


if __name__ == "__main__":
    sys.exit(main())
