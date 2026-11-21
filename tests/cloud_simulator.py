"""Process double for Pod uv / project Python. Never starts a service or opens a socket."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["POD_ROOT"])
PROJECT = Path(os.environ["PROJ"])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(value) + "\n")


def training(module, args):
    run = Path(args["log_path"])
    cfg_file = run / "config.json"
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    parent = args.get("parent_run")
    step = int(args["max_steps"])
    dynamic = module.endswith("dynamic_sampling")
    batch, group = int(args["batch_size"]), int(args["group_size"])
    index = run / "checkpoints.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
    full = [r for r in rows if r.get("state_path")]
    start = cfg.get("parent_checkpoint", {})
    if parent and not start:
        source = Path(parent)
        parent_rows = [
            json.loads(line) for line in (source / "checkpoints.jsonl").read_text().splitlines()
        ]
        start = [
            r for r in parent_rows if r["batch"] == int(args["parent_step"]) and r.get("state_path")
        ][-1]
        cfg.update(
            parent_step=start["batch"],
            parent_run=parent,
            parent_state_path=start["state_path"],
            parent_checkpoint=start,
        )
        if not dynamic:
            append(index, start | {"owner": "parent", "parent_state_path": start["state_path"]})
    cfg.update(
        dataset_builder={"batch_size": batch, "group_size": group},
        sampling_mode="dynamic" if dynamic else "ordinary",
    )
    write(cfg_file, cfg)
    last = full[-1] if full else start
    offset = last.get("next_prompt_offset", 0)
    for n in range(last.get("batch", 0), step):
        if dynamic:
            for pos in range(offset, offset + batch):
                for member in range(group):
                    append(
                        run / f"iteration_{n:06d}/train_rollout_summaries.jsonl",
                        {
                            "prompt_offset": pos,
                            "selection": "selected",
                            "total_reward": member,
                            "sampling_client_step": n,
                        },
                    )
            append(
                run / "metrics.jsonl",
                {
                    "step": n,
                    "progress/completed_updates": n + 1,
                    "optim/optimizer_steps": 1,
                    "optim/kl_sample_train_v2": 0,
                    "optim/entropy": 1,
                    "sampling/start_prompt_offset": offset,
                    "sampling/next_prompt_offset": offset + batch,
                    "sampling/selected_groups": batch,
                    "sampling/attempted_groups": batch,
                    "sampling/constant_groups": 0,
                    "sampling/surplus_groups": 0,
                },
            )
        offset += batch
    if os.environ.get("NO_SAVE"):
        append(index, {"batch": step, "sampler_path": "sampler-only"})
        return
    uri = f"tinker://model_{step}/weights/final"
    fields = {"parent_state_path": start["state_path"]} if start else {}
    if dynamic and start:
        fields["stage3_run"] = run.name
    append(
        index,
        fields
        | {
            "batch": step,
            "owner": "child",
            "state_path": uri,
            "name": "final",
            "next_prompt_offset": offset,
        },
    )
    write(Path(os.environ["CKPT_BASE"]) / f"model_{step}/final.tar.gz", "fake archive")
    if os.environ.get("FAIL_TRAIN"):
        sys.exit(7)


def evaluate(args):
    output = PROJECT / "runs/eval" / args["run_name"]
    data = Path(args["eval_file"])
    size = len(data.read_text().splitlines())
    write(
        output / "config.json",
        args
        | {
            "num_items": size,
            "training_config": json.loads(Path(args["training_config_file"]).read_text())
            if "training_config_file" in args
            else None,
        },
    )
    write(
        output / "summary.json",
        {
            "by_source": {"OVERALL": {"n": size - int(os.environ.get("SHORT_EVAL", 0))}},
            "run": {"num_errors": int(os.environ.get("EVAL_ERRORS", 0))},
        },
    )
    sys.exit(int(os.environ.get("EVAL_RC", 0)))


def main():
    args = sys.argv[1:]
    if args and args[0].endswith(".py"):
        assert Path(args[0]).is_file(), args[0]
    append(ROOT / "calls.jsonl", args)
    if "-m" in args and args[args.index("-m") + 1].startswith("search_r1."):
        pos = args.index("-m") + 1
        options = dict(arg.split("=", 1) for arg in args[pos + 1 :])
        if args[pos] == "search_r1.offline_eval":
            evaluate(options)
        else:
            training(args[pos], options)
    elif args and args[0].endswith("upload_checkpoint.py"):
        if not os.environ.get("SKIP_MARKER"):
            step = int(args[args.index("--step") + 1])
            markers = Path(args[args.index("--marker-dir") + 1])
            write(markers / f"step{step:04d}.json", {"artifact": f"simulation:{step}"})
    elif not ("ray" in args or (args and args[0].endswith("log_monitor_metrics.py"))):
        raise AssertionError(args)


if __name__ == "__main__":
    main()
