# Script guide

## Data and evaluation

| Script | Purpose |
|---|---|
| [make_eval_sets.py](make_eval_sets.py) | Build the fixed development/test JSONL files |
| [run_base_eval.sh](run_base_eval.sh) | Run local oMLX model/prompt comparisons; starts retrieval if needed |
| [compare_runs.py](compare_runs.py) | Read saved evaluation summaries, optionally by source |
| [diagnose_run.py](diagnose_run.py) | Analyze saved transcripts, termination and gold-string matches |
| [check_render.py](check_render.py) | Inspect model prompt formatting with a sample tool schema |

Analysis scripts accept an archived evaluation directory under `experiments/`
as well as a working directory under `runs/eval/`.

## Cloud layout and deployment

| Task on the Pod | Command from the project root |
|---|---|
| Train with periodic dev evaluation | `scripts/cloud/train_with_dev_eval.sh RUN [key=value ...]` |
| Evaluate one checkpoint or the base model | `scripts/cloud/eval_checkpoint.sh RUN STEP_OR_BASE SPLIT` |
| Stop training | `scripts/cloud/stop_train.sh` |

| Directory | Tools |
|---|---|
| [setup/](cloud/setup/) | [Install project dependencies](cloud/setup/proj_install.sh), [download data and the BM25 index](cloud/setup/download_data.sh), and [install SkyRL](cloud/setup/skyrl_install.sh) with the [local patch](cloud/setup/apply_patch.py) |
| [services/](cloud/services/) | [Start SkyRL](cloud/services/start_skyrl.sh) or [restart it and wait for API readiness](cloud/services/restart_skyrl.sh) |
| [workflows/](cloud/workflows/) | Drive segmented training, standalone evaluation and short-run validation |
| [checkpoints/](cloud/checkpoints/) | [Prune full states](cloud/checkpoints/ckpt_prune.py) and [upload selected checkpoints](cloud/checkpoints/upload_checkpoint.py) |
| [monitoring/](cloud/monitoring/) | [Collect resource samples](cloud/monitoring/sysmon.sh), [summarize a time window](cloud/monitoring/stats_window.py), and [log training and dev metrics to W&B](cloud/monitoring/log_monitor_metrics.py) |
| [validation/](cloud/validation/) | [Verify short runs and restarts](cloud/validation/run_train_test.sh) and [compare evaluation concurrency settings](cloud/validation/conc_test.sh) |

### Pod setup

Prepare the project checkout on the Pod. From the project root, run
`bash scripts/cloud/setup/proj_install.sh` to install the locked client and the
independent retrieval environment. SkyRL uses a third environment.

By default the scripts use [env_common.sh](cloud/env_common.sh) to read credentials from `/workspace/.secrets/env.sh`,
use `/workspace/search-r1/SkyRL` for the server checkout and write logs under
`/workspace/logs`. Set `POD_ROOT` to change the workspace root or `POD_ENV_FILE`
to choose a credentials file. The Pod can also set `SKYRL`, `CKPT_BASE`,
`RETRIEVAL_URL`, `SEARCH_R1_DATA_DIR`, `TINKER_BASE_URL` and `TINKER_API_KEY`.
Start BM25 with the chosen data directory (`DATA_DIR` for `retrieval/start.sh`).

The SkyRL patch adds padding trimming and length-based gradient checkpointing. Deployment
defaults to `GC_SHORT_SEQ_THRESHOLD=8000`: for plain text microbatches of size 1,
checkpointing is disabled when the **post-trim input tensor width** is at most
8,000 tokens and enabled above it. Larger microbatches and unsupported layouts
retain checkpointing. This counts context and actions, not only generated tokens.

Keep `trainer.gradient_checkpointing=true` and
`trainer.gradient_checkpointing_use_reentrant=false`.
Set `GC_SHORT_SEQ_THRESHOLD=0` to keep checkpointing always on.

## Training and artifact management

Run the segmented driver in its own session so it can restart the `skyrl`
session. It accepts a run name followed by training CLI overrides. The four
recipes below use 64 questions per batch and eight trajectories per question.

```bash
BATCH_SIZE=64 GROUP_SIZE=8 scripts/cloud/train_with_dev_eval.sh ordinary
BATCH_SIZE=64 GROUP_SIZE=8 scripts/cloud/train_with_dev_eval.sh limited limit_search=true
BATCH_SIZE=64 GROUP_SIZE=8 scripts/cloud/train_with_dev_eval.sh unanswered limit_search=true max_turns_unanswered_reward=-0.3
SAMPLING_MODE=dynamic BATCH_SIZE=64 GROUP_SIZE=8 scripts/cloud/train_with_dev_eval.sh dynamic limit_search=true
```

New runs default to ordinary sampling, `TOTAL_STEPS=100`, `SAVE_EVERY=5`,
`EVAL_EVERY=25` and `EVAL_CONCURRENCY=32`. By default the driver prunes
eligible child states and uploads artifacts to W&B. `PRUNE_STATES=0` and
`UPLOAD=0` disable those operations independently. Automatic pruning preserves
parent checkpoints.

For example, fork from step 25 of the `ordinary` run above, a step it keeps after
pruning, and stop at global step 40:

```bash
PARENT_RUN=ordinary PARENT_STEP=25 TOTAL_STEPS=40 EVAL_EVERY=10 SAMPLING_MODE=dynamic \
    scripts/cloud/train_with_dev_eval.sh refill limit_search=true
```

The driver evaluates steps 35 and 40. Rerun with the same name, interval and end
step to finish missing evaluations or uploads. Training resumes at the latest
complete child state.

Standalone evaluation takes a run name or absolute directory, a step or `base`,
and `dev` or `test`. By default it uses the run's training configuration.

```bash
scripts/cloud/eval_checkpoint.sh refill 35 dev
scripts/cloud/eval_checkpoint.sh refill 40 test
scripts/cloud/eval_checkpoint.sh ordinary base dev
```

Evaluation outputs are saved under `runs/eval/`.

After each dev evaluation, the driver runs `log_monitor_metrics.py` to log the
dev result and per-step rollout summaries to a separate W&B run
`<run_name>-monitor` in the training run's group. Use `--dry-run` to preview the
metrics without uploading. To run it manually on the Pod:

```bash
uv run --frozen python scripts/cloud/monitoring/log_monitor_metrics.py --run-name RUN_NAME \
    --log-path /workspace/runs/train/RUN_NAME --eval-root runs/eval \
    --dev-summary 12 runs/eval/RUN_NAME-step0012-dev-DIGEST/summary.json
```

### W&B projects

The driver forwards training options only to the training CLI. The training run
follows `wandb_project`. The monitor run and the checkpoint artifacts use the
`--project` option of `log_monitor_metrics.py` and `upload_checkpoint.py`, which
both default to `search-r1-reproduce-experiment`. `wandb_project=None` therefore
turns off only the training run's logging; `UPLOAD=0` disables artifact uploads.

## Short validation and analysis tools

- [cloud/validation/run_train_test.sh](cloud/validation/run_train_test.sh): short named training jobs,
  `run_train_test.sh NAME BATCH_SIZE MAX_STEPS [key=value ...]`. A fresh short run defaults to `behavior_if_log_dir_exists=delete`, so use a distinct name.
  With `RESTART_AT`, the default becomes `resume`. For example,
  `SAMPLING_MODE=dynamic RESTART_AT=2 scripts/cloud/validation/run_train_test.sh short 4 3 limit_search=true`
  saves at step 2, restarts the service, then verifies the complete step-3 state.
- [cloud/validation/conc_test.sh](cloud/validation/conc_test.sh): evaluation concurrency comparisons.
- [cloud/monitoring/stats_window.py](cloud/monitoring/stats_window.py): summarize a time window
  from Pod runtime logs (`/workspace/logs/sysmon.csv` and vLLM logs).
