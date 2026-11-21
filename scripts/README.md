# Script guide

This directory contains the maintained project tools.
No scripts were run against model services during experiment archiving.

## Data and evaluation

| Script | Purpose |
|---|---|
| [make_eval_sets.py](make_eval_sets.py) | Build the fixed development/test JSONL files |
| [run_base_eval.sh](run_base_eval.sh) | Run local oMLX model/prompt comparisons; starts retrieval if needed |
| [compare_runs.py](compare_runs.py) | Read saved evaluation summaries, optionally by source |
| [diagnose_run.py](diagnose_run.py) | Analyze saved transcripts, termination and gold-string matches |
| [check_render.py](check_render.py) | Inspect upstream renderer prompts with a dummy tool schema |

`check_render.py` lists upstream renderer configurations; it does not validate the
project's Unicode subclass. The diagnostic script's reported retrieval position
counts tool-response messages, not assistant turns. A gold-string match does not
prove the documents establish the answer. See the
[baseline analysis](../experiments/local-baselines/BASELINE_ANALYSIS.md)
for these interpretation limits.

Analysis scripts accept an archived evaluation directory under `experiments/`
as well as a working directory under `runs/eval/`.

## Cloud layout and deployment

| Task on the Pod | Command from the project root |
|---|---|
| Train with periodic dev evaluation | `scripts/cloud/train_with_dev_eval.sh RUN [key=value ...]` |
| Evaluate one checkpoint or the base model | `scripts/cloud/eval_checkpoint.sh RUN STEP_OR_BASE SPLIT` |
| Stop training | `scripts/cloud/stop_train.sh` |
| Verify a short training run | `scripts/cloud/validation/run_train_test.sh NAME BATCH_SIZE MAX_STEPS [key=value ...]` |

For standalone evaluation, use a step number or `base`, and choose `dev` or `test`.
The training, evaluation and short-run commands call [driver.py](cloud/workflows/driver.py).
It uses [evaluation.py](cloud/workflows/evaluation.py) for fixed-set evaluation
and [run_checks.py](cloud/workflows/run_checks.py) for checkpoint ownership, disk estimates and rollout checks.

| Directory | Tools |
|---|---|
| [setup/](cloud/setup/) | [Install project dependencies](cloud/setup/proj_install.sh), [download data and the BM25 index](cloud/setup/download_data.sh), and [install SkyRL](cloud/setup/skyrl_install.sh) with the [local patch](cloud/setup/apply_patch.py) |
| [services/](cloud/services/) | [Start SkyRL](cloud/services/start_skyrl.sh) or [restart it and wait for API readiness](cloud/services/restart_skyrl.sh) |
| [workflows/](cloud/workflows/) | Drive segmented training, standalone evaluation and short-run validation |
| [checkpoints/](cloud/checkpoints/) | [Prune full states](cloud/checkpoints/ckpt_prune.py) and [upload selected checkpoints](cloud/checkpoints/upload_checkpoint.py) |
| [monitoring/](cloud/monitoring/) | [Collect resource samples](cloud/monitoring/sysmon.sh), [summarize a time window](cloud/monitoring/stats_window.py), and [log training and dev metrics to W&B](cloud/monitoring/log_monitor_metrics.py) |
| [validation/](cloud/validation/) | [Verify short runs and restarts](cloud/validation/run_train_test.sh) and [compare evaluation concurrency settings](cloud/validation/conc_test.sh) |

### Pod setup

Synchronize the project checkout to the Pod, including its lockfiles, fixed
evaluation sets, and `scripts/cloud/`. Run the scripts in that checkout. The
helpers find each other relative to their own files. From the project root,
`bash scripts/cloud/setup/proj_install.sh` installs the locked client and the
independent retrieval environment. SkyRL uses a third environment.

By default the scripts use [env_common.sh](cloud/env_common.sh) to read credentials from `/workspace/.secrets/env.sh`,
use `/workspace/search-r1/SkyRL` for the server checkout and write logs under
`/workspace/logs`. Set `POD_ROOT` to change the workspace root or `POD_ENV_FILE`
to choose a credentials file. The Pod can also set `SKYRL`, `CKPT_BASE`,
`RETRIEVAL_URL`, `SEARCH_R1_DATA_DIR`, `TINKER_BASE_URL` and `TINKER_API_KEY`.
Start BM25 with the chosen data directory (`DATA_DIR` for `retrieval/start.sh`).

The current patch also supports length-based gradient checkpointing. Deployment
defaults to `GC_SHORT_SEQ_THRESHOLD=8000`: for plain text microbatches of size 1,
checkpointing is disabled when the **post-trim input tensor width** is at most
8,000 tokens and enabled above it. Larger microbatches and unsupported layouts
retain checkpointing. This counts context and actions, not only generated tokens.
The historical A100 archive predates this change; memory safety and speed on the
PRO 6000 have not been measured.

The shell value is serialized as
`trainer.gradient_checkpointing_short_seq_threshold` in backend config, so it
reaches Ray workers without relying on custom environment-variable forwarding.
Keep `trainer.gradient_checkpointing=true` and
`trainer.gradient_checkpointing_use_reentrant=false`. The patch preserves the
initialized checkpoint functions and input-gradient hooks, changes only layer
flags around a complete microbatch forward/backward, and restores them on exit.
Worker logs prefixed `[search-r1 gc]` show the threshold, the first sequence length
in each mode, completed microbatch counts and maximum lengths per mode. They are
per backend request, not necessarily per optimizer step.

To recover the previous always-on behavior, restart the server/driver with
`GC_SHORT_SEQ_THRESHOLD=0` (for example,
`GC_SHORT_SEQ_THRESHOLD=0 bash scripts/cloud/train_with_dev_eval.sh RUN_NAME`).
Use the same run directory and training arguments to resume from the latest
complete checkpoint. This does not retry a failed backward or preserve unsaved
updates. Install the updated patch and cloud helpers before starting the server;
the installer supports both a fresh pinned checkout and the previous padding-only
patch. Existing checkpoints remain usable.

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
`UPLOAD=0` disable those operations independently. Pruning protects parent
archives by their resolved paths and skips indexes with unknown ownership.

For example, fork from an ordinary parent's step 7 and stop at global step 18:

```bash
PARENT_RUN=ordinary PARENT_STEP=7 TOTAL_STEPS=18 EVAL_EVERY=5 SAMPLING_MODE=dynamic \
    scripts/cloud/train_with_dev_eval.sh refill limit_search=true
```

The driver evaluates steps 12, 17 and 18. The first upload covers steps
`(7, 12]`. Rerun with the same name, interval and end step to finish missing
evaluations or uploads. Training resumes at the latest complete child state.

Standalone evaluation takes a run name or absolute directory, a step or `base`,
and `dev` or `test`. By default it uses the run's configuration. An explicit
training snapshot overrides the corresponding model and environment options.
The fixed set and evaluation-only settings (such as concurrency) remain selectable.

```bash
scripts/cloud/eval_checkpoint.sh refill 12 dev
scripts/cloud/eval_checkpoint.sh refill 18 test training_config_file=/workspace/runs/train/ordinary/config.json
scripts/cloud/eval_checkpoint.sh ordinary base dev
```

The helper selects the last complete record at the requested step. For base
evaluation on the combined SkyRL server it keeps a LoRA client alive while
sampling the base model. Results under `runs/eval/` include the run, weight,
split and a digest of the evaluation inputs in the directory name. A reusable
result needs the expected question count, zero errors and a completion marker.

`log_monitor_metrics.py` fills two gaps in the training run's W&B charts. It reads
the files training and evaluation already write and logs them to a separate run
`<run_name>-monitor` (fixed id derived from the run name, `group=<run_name>`,
`job_type=monitor`, axis `training/step`); it clears inherited `WANDB_RUN_ID` and
`WANDB_RESUME`, so it never resumes the training run.

- `dev/*`: one point per dev eval from the exact summary supplied by the driver
  (overall EM, F1, answer rate, EM when answered, `max_turns_rate`, search calls,
  no-search and tool-error rates, turns, sampled tokens; EM/F1/answer rate per
  source). `--base-eval DIR` (driver: `BASE_EVAL_DIR`) adds the base model at step 0.
- `rollout/*`: one point per training step from
  `iteration_XXXXXX/train_rollout_summaries.jsonl`. The training
  `env/*/max_turns` and `env/*/stop/*` are averages over the episodes that report
  them and are always 1.0; these are fractions of all episodes. `max_turns_rate`
  counts episodes stopped by the turn cap (as in the dev termination);
  `turns_frac_5` counts all episodes that used five turns, including those that
  answered on the fifth. Also answer rate, search calls per episode, final
  sequence length (last turn's prompt + response; mean, p90, max), answer tokens,
  and per-source `max_turns_rate`/`answered_rate`.

The driver runs it after every dev eval; a failure only writes a warning to the
driver log. `<log_path>/monitor_wandb_state.json` records the files already
logged, so reruns log only new or rewritten files. `--dry-run` prints the points
without touching W&B or the state file. Manual run on the pod:

```bash
uv run --frozen python scripts/cloud/monitoring/log_monitor_metrics.py --run-name RUN_NAME \
    --log-path /workspace/runs/train/RUN_NAME --eval-root runs/eval \
    --dev-summary 12 runs/eval/RUN_NAME-step0012-dev-DIGEST/summary.json
```

## Short validation and analysis tools

- [cloud/validation/run_train_test.sh](cloud/validation/run_train_test.sh): short named training jobs.
  Usage is `run_train_test.sh NAME BATCH_SIZE MAX_STEPS [key=value ...]`. A fresh
  short run defaults to `behavior_if_log_dir_exists=delete`, so use a distinct name.
  With `RESTART_AT`, the default becomes `resume`. For example,
  `SAMPLING_MODE=dynamic RESTART_AT=2 scripts/cloud/validation/run_train_test.sh short 4 3 limit_search=true`
  saves at step 2, restarts the service, then verifies the complete step-3 state.
- [cloud/validation/conc_test.sh](cloud/validation/conc_test.sh): evaluation concurrency comparisons.
- [cloud/monitoring/stats_window.py](cloud/monitoring/stats_window.py): summarize a time window from
  `/workspace/logs/sysmon.csv` and vLLM logs. Its log-time conversion uses the host
  timezone and current year; archived analyses must account for the original clock.

These explain how earlier validation was performed. Keeping them does not schedule
additional tests or make their historical results applicable to a new GPU.
