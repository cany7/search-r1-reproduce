# Script guide

These are maintained project tools. Historical cloud copies belong to the
[A100 raw attachment](../experiments/attachments/README.md), not this directory.
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
[baseline analysis](../experiments/2026-11-16-local-baselines/BASELINE_ANALYSIS.md)
for these interpretation limits.

Analysis scripts accept an archived evaluation directory under `experiments/`
as well as a working directory under `runs/eval/`.

## Cloud layout and deployment

Cloud shell scripts expect the project under `/workspace/search-r1/project`,
SkyRL under `/workspace/search-r1/SkyRL`, and these cloud helpers copied into
`/workspace/bin`. They load cloud credentials from `/workspace/.secrets/env.sh`,
which is not part of the repository or experiment archive. Working logs and
checkpoints are written under `/workspace`.

| Script | Purpose |
|---|---|
| [cloud/env_common.sh](cloud/env_common.sh) | Shared environment and project path |
| [cloud/proj_install.sh](cloud/proj_install.sh) | Install locked client/retrieval dependencies |
| [cloud/download_data.sh](cloud/download_data.sh) | Download datasets, corpus and BM25 index |
| [cloud/skyrl_install.sh](cloud/skyrl_install.sh) | Install pinned SkyRL and apply the local padding patch |
| [cloud/start_skyrl.sh](cloud/start_skyrl.sh) | Start the configured server |
| [cloud/restart_skyrl.sh](cloud/restart_skyrl.sh) | Restart the server and wait for API readiness |
| [cloud/sysmon.sh](cloud/sysmon.sh) | Record GPU and container memory samples |

`skyrl_install.sh` invokes [cloud/apply_patch.py](cloud/apply_patch.py). The matching
[patch file](cloud/skyrl-79d4c89-trim-microbatch-padding.patch) documents the source
change. Both are retained; deleting either is not a substitute for consolidating
their maintenance.

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
`GC_SHORT_SEQ_THRESHOLD=0 bash /workspace/bin/train_with_dev_eval.sh RUN_NAME`).
Use the same run directory and training arguments to resume from the latest
complete checkpoint. This does not retry a failed backward or preserve unsaved
updates. Install the updated patch and cloud helpers before starting the server;
the installer supports both a fresh pinned checkout and the previous padding-only
patch. Existing checkpoints remain usable.

## Training and artifact management

| Script | Purpose |
|---|---|
| [cloud/train_with_dev_eval.sh](cloud/train_with_dev_eval.sh) | Segmented training, full-state dev evaluation, cleanup and background W&B uploads |
| [cloud/ckpt_prune.py](cloud/ckpt_prune.py) | Retain evaluation/latest full states and remove other or duplicate full states |
| [cloud/upload_checkpoint.py](cloud/upload_checkpoint.py) | Upload selected states, samplers and evaluation records |
| [cloud/stop_train.sh](cloud/stop_train.sh) | Explicitly stop the train session and matching training processes |

Run the segmented driver in its own session, not the `skyrl` session it restarts.
Its argument is a run name, followed by optional `train.py` arguments. The saved
training configuration supplies the development evaluation settings. The full
driver/upload/recovery workflow is not established by the archived short tests.

## Short validation and analysis tools

- [cloud/run_train_test.sh](cloud/run_train_test.sh): short named training jobs.
  It uses `behavior_if_log_dir_exists=delete`; use distinct names for new evidence.
- [cloud/conc_test.sh](cloud/conc_test.sh): evaluation concurrency comparisons.
- [cloud/stats_window.py](cloud/stats_window.py): summarize a time window from
  `/workspace/logs/sysmon.csv` and vLLM logs. Its log-time conversion uses the host
  timezone and current year; archived analyses must account for the original clock.

These explain how earlier validation was performed. Keeping them does not schedule
additional tests or make their historical results applicable to a new GPU.
