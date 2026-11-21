# Cloud deployment and training validation

This archive records the 2026-11-16 A100-SXM4-80GB checks. It contains short
validation runs, not a formal training result. The model was Qwen/Qwen3.5-4B with
LoRA rank 32. The Search-R1 runs used the historical non-thinking renderer.

## Training jobs

| Run | Scope | Recorded updates |
|---|---|---:|
| [rl_loop_test](training/rl_loop_test/) | Official cookbook loop integration check, configured with batch 128 and group size 16 | 3 |
| [t64_mb1](training/t64_mb1/) | Search-R1, 64 questions × 8 trajectories, microbatch 1, before the padding patch | 1 |
| [t64_mb1p](training/t64_mb1p/) | Same batch shape, microbatch 1, after the padding patch | 3, followed by 1 resumed update |
| [t64_mb2](training/t64_mb2/) | Same batch shape, microbatch 2, with the padding patch | 1 |

The official loop uses a different task and metric schema; its times are not a
direct Search-R1 performance comparison. The saved t64_mb1p configuration reflects
the resumed run. The retained `metrics.jsonl.pre_resume_000003` preserves the
original three-step ending; its duplicate outside that directory was removed
after confirming identical SHA-256 hashes.

## Recorded timings

The [step timing table](step_timings.csv) directly extracts saved metrics, with
source file and line numbers. The [server timing table](server_stage_timings.csv)
extracts `process_batch_requests(... took ...s)` lines from the raw logs, also
retaining line numbers. The request counts in those lines are not datum counts.

| Run / step | Total (s) | Sampling (s) | Client training phase (s) | Server forward/backward sum (s) | Trained tokens |
|---|---:|---:|---:|---:|---:|
| t64_mb1 / 0 | 1314.8 | 215.9 | 1095.2 | 1081.1 | 1,648,130 |
| t64_mb1p / 0 | 785.6 | 214.0 | 568.5 | 554.4 | 1,670,670 |
| t64_mb1p / 1 | 689.8 | 203.8 | 484.0 | 461.6 | 1,602,033 |
| t64_mb1p / 2 | 610.4 | 169.2 | 439.4 | 432.5 | 1,455,694 |
| t64_mb1p / 3, resumed | 610.0 | 168.2 | 398.7 | 390.2 | 1,295,653 |
| t64_mb2 / 0 | 842.9 | 234.6 | 605.1 | 586.8 | 1,737,534 |

Interpretation:

- The padding patch reduced the observed forward/backward time substantially.
  These runs did not replay identical sampled inputs, so the comparison is not a
  controlled numerical-equivalence or exact speedup measurement.
- Microbatch 2 did not improve this single comparison. Its input lengths differ
  from microbatch 1, so it does not establish a general microbatch optimum.
- `time/total` excludes model/sampler initialization before the training loop.
  The client training phase includes associated update/wait/synchronization work;
  do not add those components again. Checkpoint saving contributes to the resumed
  run's final total, outside its training-phase timer.
- Trained tokens are `tokens/trained`, the sum of model-input lengths submitted
  for training. They are not the sum of every rollout turn's observation and
  action lengths. Previous estimates using 2.75–3.70 million tokens per step and
  treating every turn as a datum were withdrawn.
- Recorded KL/entropy values are finite, but `compute_post_kl=false`: these are
  not post-update KL measurements. The archive does not establish loss/gradient
  equivalence for the padding patch or measured MFU.

## Evaluation checks

| Run | Questions | Purpose | EM | Evaluation-loop time (s) |
|---|---:|---|---:|---:|
| [cloud-smoke-n1](evaluations/cloud-smoke-n1/) | 7 | One question per source; functional smoke check | 28.6% | 28.5 |
| [cloud-conc-c32](evaluations/cloud-conc-c32/) | 271 | Base-model evaluation at concurrency 32 | 32.8% | 84.2 |
| [cloud-conc-c64](evaluations/cloud-conc-c64/) | 271 | Same subset at concurrency 64 | 31.0% | 86.5 |
| [cloud-ckpt-load-a4b4c43c](evaluations/cloud-ckpt-load-a4b4c43c/) | 35 | Load the three-update full state after a server restart | 45.7% | 27.7 |

All four runs report zero request exceptions. The concurrency subset uses up to
40 questions per source, with 31 Bamboogle questions. The checkpoint subset uses
five per source. Evaluation-loop times exclude policy creation/loading, so the
27.7-second value is not checkpoint-loading latency. The concurrency scores come
from separate stochastic generations and do not show that concurrency changes
model quality.

The checkpoint smoke run demonstrates that the API path completed, but its small
evaluation set does not prove that reloaded sampling matches the pre-restart
weights. No same-input pre/post-reload comparison is recorded here. Training resume
and loading were observed; the complete segmented driver, real W&B upload and
cross-Pod recovery were not exercised by these runs.

## Evidence and environment

- [experiment.json](experiment.json) records provenance and definitions of derived files.
- [files.json](files.json) provides original paths, sizes and SHA-256 for unmodified copies.
- [versions.txt](environment/versions.txt) records the SkyRL commit, GPU and client
  packages. Its Torch version is a client value, not a server environment lockfile.
  The raw vLLM startup log reports server vLLM 0.33.0.
- [sysmon.csv](monitoring/sysmon.csv) records GPU and container memory approximately
  every five seconds. `time` is Unix time; GPU memory is in MiB and the named
  container-memory columns are in GiB. Sampled maxima may miss short-lived peaks.
- [train_tests.times](monitoring/train_tests.times) and
  [conc_test.times](monitoring/conc_test.times) preserve shell-level boundaries.
  Some training attempts have no matching END line; do not fill in missing events.
- [training/examples.jsonl](training/examples.jsonl) contains the first saved
  t64_mb1p occurrence of correct/incorrect completion, turn limit, parse error and
  token limit. Each points to its original rollout record.

Historical tinker checkpoint URLs require
their original server files and database registrations; they are references, not
public weight downloads. Full model/optimizer weights are not included in this
experiment archive.
