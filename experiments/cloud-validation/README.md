# Cloud deployment and training validation

Deployment, training, evaluation concurrency, and checkpoint recovery were
validated on an NVIDIA A100-SXM4-80GB using Qwen/Qwen3.5-4B with LoRA rank 32
on 2026-11-16.

## Training validation

All Search-R1 runs used 64 questions per batch and eight trajectories per question.

| Run | Microbatch size | Padding trimming | Completed updates |
|---|---:|---|---|
| [t64_mb1](training/t64_mb1/) | 1 | Disabled | 1 |
| [t64_mb1p](training/t64_mb1p/) | 1 | Enabled | 3 + 1 after resume |
| [t64_mb2](training/t64_mb2/) | 2 | Enabled | 1 |

## Evaluation validation

| Run | Questions | Purpose |
|---|---:|---|
| [cloud-smoke-n1](evaluations/cloud-smoke-n1/) | 7 | Functional smoke test |
| [cloud-conc-c32](evaluations/cloud-conc-c32/) | 271 | Base-model evaluation at concurrency 32 |
| [cloud-conc-c64](evaluations/cloud-conc-c64/) | 271 | Evaluate the same subset at concurrency 64 |
| [cloud-ckpt-load-a4b4c43c](evaluations/cloud-ckpt-load-a4b4c43c/) | 35 | Load the three-update full state and evaluate after a server restart |

All four evaluations completed with zero recorded request exceptions.

## Artifacts and environment

- [Environment versions](environment/versions.txt): hardware, SkyRL revision, and client/server package versions.
- Timing records: [training steps](step_timings.csv) and [server stages](server_stage_timings.csv).
- [Resource monitoring](monitoring/) and [training trajectory examples](training/examples.jsonl).
- [SDK integration validation](training/rl_loop_test/): three updates using the upstream cookbook loop.
- [Experiment metadata](experiment.json).

This validation archive contains no model or optimizer weights; its `tinker://`
checkpoint references point to the original server and are no longer downloadable.
