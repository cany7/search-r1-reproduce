# Experiment archive

[中文版](../docs/experiments_zh.md)

## Experiments

| Experiment | Purpose | Contents |
|---|---|---|
| [Local baselines](local-baselines/) | Compare models and prompts using oMLX | Five configurations evaluated on 631 development questions; run configurations, summaries and trajectories |
| [Cloud validation](cloud-validation/README.md) | Validate deployment, training performance, concurrency and checkpoint recovery | Training and evaluation records, timing tables and resource monitoring |
| [Main reproduction](reproduce-main/) | Two-stage RL training: 0 → 50 → 75, with an extension to step 100 | Training records, 12 evaluations and the selected step-75 checkpoint reference |
| [Dynamic sampling](dynamic-sampling/) | Continue training from step 100 to 105 with dynamic sampling | Training records, step-105 dev evaluation and restart validation |
| [Max-turns unanswered penalty](max-turns-penalty/) | Continue training from step 75 to 100, changing the reward for unanswered episodes at the turn cap from −0.1 to −0.3 | Training records and step-100 dev/test evaluations |

Directory names under `local-baselines/evaluations/` use `q35` for Qwen3.5-4B and
`q3i` for Qwen3-4B-Instruct-2507; `think` indicates thinking mode, `fast` and
`original` identify prompt variants, and `n100` means up to 100 questions per source.

## Files

- `config.json`: Run configuration.
- `summary.json`: Evaluation metrics.
- `trajectories.jsonl`: Per-question evaluation trajectories; large files are gzip-compressed (`.jsonl.gz`).
- `metrics.jsonl`: Training metrics.
- `checkpoints.jsonl`: Checkpoint references; uploaded checkpoint weights are on W&B (see `uploads/*.json`).
- `experiment.json`: Experiment metadata.
- [formal-environment.txt](formal-environment.txt): GPU, container image and server/client versions for the main reproduction, dynamic sampling and max-turns penalty experiments.

## W&B

Training curves, evaluation results and checkpoints are available in the
[W&B workspace](https://forge.coreweave.com/wandb/timye7-personal/search-r1-reproduce?nw=sw0myi4atxa).
