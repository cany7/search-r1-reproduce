# Experiment archive

[中文版](README_zh.md)

This directory archives completed experiments. Working outputs remain under
`runs/`.

## Start here

| Stage | Purpose | Available evidence |
|---|---|---|
| [Local baselines](local-baselines/BASELINE_ANALYSIS.md) | Compare models and prompts using oMLX | Five runs of 631 development questions each; configurations, summaries, all transcripts and bilingual analysis |
| [Cloud validation](cloud-validation/README.md) | Check deployment, training performance, concurrency, checkpoint loading and resume | Four training jobs, four evaluation runs, timing tables, GPU/memory samples and selected training examples |
| [Main reproduction](reproduce-main/) | RL stages 0 → 50 → 75, followed by a 75 → 100 comparison | Training records, 12 evaluations and the selected step-75 checkpoint reference |
| [Dynamic sampling](dynamic-sampling/) | Branch from step 100 | Training through step 105, its dev evaluation and a separate restart smoke run |
| [Max-turns penalty](max-turns-penalty/) | Branch from step 75 | Training through step 100 and its dev and test evaluations |

The five local runs contain 3,155 evaluation records. The cloud smoke/concurrency/
checkpoint evaluations contain 584 records across four runs; these include repeated
questions and must not be described as 584 unique questions.

## How to read a run

- `config.json` preserves the configuration present when the outputs were collected.
- `summary.json` contains the recorded evaluation aggregates. `trajectories.jsonl`
  contains per-question trajectories where available.
- Training `metrics.jsonl` contains recorded steps and, where present, a final
  summary row. Do not count the final summary as an additional optimizer update.
- `checkpoints.jsonl` contains checkpoint references, not checkpoint weights.
- `experiment.json` explains scope, provenance, missing information and any derived
  files. `files.json` records the original path and size of each archived source file.

The historical Qwen3.5 non-thinking runs used `qwen3_5_disable_thinking`, before
the Unicode correction in `1b26fd7`. The formal training records use the Unicode
renderer. In formal evaluation configs, `eval_file` identifies the dataset, including
dev runs saved with `split: "test"`. Future baselines and trained-model evaluations
must share their rendering and scoring settings. The local development results are
not a formal cloud baseline or a held-out test score.

## Provenance limits

- The original project execution commits were not reliably captured. Archive
  commits identify packaging changes, not the code that produced the old runs.
- The cloud `code.diff` files only say that the installed cookbook was not in a Git
  repository.
- Dataset hashes in `experiment.json` were recorded during archiving, not at
  execution time. Configurations and manifests are kept verbatim rather than
  silently rewritten to match the current source tree.
- Saved transcripts and rollout summaries do not contain every original token ID
  or complete RNG state. They support behavioral and metric analysis, not exact
  token-level replay.
- Derived training examples are the first occurrences of five specified outcomes;
  they are illustrations, not a representative sample for estimating rates.

## Keeping the record organized

New runs continue writing to `runs/`. Once a run is complete, archive its immutable
configuration, scores and traces under the directory for that experiment, retaining
failures and superseded conclusions with their explanations. Record the actual
source commit, dataset/model revisions, renderer, seed, GPU, environment and
checkpoint references at execution time when available.

Keep full runtime logs and large training traces with the original run outputs.
Preserve historical results when a renderer or scoring rule changes. `local_docs/`,
including `PROGRESS.md`, remains local.
