# Experiment archive

[中文版](README_zh.md)

This directory separates completed experimental evidence from the working outputs
under `runs/`. It records model selection and infrastructure validation before
formal Search-R1 training. No experiments were rerun to create this archive.

## Start here

| Stage | Purpose | Available evidence |
|---|---|---|
| [Local baselines](local-baselines/BASELINE_ANALYSIS.md) | Compare models and prompts using oMLX | Five runs of 631 development questions each; configurations, summaries, all transcripts and bilingual analysis |
| [Cloud validation](cloud-validation/README.md) | Check deployment, training performance, concurrency, checkpoint loading and resume | Four training jobs, four evaluation runs, timing tables, GPU/memory samples and selected training examples |
| Formal training | Train and evaluate the final model under the selected configuration | Not yet present |

The five local runs contain 3,155 evaluation records. The cloud smoke/concurrency/
checkpoint evaluations contain 584 records across four runs; these include repeated
questions and must not be described as 584 unique questions.

## How to read a run

- `config.json` preserves the configuration present when the outputs were collected.
- `summary.json` contains the recorded evaluation aggregates; `trajectories.jsonl`
  contains every saved evaluation record, including failures.
- Training `metrics.jsonl` contains recorded steps and, where present, a final
  summary row. Do not count the final summary as an additional optimizer update.
- `checkpoints.jsonl` contains checkpoint references, not checkpoint weights.
- `experiment.json` explains scope, provenance, missing information and any derived
  files. `files.json` records the original path and size of each archived source file.

The historical Qwen3.5 non-thinking runs used `qwen3_5_disable_thinking`, before
the Unicode correction in `1b26fd7`. Future baselines and trained-model evaluations
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
configuration, scores and traces under a dated experiment directory, retaining
failures and superseded conclusions with their explanations. Record the actual
source commit, dataset/model revisions, renderer, seed, GPU, environment and
checkpoint references at execution time when available.

Keep full runtime logs and large training traces with the original run outputs.
Preserve historical results when a renderer or scoring rule changes. `local_docs/`,
including `PROGRESS.md`, remains local.
