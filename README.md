# search-r1-reproduce

A small-scale reproduction of Search-R1 with Qwen3.5, BM25 retrieval and LoRA.
Ordinary and dynamic sampling share a training loop in this repository. The
Tinker SDK sends sampling and optimizer requests to a model service such as SkyRL.

[Chinese version](README_zh.md)

## Experiment record

[Browse the experiments](experiments/README.md) for configurations, scores,
evaluation trajectories, training metrics and provenance.

- [Local baseline analysis](experiments/local-baselines/BASELINE_ANALYSIS.md):
  five model/prompt configurations, 631 development questions each, evaluated with oMLX.
- [Cloud validation](experiments/cloud-validation/README.md): deployment,
  padding optimization, microbatch/concurrency comparisons, checkpoint loading and resume.
- The [main reproduction](experiments/reproduce-main/README.md) uses the step-75
  checkpoint, which scored 892/1,894 (47.10% EM) on the limited-search test set.

The local baselines and cloud validation runs predate the Unicode renderer correction. Their original
configurations and scores are preserved; they are not measurements of the current
renderer.

## Versions

| Component | Version |
| --- | --- |
| Tinker SDK | 0.39.0 in the client lockfile |
| Adapted cookbook sources | Commit `1991897`. The project ships the selected code and its Apache-2.0 license. See the [source map](third_party/tinker_cookbook/UPSTREAM.md). |
| Search-R1 | Commit `598e61b` for `retrieval_server.py` |
| SkyRL | Pinned to `79d4c89` with local patches. The A100 archive used padding trimming; current deployment also switches gradient checkpointing by sequence length (not yet GPU-validated). See the [cloud script guide](scripts/README.md). Client and server environments are separate. |
| Python | 3.12.14, selected with `pyenv local`. uv uses the installed interpreter. |
| Pyserini | 2.4.0, requires JDK 21 |

## Repository layout

Follow one training update through these files.

| Part | What to read |
| --- | --- |
| Entry points | [Ordinary](search_r1/train.py) and [dynamic](search_r1/train_dynamic_sampling.py) CLIs build a [shared configuration](search_r1/config.py). |
| Start or resume | [runtime.py](search_r1/training/runtime.py) locks the run and initializes the SDK client. [checkpoints.py](search_r1/training/checkpoints.py) selects the full state, optimizer and next prompt offset. |
| Collect trajectories | [loop.py](search_r1/training/loop.py) collects question groups through [sampling.py](search_r1/training/sampling.py). Each episode uses the [rollout](search_r1/env/rollout.py), [search environment](search_r1/env/search_tool_env.py), [BM25 tool and scorer](search_r1/env/tools.py), and [Qwen renderers](search_r1/env/renderers/). |
| Update and save | [data.py](search_r1/training/data.py) centers rewards within a group and aligns action targets and old logprobs. [update.py](search_r1/training/update.py) submits gradients, takes an Adam step and publishes sampling weights. The loop writes trajectories and [metrics](search_r1/training/logging.py) before committing the checkpoint index. |
| Evaluate | [offline_eval.py](search_r1/offline_eval.py) uses the same environment and rollout with a [Tinker or oMLX backend](search_r1/backends/). The fixed [dev and test sets](eval_sets/) are versioned. |

The [retrieval service](retrieval/) has its own environment. Scripts for deployment
and analysis live under [scripts/](scripts/README.md). Local data and working
runs are Git-ignored.

Training and base-model evaluation default to
`search_r1/qwen3_5_disable_thinking_unicode`. This renderer preserves non-ASCII
characters in list/dict tool arguments; it retains the cookbook's message rendering,
parsing and rollout behavior. Its name is saved in training/evaluation configuration
and training-client metadata. Checkpoint evaluation uses the recorded renderer when
available. The original `qwen3_5_disable_thinking` remains available for historical runs.

## Download the data

Run these commands from the repository root.

```bash
mkdir -p data
(
    cd data
    hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset --local-dir nq_hotpotqa_train
    hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir bm25-index
    hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir corpus
    gunzip -k corpus/wiki-18.jsonl.gz
)
```

## Set up the environments

```bash
uv sync --locked
(cd retrieval && uv sync --locked)
cp .env.example .env             # First-time setup
```

Set `OMLX_API_KEY` in `.env` for local evaluation and `WANDB_API_KEY` for
training. The training entry point reads `.env` from the repository root.
Existing environment variables take precedence.

Training logs to Weights & Biases (W&B) by default under the
`search-r1-reproduce` project. The generated run name includes the model, batch
size, learning rate, and timestamp. Set `wandb_project=<project_name>` to use a
different project, or `wandb_project=None` to disable uploads and keep local logs.

## Four training recipes

Start BM25, configure `TINKER_BASE_URL` and `TINKER_API_KEY` (for a self-hosted Pod,
see the [script guide](scripts/README.md)), and run from the project root.
Each example uses 64 questions and eight trajectories per question. The ordinary
CLI defaults to 512 questions. Append `max_steps`, `save_every` (default 25) and
`log_path` to choose the global end step, save interval and run directory.

```bash
# Ordinary sampling
uv run --frozen python -m search_r1.train batch_size=64 group_size=8

# Limit each turn to the first search call and its first three queries
uv run --frozen python -m search_r1.train batch_size=64 group_size=8 limit_search=true

# Assign -0.3 to training episodes that reach the turn cap without a valid answer
uv run --frozen python -m search_r1.train batch_size=64 group_size=8 limit_search=true max_turns_unanswered_reward=-0.3

# Refill groups whose rewards are all equal using fresh questions
uv run --frozen python -m search_r1.train_dynamic_sampling batch_size=64 group_size=8 limit_search=true
```

A valid answer on the final turn receives the usual score. Evaluation inherits
the saved search rule and base scorer. The unanswered penalty applies only in
training. Dynamic sampling holds the policy fixed during refill and requires
a full batch to update. Ordinary sampling also supports streaming minibatches.

Both entries can start at the base model. To fork an ordinary run, pass
`parent_run=PATH parent_step=N` and a new `log_path`. To resume, reuse the
run directory and recipe with `behavior_if_log_dir_exists=resume`. Both restore
optimizer state. For weights-only initialization use `load_checkpoint_path=URI`.

The Pod driver runs development evaluation between segments. From the synchronized
checkout, use `scripts/cloud/train_with_dev_eval.sh RUN [key=value ...]`.
Use `scripts/cloud/eval_checkpoint.sh RUN STEP dev` or replace `dev` with
`test` for a held-out evaluation. The driver selects the exact full-state record
and the run's configuration. See the [script guide](scripts/README.md) for
base-model evaluation, configuration overrides and upload behavior.

## Run the retrieval server

```bash
./retrieval/start.sh             # Defaults to 127.0.0.1:8001
./retrieval/stop.sh
```

The first startup can take a few minutes while the server converts the corpus
to an Arrow cache under `~/.cache/huggingface/datasets`.

## Evaluate models

Start oMLX before running a local evaluation. The base-model evaluation
script reuses a running retrieval server or starts one and stops it on exit.

```bash
./scripts/run_base_eval.sh                       # All five base-model configurations
LIMIT=50 ./scripts/run_base_eval.sh q35-fast      # One configuration, first 50 questions per source
uv run python scripts/compare_runs.py runs/eval/base-*
uv run python scripts/diagnose_run.py runs/eval/base-q35-fast-n50
```

To inspect the archived local baselines without generating new samples:

```bash
uv run python scripts/compare_runs.py experiments/local-baselines/evaluations/base-*
uv run python scripts/diagnose_run.py experiments/local-baselines/evaluations/base-q35-fast-n100
```

To run an evaluation directly, start the retrieval server first.

```bash
uv run python -m search_r1.offline_eval backend=omlx omlx_model=Qwen3.5-4B-bf16 \
    eval_file=eval_sets/dev.jsonl limit_per_source=10 run_name=try
```

Each run writes `config.json`, `trajectories.jsonl`, and `summary.json` under
`runs/eval/<run_name>/`. Completed historical evaluations have been moved to
`experiments/`; the per-experiment manifests record their original paths and sizes.

## Run the tests

```bash
uv run --locked pytest tests/ -q
```

The suite uses fixed reference trajectories, local files and simulated SDK/Pod
processes. It covers both sampling modes, checkpoint recovery, and offline CLI
help in an environment without the cookbook package. GPU training and live
service behavior are outside this validation.
