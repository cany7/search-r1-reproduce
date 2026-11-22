# search-r1-reproduce

A reproduction of Search-R1's multi-turn search RL training with Qwen3.5-4B,
BM25 retrieval and LoRA. Training runs on a single RunPod GPU with self-hosted
SkyRL through the Tinker SDK.

Training adopts three choices from [Search-R1++](https://arxiv.org/abs/2602.19526):
REINFORCE, the F1+ reward (F1, minus 0.1 for never searching and 0.1 for no valid
answer), and the fast-thinking prompt. Advantages are rewards minus the group
mean, with no clipping, std normalization or KL penalty. The trainer supports
ordinary and dynamic sampling, a per-turn search limit and a max-turns unanswered
penalty, with checkpoint recovery and periodic dev-set evaluation.

[Chinese version](README_zh.md)

## Experiment record

[Browse the experiments](experiments/README.md) for configurations, scores,
evaluation trajectories and training metrics.
Explore the four training runs and their checkpoint evaluations in the [public W&B workspace](https://forge.coreweave.com/wandb/timye7-personal/search-r1-reproduce?nw=sw0myi4atxa).

## Code guide

Follow one training update through these files.

| Part | What to read |
| --- | --- |
| Entry points | [Ordinary](search_r1/train.py) and [dynamic](search_r1/train_dynamic_sampling.py) CLIs build a [shared configuration](search_r1/config.py). |
| Start or resume | [runtime.py](search_r1/training/runtime.py) initializes the run; [checkpoints.py](search_r1/training/checkpoints.py) saves and restores model and optimizer state. |
| Collect trajectories | The shared [training loop](search_r1/training/loop.py) coordinates [sampling](search_r1/training/sampling.py) and [multi-turn rollouts](search_r1/env/rollout.py) in the [search environment](search_r1/env/search_tool_env.py), using [BM25 retrieval and scoring](search_r1/env/tools.py). |
| Update and save | [data.py](search_r1/training/data.py) computes group-centered advantages; [update.py](search_r1/training/update.py) submits the importance-sampling policy-gradient loss, applies an Adam step and refreshes sampler weights. The loop saves checkpoints and [logs trajectories and metrics](search_r1/training/logging.py). |
| Evaluate | [offline_eval.py](search_r1/offline_eval.py) uses the same environment and rollout with a [Tinker or oMLX backend](search_r1/backends/). The fixed [dev and test sets](eval_sets/) are included in the repository. |

The [retrieval service](retrieval/) has its own environment. The
[script guide](scripts/README.md) covers deployment and analysis tools.

## Set up the environments

| Component | Version |
| --- | --- |
| Tinker SDK | 0.39.0 |
| Adapted cookbook components | Commit `1991897`, Apache-2.0. See the [source map](third_party/tinker_cookbook/UPSTREAM.md) and [included license](third_party/tinker_cookbook/LICENSE). |
| Search-R1 retrieval server | Commit `598e61b` for `retrieval_server.py` |
| SkyRL | Commit `79d4c89` with padding-trimming and length-based gradient checkpointing patches; see the [script guide](scripts/README.md). |
| Python | 3.12 |
| Pyserini | 2.4.0, requires JDK 21 |

Run all commands below from the repository root.

```bash
uv sync --locked                   # Main environment
(cd retrieval && uv sync --locked) # Retrieval environment
cp .env.example .env               # First-time setup
```

Training needs `TINKER_BASE_URL`, `TINKER_API_KEY` and `WANDB_API_KEY` in the
environment; on the Pod they are set as container environment variables. Local
evaluation reads `OMLX_BASE_URL` and `OMLX_API_KEY` from `.env`.
Training metrics are logged to Weights & Biases (W&B) under
`search-r1-reproduce-experiment` by default; use `wandb_project=<project_name>`
to choose another project or `wandb_project=None` to disable training metric
uploads while keeping local logs.

## Download the data

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

## Run the retrieval server

```bash
./retrieval/start.sh             # Defaults to 127.0.0.1:8001
./retrieval/stop.sh
```

The first startup can take a few minutes while the server builds the corpus cache.

## Four training recipes

Start BM25 before training. Each example uses 64 questions and eight
trajectories per question. Append `max_steps`, `save_every` and `log_path` to
choose the global end step, save interval and run directory.

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

The archived runs, with their starting checkpoints and configurations, are in the
[main reproduction](experiments/reproduce-main/README.md) (ordinary and limited search),
[max-turns unanswered penalty](experiments/max-turns-penalty/README.md) and
[dynamic sampling](experiments/dynamic-sampling/README.md) experiments.

The max-turns unanswered penalty applies only during training; evaluation uses
the run's search rule and the F1+ reward. Dynamic sampling holds the policy fixed
while filtering equal-reward groups and refilling a full batch before updating.

Both entry points can start from the base model or fork from an ordinary-sampling
run using `parent_run=PATH parent_step=N` and a new `log_path`. To resume the same
run, reuse its directory and recipe with `behavior_if_log_dir_exists=resume`.
Forking and resuming both restore optimizer state. For weights-only
initialization, use `load_checkpoint_path=URI`.

On a Pod, `scripts/cloud/train_with_dev_eval.sh` trains in segments with
development evaluation in between, and `scripts/cloud/eval_checkpoint.sh`
evaluates a single checkpoint. See the [script guide](scripts/README.md).

## Evaluate models

Local evaluation runs on a Mac with oMLX and was used to compare base models and
prompts before cloud training. Start oMLX first. The base-model evaluation script
reuses a running retrieval server or starts one and stops it on exit.

```bash
./scripts/run_base_eval.sh                       # All five base-model configurations, dev set
LIMIT=50 ./scripts/run_base_eval.sh q35-fast      # One configuration, first 50 questions per source
uv run python scripts/compare_runs.py runs/eval/base-*
uv run python scripts/diagnose_run.py runs/eval/base-q35-fast-n50
```

Archived baseline results are available through the [experiment index](experiments/README.md).

To run an evaluation directly, start the retrieval server first.

```bash
uv run python -m search_r1.offline_eval backend=omlx omlx_model=Qwen3.5-4B-bf16 \
    eval_file=eval_sets/dev.jsonl limit_per_source=10 run_name=try
```

Each run writes `config.json`, `trajectories.jsonl`, and `summary.json` under
`runs/eval/<run_name>/`.

## Run the tests

```bash
uv run --locked pytest tests/ -q
```

Automated tests cover ordinary and dynamic sampling, training and evaluation
workflows, and checkpoint recovery.
