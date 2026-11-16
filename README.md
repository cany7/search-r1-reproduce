# search-r1-reproduce

Search-R1 小规模复现。计划见 knowledge-base 的《Search-R1 复现计划》。

## 固定的版本

| 项 | 版本 |
|---|---|
| tinker-cookbook | `1991897`（2026-11-14），在 `pyproject.toml` 里固定 |
| Search-R1（只取 `retrieval_server.py`） | `598e61b` |
| SkyRL（云端才用） | `79d4c89`（2026-11-13） |
| Python | 3.12.14，`pyenv local` 指定，uv 只用它 |
| pyserini | 2.4.0，需要 JDK 21 |

## 目录

- `search_r1/`：基于 cookbook 的 `recipes/search_tool` 修改。
  - `tools.py`：BM25 检索工具、reward（F1+，原版可选）。
  - `search_env.py`：prompt（fast，原版可选）、数据加载、环境构建。
  - `train.py`：训练入口（云端用）。
  - `offline_eval.py`：评测，支持 tinker 和本地 omlx 两种后端，逐条轨迹落盘。
  - `local_completer.py`：本地推理适配，只在本地用。
- `retrieval/`：BM25 检索服务，独立的 uv 环境。`retrieval_server.py` 相对原文件只加了 `--host` / `--port`。
- `eval_sets/`：固定的开发集和测试集，由 `scripts/make_eval_sets.py` 生成。
- `scripts/`：环境自检、评测集划分、基座评测、结果对比。
- `tests/`：reward 的单元测试。
- `data/`、`runs/`：数据和评测输出，不进 git。

## 数据

```bash
cd data
hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset --local-dir nq_hotpotqa_train
hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir bm25-index
hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir corpus
gunzip -k corpus/wiki-18.jsonl.gz
```

## 环境

```bash
uv sync                          # 主环境
(cd retrieval && uv sync)        # 检索环境
cp .env.example .env             # 然后填 OMLX_API_KEY
```

## 启动检索服务

```bash
./retrieval/start.sh             # 默认 127.0.0.1:8001
./retrieval/stop.sh
```

首次启动会把语料转成 Arrow 缓存（在 `~/.cache/huggingface/datasets`），需要几分钟。

## 评测

本地用 oMLX，需要事先开着。`run_base_eval.sh` 自己启动和停止检索服务：

```bash
./scripts/run_base_eval.sh                       # 5 个基座配置，开发集
LIMIT=50 ./scripts/run_base_eval.sh q35-fast     # 单个配置，每个来源前 50 题
uv run python scripts/compare_runs.py runs/eval/base-*
uv run python scripts/diagnose_run.py runs/eval/<run_name>
```

单独调用（要先自己起检索服务）：

```bash
uv run python -m search_r1.offline_eval backend=omlx omlx_model=Qwen3.5-4B-bf16 \
    eval_file=eval_sets/dev.jsonl limit_per_source=10 run_name=try
```

输出在 `runs/eval/<run_name>/`：`config.json`、`trajectories.jsonl`、`summary.json`。

## 测试

```bash
uv run pytest tests/
```
