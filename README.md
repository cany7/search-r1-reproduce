# search-r1-reproduce

Search-R1 小规模复现。计划见 knowledge-base 的《Search-R1 复现计划》。

## 固定的版本

| 项 | 版本 |
|---|---|
| tinker-cookbook | `1991897`（2026-11-14），在 `pyproject.toml` 里固定 |
| Search-R1（只取 `retrieval_server.py`） | `598e61b` |
| SkyRL（云端才用） | 原参照 `79d4c89`；允许按官方 CUDA 13 镜像选兼容版本，部署时记录并固定实际 commit |
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
cp .env.example .env             # 首次配置；本地评测填 OMLX_API_KEY，训练填 WANDB_API_KEY
```

训练默认开启 W&B，项目名为 `search-r1-reproduce`，run 名自动包含模型、batch、学习率和时间。
训练入口自动读取项目根目录的 `.env`，已有环境变量优先。
可用 `wandb_project=<项目名>` 改名，或 `wandb_project=None` 关闭上传；本地日志仍会保存。

迁移到云端时带上 `pyproject.toml` 和 `uv.lock`，执行 `uv sync --locked` 即会安装包括
`wandb` 在内的锁定依赖。云端另行配置 `WANDB_API_KEY`，无需复制本地 `.venv`。

## 云端训练（待验证）

先启动 SkyRL Tinker 服务和 BM25 检索服务，再运行：

```bash
TINKER_API_KEY=tml-dummy TINKER_BASE_URL=http://127.0.0.1:8000 \
    uv run python -m search_r1.train batch_size=64 group_size=8 max_steps=100 \
    save_every=25 log_path=/workspace/runs/search-r1-train
```

`save_every` 默认 25；检查点冒烟测试可设为 1。它只控制保存，固定开发集的评测
需另行用 `offline_eval.py` 加载检查点运行，不能仅靠 `eval_every=25` 开启。
训练入口的 `batch_size` 默认仍是 cookbook 的 512 题，所以当前方案须显式传 64。
正式运行前先完成计划第 4 步的 2–3 步训练及检查点测试。

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
