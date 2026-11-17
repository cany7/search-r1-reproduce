# search-r1-reproduce

基于 tinker-cookbook `recipes/search_tool` 的 Search-R1 小规模复现，使用 BM25 检索、
Qwen3.5 和通过自部署 SkyRL Tinker 服务进行的 LoRA 训练。

[English version](README.md)

## 实验记录

[实验索引](experiments/README_zh.md)包含配置、分数、完整评测轨迹、短训指标、监控数据和来源记录。

- [本地基座分析](experiments/2026-11-16-local-baselines/BASELINE_ANALYSIS_zh.md)：
  五组模型／prompt 配置，每组 631 道开发集题目，使用 oMLX。
- [A100 验证记录](experiments/2026-11-16-a100-validation/README.md)：部署、padding 优化、
  microbatch／并发比较、检查点加载和续训。
- 归档中尚无正式训练及最终评测结果。

这些历史运行早于 Unicode renderer 修正，原配置和评分保持不变，不能视为当前 renderer 的实测。
[原始日志附件说明](experiments/attachments/README.md)记录了已在本地准备、尚未上传到 GitHub Releases 的归档包。

## 固定的版本

| 项 | 版本 |
|---|---|
| tinker-cookbook | `1991897`（2026-11-14），在 `pyproject.toml` 里固定 |
| Search-R1（只取 `retrieval_server.py`） | `598e61b` |
| SkyRL（云端才用） | 固定 `79d4c89` 加本地补丁；A100 归档使用 padding 裁剪，当前部署另加按序列长度切换梯度检查点（尚无 GPU 验证），见[脚本说明](scripts/README.md)。客户端与服务端环境独立 |
| Python | 3.12.14，`pyenv local` 指定，uv 只用它 |
| pyserini | 2.4.0，需要 JDK 21 |

## 目录

- `search_r1/`：基于 cookbook 的 `recipes/search_tool` 修改。
  - `tools.py`：BM25 检索工具、reward（F1+，原版可选）。
  - `search_env.py`：prompt（fast，原版可选）、数据加载、环境构建。
  - `renderers.py`：保留 Qwen3.5 工具参数中的 Unicode 字符，与模型 chat template 一致。
  - `train.py`：训练入口（云端用）。
  - `offline_eval.py`：评测，支持 tinker 和本地 omlx 两种后端，逐条轨迹落盘。
  - `local_completer.py`：本地推理适配，只在本地用。
- `retrieval/`：BM25 检索服务，独立的 uv 环境。`retrieval_server.py` 相对原文件只加了 `--host` / `--port`。
- `eval_sets/`：固定的开发集和测试集，由 `scripts/make_eval_sets.py` 生成。
- `experiments/`：用于 GitHub 展示的归档结果与分析。
- `scripts/`：部署、评测和分析工具，见[脚本说明](scripts/README.md)。
- `tests/`：reward、云端操作和 W&B 兼容检查。
- `data/`、`runs/`：本地数据与运行中的输出，不进 Git。
- `local_docs/`：本地工作记录，继续忽略，不作为公开实验记录提交。

训练和基座评测默认使用 `search_r1/qwen3_5_disable_thinking_unicode`，仅修正列表／字典
工具参数中的非 ASCII 字符转义，沿用 cookbook 的消息渲染、解析和 rollout 逻辑。
renderer 名称写入训练／评测配置和训练客户端元数据；检查点评测优先使用记录的名称。
原 `qwen3_5_disable_thinking` 仍可用于历史运行。

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

## 云端训练

先启动 SkyRL Tinker 服务和 BM25 检索服务，部署目录约定见[脚本说明](scripts/README.md)。
以下参数只展示 CLI 用法，不代表后续正式训练规模或 GPU 已确定：

```bash
TINKER_API_KEY=tml-dummy TINKER_BASE_URL=http://127.0.0.1:8000 \
    uv run python -m search_r1.train batch_size=64 group_size=8 max_steps=100 \
    save_every=5 log_path=/workspace/runs/search-r1-train
```

直接训练 CLI 的 `save_every` 默认 25，此处覆盖为 5。训练入口的 `batch_size` 默认仍是
cookbook 的 512 题，所以示例显式传 64。

固定开发集评测需另行运行 `offline_eval.py backend=tinker`，将 `checkpoints.jsonl` 中的
完整训练状态路径传给 `tinker_state_url`，并指定 `eval_file=eval_sets/dev.jsonl`。
自部署 SkyRL 应使用这条完整状态加载路径，`tinker_checkpoint_url` 的 sampler 路径不能可靠替代。
训练参数 `eval_every` 不会自动执行这项评测。

分段训练和开发集评测使用 `scripts/cloud/train_with_dev_eval.sh <run_name> [额外 train.py 参数]`。
默认 `TOTAL_STEPS=100`、`BATCH_SIZE=64`、`GROUP_SIZE=8`、`EVAL_EVERY=25`、`SAVE_EVERY=5`、
`EVAL_CONCURRENCY=32`、`PRUNE_STATES=1`、`UPLOAD=1`；正式运行按选定配置显式设置。
驱动从保存的训练配置读取评测参数，并上传检查点到 W&B。基座评测需单独进行。
已有 A100 短测不代表完整分段驱动和真实上传／恢复流程已经验证。

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

读取已归档的本地基座结果，不生成新样本：

```bash
uv run python scripts/compare_runs.py experiments/2026-11-16-local-baselines/evaluations/base-*
uv run python scripts/diagnose_run.py experiments/2026-11-16-local-baselines/evaluations/base-q35-fast-n100
```

单独调用（要先自己起检索服务）：

```bash
uv run python -m search_r1.offline_eval backend=omlx omlx_model=Qwen3.5-4B-bf16 \
    eval_file=eval_sets/dev.jsonl limit_per_source=10 run_name=try
```

新输出仍写到 `runs/eval/<run_name>/`：`config.json`、`trajectories.jsonl`、`summary.json`。
已完成的历史评测移入 `experiments/`，各实验文件清单保留原路径和校验值。

## 测试

```bash
uv run pytest tests/
```
