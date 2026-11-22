# search-r1-reproduce

基于 Qwen3.5-4B、BM25 检索和 LoRA 复现 Search-R1 的多轮搜索强化学习训练，通过 Tinker SDK 在 RunPod 单卡自部署的 SkyRL 上运行。

训练采用 [Search-R1++](https://arxiv.org/abs/2602.19526) 的三项设置：REINFORCE、F1+ 奖励（F1 分数，从未搜索扣 0.1，没有有效答案扣 0.1）和 fast thinking prompt。优势为奖励减组内均值，不使用 clip、标准差归一化和 KL。训练支持普通采样、动态采样、每轮搜索限制和轮数上限未作答惩罚，并提供检查点恢复与定期开发集评测。

[English version](README.md)

## 实验记录

[实验索引](experiments/README_zh.md)包含配置、分数、评测轨迹和训练指标。
四次训练的曲线和各 checkpoint 的评测结果见[公开 W&B 工作区](https://forge.coreweave.com/wandb/timye7-personal/search-r1-reproduce?nw=sw0myi4atxa)。

## 代码导读

可以按一次训练更新的顺序阅读代码。

| 环节 | 实现 |
|---|---|
| 参数入口 | [普通](search_r1/train.py)和[动态](search_r1/train_dynamic_sampling.py)入口共用一套[配置](search_r1/config.py)。 |
| 启动和恢复 | [runtime.py](search_r1/training/runtime.py) 初始化运行；[checkpoints.py](search_r1/training/checkpoints.py) 保存与恢复模型和优化器状态。 |
| 收集轨迹 | 共用的[训练循环](search_r1/training/loop.py)协调[采样](search_r1/training/sampling.py)与[多轮作答](search_r1/env/rollout.py)，通过[搜索环境](search_r1/env/search_tool_env.py)执行 [BM25 检索与评分](search_r1/env/tools.py)。 |
| 更新和保存 | [data.py](search_r1/training/data.py) 计算组内中心化优势；[update.py](search_r1/training/update.py) 提交重要性采样策略梯度损失，执行 Adam 更新并刷新采样权重。主循环保存检查点并[记录轨迹与指标](search_r1/training/logging.py)。 |
| 评测 | [offline_eval.py](search_r1/offline_eval.py) 通过 [Tinker 或 oMLX 后端](search_r1/backends/)复用同一套环境和多轮作答逻辑。固定的[开发集和测试集](eval_sets/)随仓库提供。 |

[检索服务](retrieval/)使用独立环境。[脚本指南](scripts/README.md)说明部署和分析工具的用法。

## 环境配置

| 项 | 版本 |
|---|---|
| Tinker SDK | 0.39.0 |
| 基于 cookbook 的组件 | Commit `1991897`，Apache-2.0，详见[来源映射](third_party/tinker_cookbook/UPSTREAM.md)和[随仓库提供的许可证](third_party/tinker_cookbook/LICENSE) |
| Search-R1 检索服务 | `retrieval_server.py` 来源 commit `598e61b` |
| SkyRL | Commit `79d4c89`，附带 padding 裁剪和按序列长度切换梯度检查点的补丁，见[脚本指南](scripts/README.md) |
| Python | 3.12 |
| Pyserini | 2.4.0，需要 JDK 21 |

下文命令均在项目根目录执行。

```bash
uv sync --locked                   # 主环境
(cd retrieval && uv sync --locked) # 检索环境
cp .env.example .env               # 首次配置
```

训练需要环境变量 `TINKER_BASE_URL`、`TINKER_API_KEY` 和 `WANDB_API_KEY`，在 Pod 上配置为容器环境变量。本地评测从 `.env` 读取 `OMLX_BASE_URL` 和 `OMLX_API_KEY`。
训练指标默认记录到 Weights & Biases（W&B）的 `search-r1-reproduce-experiment` 项目；可用 `wandb_project=<项目名>` 指定其他项目，或 `wandb_project=None` 关闭训练指标上传，本地日志仍会保存。

## 数据准备

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

## 启动检索服务

```bash
./retrieval/start.sh             # 默认 127.0.0.1:8001
./retrieval/stop.sh
```

首次启动需要建立语料缓存，可能耗时数分钟。

## 四种训练配方

训练前先启动 BM25。四个示例每批选 64 道题，每题采样 8 条轨迹。可追加 `max_steps`、`save_every` 和 `log_path`，分别指定全局终点、保存间隔和运行目录。

```bash
# 普通采样
uv run --frozen python -m search_r1.train batch_size=64 group_size=8

# 限制搜索，每轮只执行第一个 search 的前三条 query
uv run --frozen python -m search_r1.train batch_size=64 group_size=8 limit_search=true

# 训练时达到轮数上限且没有有效答案，终局奖励设为 -0.3
uv run --frozen python -m search_r1.train batch_size=64 group_size=8 limit_search=true max_turns_unanswered_reward=-0.3

# 动态采样，剔除奖励全同的题组，再用新题补齐
uv run --frozen python -m search_r1.train_dynamic_sampling batch_size=64 group_size=8 limit_search=true
```

归档实验的训练起点与完整配置见[主复现](experiments/reproduce-main/README.md)（普通采样与限制搜索）、[轮数上限未作答惩罚](experiments/max-turns-penalty/README.md)和[动态采样](experiments/dynamic-sampling/README.md)。

轮数上限未作答惩罚只在训练时生效；评测沿用该 run 的搜索规则，奖励按 F1+ 计算。动态采样在同一策略下过滤奖励相同的题组，补齐整批后再更新。

普通采样和动态采样入口均可从基座开始，也可从普通采样 run 的检查点分叉，传入 `parent_run=路径 parent_step=N` 和新的 `log_path`。同一 run 续训时复用原目录与配方，传入 `behavior_if_log_dir_exists=resume`。分叉与续训均恢复优化器状态；仅加载权重时使用 `load_checkpoint_path=URI`。

在 Pod 上，`scripts/cloud/train_with_dev_eval.sh` 分段训练并在段间评测开发集，`scripts/cloud/eval_checkpoint.sh` 单独评测一个检查点，详见[脚本指南](scripts/README.md)。

## 评测

本地评测在 Mac 上用 oMLX 运行，用于云端训练前比较基座模型和 prompt。评测前需先启动 oMLX。`run_base_eval.sh` 会复用已运行的检索服务，没有则自行启动，并在退出时停止：

```bash
./scripts/run_base_eval.sh                       # 5 个基座配置，开发集
LIMIT=50 ./scripts/run_base_eval.sh q35-fast      # 单个配置，每个来源前 50 题
uv run python scripts/compare_runs.py runs/eval/base-*
uv run python scripts/diagnose_run.py runs/eval/base-q35-fast-n50
```

已归档的基座结果见[实验索引](experiments/README_zh.md)。

直接调用评测模块时，先启动检索服务：

```bash
uv run python -m search_r1.offline_eval backend=omlx omlx_model=Qwen3.5-4B-bf16 \
    eval_file=eval_sets/dev.jsonl limit_per_source=10 run_name=try
```

每次评测在 `runs/eval/<run_name>/` 下写出 `config.json`、`trajectories.jsonl`、`summary.json`。

## 测试

```bash
uv run --locked pytest tests/ -q
```

自动化测试覆盖普通与动态采样、训练与评测流程，以及检查点恢复。
