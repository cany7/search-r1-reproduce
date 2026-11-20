# search-r1-reproduce

使用 Qwen3.5、BM25 检索和 LoRA 小规模复现 Search-R1。普通采样和动态采样共用仓库内的训练循环，通过 Tinker SDK 向模型服务（如自部署 SkyRL）提交采样和优化器请求。

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
| Tinker SDK | 客户端锁文件固定为 0.39.0 |
| 迁入的 cookbook 源码 | 取自 `1991897`，代码及 Apache-2.0 许可随仓库提供，详见[来源映射](third_party/tinker_cookbook/UPSTREAM.md) |
| Search-R1（只取 `retrieval_server.py`） | `598e61b` |
| SkyRL（云端才用） | 固定 `79d4c89` 加本地补丁；A100 归档使用 padding 裁剪，当前部署另加按序列长度切换梯度检查点（尚无 GPU 验证），见[脚本说明](scripts/README.md)。客户端与服务端环境独立 |
| Python | 3.12.14，`pyenv local` 指定，uv 只用它 |
| pyserini | 2.4.0，需要 JDK 21 |

## 目录

可以按一次训练更新的顺序阅读代码。

| 环节 | 实现 |
|---|---|
| 参数入口 | [普通](search_r1/train.py)和[动态](search_r1/train_dynamic_sampling.py)入口共用一套[配置](search_r1/config.py)。 |
| 启动和恢复 | [runtime.py](search_r1/training/runtime.py) 锁定 run 并初始化 SDK 客户端。[checkpoints.py](search_r1/training/checkpoints.py) 选择完整状态，恢复优化器和下一题游标。 |
| 收集轨迹 | [loop.py](search_r1/training/loop.py) 通过 [sampling.py](search_r1/training/sampling.py) 收集题组。每条轨迹经过[多轮作答](search_r1/env/rollout.py)、[搜索环境](search_r1/env/search_tool_env.py)、[BM25 工具与评分](search_r1/env/tools.py)和 [Qwen renderer](search_r1/env/renderers/)。 |
| 更新和保存 | [data.py](search_r1/training/data.py) 用组内奖励减均值计算优势，对齐 action target 与旧 logprob。[update.py](search_r1/training/update.py) 提交梯度、执行 Adam 并发布采样权重。主循环写出轨迹与[指标](search_r1/training/logging.py)后提交检查点索引。 |
| 评测 | [offline_eval.py](search_r1/offline_eval.py) 通过 [Tinker 或 oMLX 后端](search_r1/backends/)复用同一套环境和多轮作答逻辑。固定的[开发集和测试集](eval_sets/)随仓库提供。 |

[检索服务](retrieval/)使用独立环境。[脚本指南](scripts/README.md)说明部署和分析工具的用法。Git 忽略本地数据及运行输出。

训练和基座评测默认使用 `search_r1/qwen3_5_disable_thinking_unicode`，仅修正列表／字典
工具参数中的非 ASCII 字符转义，沿用 cookbook 的消息渲染、解析和 rollout 逻辑。
renderer 名称写入训练／评测配置和训练客户端元数据；检查点评测优先使用记录的名称。
原 `qwen3_5_disable_thinking` 仍可用于历史运行。

## 数据

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

## 环境

```bash
uv sync --locked                 # 主环境
(cd retrieval && uv sync --locked)  # 检索环境
cp .env.example .env             # 首次配置；本地评测填 OMLX_API_KEY，训练填 WANDB_API_KEY
```

训练默认开启 W&B，项目名为 `search-r1-reproduce`，run 名自动包含模型、batch、学习率和时间。
训练入口自动读取项目根目录的 `.env`，已有环境变量优先。
可用 `wandb_project=<项目名>` 改名，或 `wandb_project=None` 关闭上传；本地日志仍会保存。

## 四种训练配方

先启动 BM25，配置 `TINKER_BASE_URL` 和 `TINKER_API_KEY`，再从项目根目录运行。自部署 Pod 的配置见[脚本指南](scripts/README.md)。四个示例每批选 64 道题，每题采样 8 条轨迹。普通 CLI 默认每批 512 题。可追加 `max_steps`、`save_every`（默认 25）和 `log_path`，分别指定全局终点、保存间隔和运行目录。

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

最后一轮有效回答沿用基础评分。评测继承快照里的搜索规则和基础评分，封顶未答惩罚仅用于训练。动态采样在同一策略下补齐整批后才更新。普通采样另支持流式 minibatch。

两个入口均可从基座起跑。从普通 run 分叉时传 `parent_run=路径 parent_step=N` 和新的 `log_path`。同一 run 续训时复用原目录与配方，传 `behavior_if_log_dir_exists=resume`。两者都恢复优化器。仅加载权重时用 `load_checkpoint_path=URI`。

在同步的 Pod 项目副本中运行 `scripts/cloud/train_with_dev_eval.sh RUN [key=value ...]`，可在分段训练之间评测开发集。单独评测用 `scripts/cloud/eval_checkpoint.sh RUN STEP dev`，测试集用 `test`。驱动读取指定步的完整状态和 run 的配置。基座评测、跨配置评测和上传选项见[脚本指南](scripts/README.md)。

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
uv run --locked pytest tests/ -q
```

测试使用固定参考轨迹、本地文件和模拟的 SDK／Pod 进程，覆盖两种采样、检查点恢复，以及未安装 cookbook 时的离线 CLI 帮助。重构后的 GPU 训练和在线服务行为未经实测。详见[验收记录](docs/REFACTOR_STATUS.md)。
