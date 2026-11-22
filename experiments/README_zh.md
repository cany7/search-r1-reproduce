# 实验归档

[English version](README.md)

## 实验列表

| 实验 | 目的 | 归档内容 |
|---|---|---|
| [本地基座](local-baselines/) | 使用 oMLX 比较模型与提示词配置 | 五组配置，各评测 631 道开发集题目；运行配置、评测汇总和逐题轨迹 |
| [云端验证](cloud-validation/README.md) | 验证部署、训练性能、并发和检查点恢复 | 训练与评测记录、耗时表及资源监控 |
| [主复现](reproduce-main/) | 两阶段强化学习训练：0 → 50 → 75，并延长至第 100 步进行对照 | 训练记录、12 组评测和选用的第 75 步检查点引用 |
| [动态采样](dynamic-sampling/) | 使用动态采样从第 100 步续训至第 105 步 | 训练记录、第 105 步 dev 评测和重启续训验证 |
| [轮数上限未作答惩罚](max-turns-penalty/) | 从第 75 步续训至第 100 步，将达到轮数上限仍未作答的奖励由 −0.1 调整为 −0.3 | 训练记录和第 100 步 dev/test 评测 |

`local-baselines/evaluations/` 下的目录名中，`q35` 表示 Qwen3.5-4B，`q3i` 表示 Qwen3-4B-Instruct-2507；`think` 表示开启思考模式，`fast` 和 `original` 表示提示词变体，`n100` 表示每个来源最多 100 题。

## 文件说明

- `config.json`：运行配置。
- `summary.json`：评测指标。
- `trajectories.jsonl`：逐题评测轨迹；较大的文件以 gzip 压缩（`.jsonl.gz`）。
- `metrics.jsonl`：训练指标。
- `checkpoints.jsonl`：检查点引用；已上传检查点的权重在 W&B 上（见 `uploads/*.json`）。
- `experiment.json`：实验元数据。
- [formal-environment.txt](formal-environment.txt)：主复现、动态采样和轮数上限未作答惩罚实验使用的 GPU、容器镜像及服务端／客户端版本。

## W&B

训练曲线、评测结果和检查点见 [W&B 工作区](https://forge.coreweave.com/wandb/timye7-personal/search-r1-reproduce?nw=sw0myi4atxa)。
