# 延长训练（第 76–100 步）

此阶段从第 75 步续训到第 100 步，沿用阶段二的搜索规则和训练配置。

| 归档文件 | 原路径 | 归档范围 |
|---|---|---|
| [指标](metrics.jsonl) | `runs/formal-2026-11-17-s2/metrics.jsonl` | `step=75…99`，对应更新 76–100 |
| [检查点索引](checkpoints.jsonl) | `runs/formal-2026-11-17-s2/checkpoints.jsonl` | `batch=80…100`，包含 `batch=100, name=final` |
| [上传记录](uploads/step0100.json) | `runs/formal-2026-11-17-s2/uploads/step0100.json` | 第 100 步的权重引用与上传信息 |

本阶段的原始 rollout 对应 `iteration_000075` 至 `iteration_000099`。原指标末行 `step=100` 是运行总结，第 100 次更新的指标记录在 `step=99`。
