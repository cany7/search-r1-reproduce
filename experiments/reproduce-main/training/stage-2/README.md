# 阶段二（第 51–75 步）

此阶段从第 50 步续训到第 75 步。每轮只执行第一个搜索调用中的前 3 条合法查询。

| 归档文件 | 原路径 | 归档范围 |
|---|---|---|
| [指标](metrics.jsonl) | `runs/formal-2026-11-17-s2/metrics.jsonl` | `step=50…74`，对应更新 51–75 |
| [检查点索引](checkpoints.jsonl) | `runs/formal-2026-11-17-s2/checkpoints.jsonl` | `batch=55…75`，包含最终选用的 `batch=75, name=final` |
| [上传记录](uploads/step0075.json) | `runs/formal-2026-11-17-s2/uploads/step0075.json` | 第 75 步的权重引用与上传信息 |

阶段二和延长训练共用 `runs/formal-2026-11-17-s2/config.json`。本阶段的原始 rollout 对应 `iteration_000050` 至 `iteration_000074`。
