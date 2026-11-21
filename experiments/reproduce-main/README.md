# Search-R1 主复现

最终模型选用**阶段二第 75 步**的 checkpoint。第 75 步在新版搜索规则的 test 上答对 892/1,894 题（EM 47.10%），延长训练到第 100 步后答对 876 题（46.25%）。

| 训练目录 | 起点 → 终点 | 用途 |
|---|---|---|
| [阶段一](training/stage-1/) | 基座 → 50 | 原搜索规则下的 RL 训练 |
| [阶段二](training/stage-2/) | 50 → 75 | 每轮最多执行 3 条合法查询，产出最终 checkpoint |
| [延长训练](training/extended-training/) | 75 → 100 | 沿用阶段二配置，观察继续训练的效果 |

评测按[原搜索规则](evaluations/original-search/)和[限制搜索规则](evaluations/limited-search/)归档，各自再按 dev/test 和 checkpoint 分目录。限制搜索规则每轮只执行第一个搜索调用中的前 3 条合法查询。

| 评测规则 | Dev | Test |
|---|---|---|
| 原搜索规则 | [基座](evaluations/original-search/dev/base/)、[第 25 步](evaluations/original-search/dev/step-0025/)、[第 50 步](evaluations/original-search/dev/step-0050/) | [基座](evaluations/original-search/test/base/)、[第 25 步](evaluations/original-search/test/step-0025/)、[第 50 步](evaluations/original-search/test/step-0050/) |
| 限制搜索规则 | [第 25 步](evaluations/limited-search/dev/step-0025/)、[第 50 步](evaluations/limited-search/dev/step-0050/)、[第 75 步](evaluations/limited-search/dev/step-0075/)、[第 100 步](evaluations/limited-search/dev/step-0100/) | [第 75 步](evaluations/limited-search/test/step-0075/)、[第 100 步](evaluations/limited-search/test/step-0100/) |

阶段一归档包含配置、指标、检查点索引和监控数据。原搜索规则第 50 步 test 的仓库记录包含配置和汇总。

[实验元数据](experiment.json)记录阶段筛选区间，[文件清单](files.json)记录原样复制的文件、原路径和大小。
