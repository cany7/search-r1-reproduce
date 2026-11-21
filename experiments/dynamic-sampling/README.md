# 动态采样实验

从主复现延长训练的第 100 步出发，过滤组内奖励相同的题组并补采，正式训练完成到第 105 步。该 checkpoint 使用限制搜索规则评测 dev。

[正式训练](training/)、[重启续训 smoke test](smoke/)和[第 105 步 dev 评测](evaluations/limited-search/dev/step-0105/)分别归档。Smoke run 到第 102 步后重启，续训至第 103 步。

[文件清单](files.json)记录归档路径、原路径和大小。
