# 五轮封顶未作答惩罚实验

从主复现的第 75 步出发，把五轮用尽仍未作答的终局奖励从 −0.1 调为 −0.3，训练至第 100 步。对照组是主复现使用原奖励的[延长训练](../reproduce-main/training/extended-training/)。

[训练记录](training/)及第 100 步的 [dev](evaluations/limited-search/dev/step-0100/) 和 [test](evaluations/limited-search/test/step-0100/) 评测均已归档。限制搜索规则下，test 答对 849/1,894 题（EM 44.83%），原奖励的延长训练答对 876/1,894 题（EM 46.25%）。

[文件清单](files.json)记录归档路径、原路径和大小。
