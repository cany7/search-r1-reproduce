# REFACTOR STATUS

更新：2026-11-20。完整范围见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)。

## 接续须知

- 当前工作分支：`codex/self-contained-refactor`。
- 用户本轮授权阶段 1–3；后续阶段尚未开始。
- 不运行训练、不做云端验证、不连接或修改云端服务/归档、不推送。
- 原 main=2c53127、codex/stage3-dapo=094bc2c、claude/r75-cap030=8165fec 必须保持；最终全量验收后才替换 main。
- 已删除的 handoff 和未跟踪 runs/local_docs 已由用户备份；这两份 tracked 文档应足以接续。
- `.codex-attachments/` 是用户未跟踪目录，不纳入提交。

## 阶段状态

| 阶段 | 状态 | 证据/下一步 |
|---|---|---|
| 1 合并与基线 | 完成 | 新分支已 ff 到 8165fec，含两个实验提交；55 passed |
| 2 父归档保护 | 完成 | owner=parent/child；未知归属跳过整索引；解析后的实际路径保护；19 passed |
| 3 公共组件/评测迁移 | 进行中 | 离线评测导入链移除 cookbook，保持训练过渡兼容 |
| 4 统一训练循环 | 未开始 | 本轮范围外 |
| 5 恢复与兼容 | 未开始 | 本轮范围外 |
| 6 云端脚本整合 | 未开始 | 仅计划本地实现/模拟 |
| 7 清除依赖/最终验收 | 未开始 | 全部完成后才替换 main |

## 验证记录

- 基线：`uv run --frozen pytest tests/` → **55 passed in 8.14s**，无已有失败。
- 环境：macOS arm64，Python 3.12.14，uv 0.13.9，pytest 9.2.0，tinker 0.39.0，cookbook 0.6.2.dev1+g1991897cb，transformers 5.23.0，torch 2.15.0。
- 阶段 2：`uv run --frozen pytest tests/test_checkpoint_pruning.py tests/test_cloud_operations.py` → **19 passed in 5.39s**；两个改动文件 ruff 通过。
- 删除测试仅操作 pytest 临时文件，覆盖父/子共享 URI、symlink 别名、未标归属旧父、同一步最后完整记录、sampler-only、dry-run、路径越界。

## 已确定的迁移约定

- 新完整状态归属字段为 `owner="parent"` 或 `owner="child"`。当前旧训练仍不写 owner，因此现有无标记索引会保守跳过自动清理（包括无法证明是独立 run 的索引）；阶段 5 新写盘实现须补齐此字段。
- 清理先解析和验证全部路径再删除；保持原索引不变。不得凭 model ID 或 step 猜测归属。

## 接续入口

先读本文件和方案，检查 git status/分支，再根据最新阶段记录继续。不要重复合并已包含的实验提交，不要将阶段 3 完成误认为已完全移除 cookbook 安装依赖。
