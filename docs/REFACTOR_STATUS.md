# REFACTOR STATUS

更新：2026-11-20。**已按用户最新要求暂停，勿自动继续第三步。** 完整范围见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)。

## 接续须知

- 当前工作分支：`codex/self-contained-refactor`。
- 用户最初授权阶段 1–3，随后明确要求“不要继续第三步，完成当前就暂停”。已保存当前代码和记录，第三步尚未验收；等待用户明确继续后再推进。
- 不运行训练、不做云端验证、不连接或修改云端服务/归档、不推送。
- 原 main=2c53127、codex/stage3-dapo=094bc2c、claude/r75-cap030=8165fec 必须保持；最终全量验收后才替换 main。
- 已删除的 handoff 和未跟踪 runs/local_docs 已由用户备份；这两份 tracked 文档应足以接续。
- `.codex-attachments/` 是用户未跟踪目录，不纳入提交。

## 阶段状态

| 阶段 | 状态 | 证据/下一步 |
|---|---|---|
| 1 合并与基线 | 完成 | 新分支已 ff 到 8165fec，含两个实验提交；55 passed |
| 2 父归档保护 | 完成 | owner=parent/child；未知归属跳过整索引；解析后的实际路径保护；19 passed |
| 3 公共组件/评测迁移 | 部分落地，已暂停 | 新组件已提取、调用方已调整；尚缺独立导入/行为对照/专项验收，不得标记完成 |
| 4 统一训练循环 | 未开始 | 本轮范围外 |
| 5 恢复与兼容 | 未开始 | 本轮范围外 |
| 6 云端脚本整合 | 未开始 | 仅计划本地实现/模拟 |
| 7 清除依赖/最终验收 | 未开始 | 全部完成后才替换 main |

## 验证记录

- 基线：`uv run --frozen pytest tests/` → **55 passed in 8.14s**，无已有失败。
- 环境：macOS arm64，Python 3.12.14，uv 0.13.9，pytest 9.2.0，tinker 0.39.0，cookbook 0.6.2.dev1+g1991897cb，transformers 5.23.0，torch 2.15.0。
- 阶段 2：`uv run --frozen pytest tests/test_checkpoint_pruning.py tests/test_cloud_operations.py` → **19 passed in 5.39s**；两个改动文件 ruff 通过。
- 删除测试仅操作 pytest 临时文件，覆盖父/子共享 URI、symlink 别名、未标归属旧父、同一步最后完整记录、sampler-only、dry-run、路径越界。
- 第三步当前中间态：`uv run --frozen pytest tests/` → **66 passed in 6.00s**（原 55 项 + 11 项清理测试）。这不是迁移专项验收；现有环境仍安装 cookbook。
- `import search_r1.offline_eval` 在当前环境成功；新 env/backends 及相关训练/评测入口、check_render 的 ruff 检查已通过。暂停收尾时 `git diff --check` 通过。

## 第三步已保存的中间态

- 原 search_env/search_tool_env/tools 移到 `search_r1/env/`，默认数据目录调整到项目根，保留 SEARCH_R1_DATA_DIR。
- 提取本地消息/轨迹类型、工具分发、消息环境、Qwen 文本 renderer、tokenizer 和单轨迹 rollout。
- 本地 completer 移到 `backends/`；评测改用本地组件及明确的 SDK create-client/load-state 调用，保留服务/训练客户端引用以覆盖采样生命周期。
- 更新脚本、测试和训练入口的模块路径。源码搜索显示 env/backends/offline_eval 已没有 cookbook import，但尚未做禁止 cookbook 导入的隔离验证。
- 新建 `training/cookbook_bridge.py`：只供仍使用旧循环的 train/train_dapo 调用，注册 renderer 并转换旧 runner 按类识别的 initial-overflow sentinel。训练仍依赖 cookbook；此桥须在阶段 4–5 删除。
- 许可和初始源文件映射位于 `third_party/tinker_cookbook/`。上游完整树的额外 NOTICE/归属核查尚未完成。

## 恢复工作时第三步的待办

1. 审核当前提取代码的边界、冗余和说明；renderer 中部分历史命名/多模态类型仍待整理，当前只支持文本搜索需要的模型格式。
2. 用固定离线 tokenizer/输入建立与 pinned cookbook 的 prompt tokens、stop、XML/JSON 工具解析、Unicode、多轮历史的对照 fixtures；最终回归测试不 import cookbook。
3. 增加 rollout 的正常作答、搜索、长度截断、结构/内容解析错误、初始及中途上下文溢出验证；现有奖励测试虽然通过，不能替代迁移对照。
4. 检查临时训练 bridge 在 initial-overflow 和真实旧 rollout 调用中的兼容性（仅 fake policy 本地测试，不运行训练）。
5. 增加 subprocess 导入阻断检查，确保离线评测/辅助脚本不间接导入 cookbook；验证 CLI 帮助和数据根目录覆盖。
6. 补齐 backend 的 fake SDK 调用/元数据回退测试、资源生命周期检查、完整 upstream NOTICE 核查。
7. 完成阶段 3 专项检查和状态记录后，仍按最新用户指令决定是否继续；阶段 4–7 均未开始。

## 提交与原分支保护

- `d84c473`：保存完整计划和状态文档（此前新分支已 ff 合入 094bc2c、8165fec）。
- `55101ee`：父归档保护及本地文件回归测试，阶段 2 完成。
- 后续 WIP 提交：保存上述第三步中间态和暂停记录，不能视为阶段 3 完成。
- 暂停收尾核对原分支仍为 main=`2c53127`、codex/stage3-dapo=`094bc2c`、claude/r75-cap030=`8165fec`。

## 已确定的迁移约定

- 新完整状态归属字段为 `owner="parent"` 或 `owner="child"`。当前旧训练仍不写 owner，因此现有无标记索引会保守跳过自动清理（包括无法证明是独立 run 的索引）；阶段 5 新写盘实现须补齐此字段。
- 清理先解析和验证全部路径再删除；保持原索引不变。不得凭 model ID 或 step 猜测归属。

## 接续入口

先读本文件和方案，检查 git status/分支。**当前暂停，须收到用户继续要求后再工作。** 不要重复合并实验提交，不要将 66 项测试通过或离线评测源码不再 import cookbook 误认为第三步已验收，更不要误认为已移除 cookbook 安装依赖。
