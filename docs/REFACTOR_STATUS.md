# REFACTOR STATUS

更新：2026-11-20。**阶段 1–3 已完成；本次在第三步验收后停止，阶段 4–7 尚未开始。** 原完整方案保留在 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)。按用户最新要求，进度、验收说明和接续记录统一维护在本文，不再另建说明文档。

## 接续须知

- 当前工作分支：`codex/self-contained-refactor`。
- 用户在 WIP `805f753` 暂停后明确要求“继续完成第三步的剩余内容”，此次已完成。下一次只有用户授权继续才开始第四步。
- 不运行训练、不做云端验证、不连接或修改云端服务/归档、不推送。实验已结束，未来实际复现的问题另行处理。
- 原 main=`2c53127`、codex/stage3-dapo=`094bc2c`、claude/r75-cap030=`8165fec` 保持不变；最终全量验收后才替换 main。
- 已删除的 handoff 和未跟踪 runs/local_docs 已由用户备份；接续不依赖它们。
- `.codex-attachments/` 是用户未跟踪目录，不纳入提交、不修改。

## 阶段状态

| 阶段 | 状态 | 证据/下一步 |
|---|---|---|
| 1 合并与基线 | 完成 | 新分支 ff 到 8165fec，含两个实验提交；55 passed |
| 2 父归档保护 | 完成 | owner=parent/child；未知归属跳过整索引；解析后的实际路径保护；19 passed |
| 3 公共组件/评测迁移 | 完成 | 7 renderer、12 类轨迹对照；导入/网络阻断；SDK 模拟与生命周期；全套 130 passed |
| 4 统一训练循环 | 未开始 | 下一阶段：训练数据、优势、更新、普通/动态循环及必要 streaming |
| 5 恢复与兼容 | 未开始 | checkpoint/游标、通用分叉、W&B、写盘顺序与旧配置 |
| 6 云端脚本整合 | 未开始 | 仅本地实现/模拟，包括旧 SkyRL 基座模式 |
| 7 清除依赖/最终验收 | 未开始 | 干净环境不安装 cookbook；最终整体文档；全量验收后才替换 main |

## 验证记录

- 基线：`uv run --frozen pytest tests/` → **55 passed in 8.14s**。
- 环境：macOS arm64，Python 3.12.14，uv 0.13.9，pytest 9.2.0，tinker 0.39.0，cookbook 0.6.2.dev1+g1991897cb，transformers 5.23.0，torch 2.15.0。
- 阶段 2：checkpoint pruning + cloud operations → **19 passed in 5.39s**；ruff 通过。删除测试仅操作 pytest 临时文件。
- 第三步暂停中间态：**66 passed in 6.00s**，当时尚无迁移专项验收。
- 第三步完成：`uv run --frozen pytest tests/ -q` → **130 passed in 17.21s**。新增 64 项，包含独立子进程中的导入阻断与网络阻断回归。
- 参考 fixture 来自原项目 `8165fec2edbd0fc569375ae3d6a21043ddc537c3` + cookbook `1991897cba59ea9d9b0030d183d1867b28eba9d9`，不是从迁移后的实现生成。原项目通过 git archive 解到临时目录，没有切换或移动原分支。
- 源文件与许可核对：13 个提取源文件在安装包和 pinned Git 树中逐字节一致；LICENSE 一致，该树无 NOTICE。来源映射见 `third_party/tinker_cookbook/UPSTREAM.md`。
- 迁入代码、新测试及相关调用方 ruff、`git diff --check` 通过。未执行实际训练、模型推理、云端或检索请求。

## 第三步交付内容

- 原 search_env/search_tool_env/tools 移至 `search_r1/env/`；local_completer 移至 `backends/`；脚本、测试和训练调用方均更新路径。默认数据根回到项目根，并验证 SEARCH_R1_DATA_DIR 的临时 parquet 读取。
- 本地消息/轨迹类型、工具分发、消息环境、Qwen 文本 renderer、tokenizer 和单轨迹 rollout。保留历史 renderer 名及 Unicode 工具参数；明确拒绝多模态输入和本地 runner 不支持的通用 budgets。
- 评测改用本地组件、显式 SDK create/load/采样与本地 checkpoint 元数据读取，不再依赖 cookbook import 或评测用 SDK monkey patch。
- 评测客户端生命周期完整覆盖成功、失败、取消；取消时先结束所有轨迹任务，再关闭检索/SDK。关闭异常记录 warning，保留原始错误/结果。
- 固定对照覆盖 prompt token hash/count、stop、XML/JSON/Unicode、thinking 和多轮工具响应，以及正常作答、搜索限制、封顶、截断、结构/内容错误、参数错误、上下文溢出。
- 在全新子进程中禁止所有 cookbook import 和 socket connect：env/backends/eval 与 check_render、diagnose_run、make_eval_sets 可导入；组件/backend/tokenizer/reward 专项回归通过；评测 CLI 帮助可输出。
- `training/cookbook_bridge.py` 仍是训练专用临时适配：注册本地 renderer 并转换 initial-overflow sentinel。通过真实旧单轨迹 runner + fake policy 验证；不是训练验证。

## 必须保留的边界

- **训练尚未摆脱 cookbook**：train.py、train_dapo.py、wandb_compat.py、training/cookbook_bridge.py 仍 import 上游；相关训练/W&B/bridge 测试也仍依赖它。当前不能删除 pyproject/lock 中的安装依赖。
- 本地 runner 未迁入 cookbook 全部 hooks、通用预算、流式响应解析、多模态等框架功能。第四步的普通训练 minibatch streaming 须独立迁移，不要误认为已完成。
- 固定离线 tokenizer 验证格式与 token 边界的迁移等价性，未验证真实 Qwen tokenizer、GPU 或服务运行。用户不要求云端验收，此限制不是第三步待办。
- 导入阻断发生在当前依赖齐全的环境，完全不安装 cookbook 的环境及直接依赖声明属于第七步。
- 固定 chz CLI 的 `--help` 原有退出码为 1；本次保留，隔离测试检查其帮助内容和空 stderr。
- 旧 SkyRL 基座评测的活跃 LoRA 客户端模式留在第六步；当前基座分支保留直接 SDK base_model 采样。

## 第三步后的职责关系

| 部分 | 当前实现位置与责任 |
|---|---|
| 评测入口 | `offline_eval.py`：配置、题集、并发、轨迹记录/汇总、资源生命周期 |
| 单条轨迹循环 | `env/rollout.py`：采样 → 环境 step → 判断结束，完全在本仓库 |
| 消息与搜索环境 | `env/message_env.py` 处理 token/消息适配和上下文限制；`agent_tool_env.py` 管理历史、工具与终局评分；`search_tool_env.py` 实现项目搜索限制与训练专用惩罚 |
| 格式与工具 | `env/renderers/` 负责 Qwen 文本格式和解析；`tool_dispatch.py` 验证/分发工具参数；`tools.py` 实现 BM25 与评分 |
| SDK 交互 | `backends/completers.py` 采样，`tinker_backend.py` 显式加载状态/读取元数据/关闭会话；SDK 负责客户端协议与向服务发送请求 |
| 训练过渡层 | 普通/动态入口仍使用 cookbook 的训练循环、数据整理、更新、checkpoint 和日志；通过 `training/cookbook_bridge.py` 使用本地环境，第四、五步再移除 |

## 固定对照 fixture 的来源与使用

- `tests/fixtures/cookbook_reference.json` 来自原项目 `8165fec` 和 pinned cookbook；输入与离线 tokenizer 定义在 `tests/migration_cases.py`。正常 pytest 只读 JSON，不能为了通过测试而从新实现重新生成期望结果。
- 固定 tokenizer 将普通 Unicode 字符映射为 `ord(character) + 1000`，五个 Qwen 特殊 token 使用专用 ID；模拟 backend 保留 Qwen3 区分内容内特殊 token 文本的行为。每个 prompt 保存文本、token 数及 `json.dumps(token_ids)` 的 SHA-256，避免文本相同但 token 不同的差异被漏掉。
- 7 renderer 的对照输入包括初始提示、多轮历史、prefill、连续工具响应、Unicode 参数、thinking 历史及 XML/JSON/错误/未闭合调用、缺少/重复停止 token。
- 12 类完整轨迹额外对照采样 tokens/logprobs、工具参数与结果、reward、metrics/logs、消息历史、最终 observation；覆盖上下文上限处正常回答仍保留真实奖励。
- `tests/test_component_migration.py`、`test_eval_backends.py`、`test_tokenizer_loading.py`、`test_reward.py` 可在 cookbook 导入/网络阻断下运行；阻断器与独立 CLI/数据目录检查在 `test_eval_isolation.py`。`test_cookbook_bridge.py` 专门保留旧 runner 集成验证，未来随 bridge 删除。
- 如需有意更新参考结果：先 `git archive 8165fec` 到临时目录，在进程导入任何 search_r1 模块前将临时目录放到 sys.path 首位，并加入当前 `tests/`。从旧 `search_r1.renderers/search_tool_env/tools` 及 pinned cookbook 取得 renderer、环境、工具、token result、do_single_rollout；分别调用共享 `renderer_snapshot` / `rollout_snapshot`，记录完整上游和项目 commit。应使用独立环境并人工审查 fixture 差异，不在常规测试中自动重建。
- 上述编码器用于验证迁移等价性，不是实际 Qwen tokenizer；没有下载模型文件。所有采样/检索结果均为固定本地数据。

## 下一次从第四步开始

1. 先读总计划与本文，核对 git status 和原三个分支指针。不要重复合并实验提交。
2. 迁入组内减均值优势、轨迹转 datum、action token mask、旧 logprob 对齐、loss/Adam 参数和发布采样权重的逻辑；以固定数据/fake SDK 测试对照。
3. 统一普通和动态训练主循环，普通取固定题组；动态组内全同筛除、同策略新题补齐、游标计入超额题组、不完整 batch 不更新。保留普通 minibatch streaming 并明确动态不支持组合。
4. 根据迁移进度删除训练 bridge 和专用 bridge 测试。阶段 5 再统一恢复/分叉/W&B 与旧配置；不要提前宣称 cookbook 已完全删除。
5. 继续遵循 no training/no cloud 的授权边界。其余验收和目标结构见总计划，不从历史 runs 推断新行为。

## 提交与原分支保护

- `d84c473`：完整计划和状态文档；新分支此前已 ff 合入 094bc2c、8165fec。
- `55101ee`：父归档保护及本地文件回归测试，阶段 2 完成。
- `805f753`：第三步中间态及暂停记录（WIP）。
- 本次完成提交：`Complete stage 3 local component migration and offline validation`；具体 hash 见当前分支 git log。
- 本次核对原分支仍为 main=`2c53127`、codex/stage3-dapo=`094bc2c`、claude/r75-cap030=`8165fec`。

## 已确定的检查点迁移约定

- 新完整状态归属字段为 `owner="parent"` 或 `owner="child"`。当前旧训练仍不写 owner，因此现有无标记索引会保守跳过自动清理（包括无法证明独立的 run）；阶段 5 新写盘实现须补齐此字段。
- 清理先解析和验证全部路径再删除；保持原索引不变。不得凭 model ID 或 step 猜测归属。
