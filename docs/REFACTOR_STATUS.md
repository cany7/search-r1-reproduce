# REFACTOR STATUS

更新：2026-11-20。**阶段 1–5 已完成。下一步是阶段 6 的云端驱动与评测脚本整合，只做本地模拟。** 原完整方案保留在 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)。按用户最新要求，进度、验收说明和接续记录统一维护在本文，不再另建说明文档。

## 接续须知

- 当前工作分支：`codex/self-contained-refactor`。
- 用户本次授权“继续完成第五步”。本次交付起点、分叉与恢复协议。阶段 6–7 留待后续。
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
| 4 统一训练循环 | 完成 | 本地主循环、数据/优势/更新、动态补齐、普通 streaming；固定上游对照 + fake SDK + 导入阻断 |
| 5 恢复与兼容 | 完成 | 指定父步、优化器与游标、归属与锁、W&B 身份、保存失败重放和旧格式回归 |
| 6 云端脚本整合 | 未开始 | 仅本地实现/模拟，包括旧 SkyRL 基座模式 |
| 7 清除依赖/最终验收 | 未开始 | 干净环境不安装 cookbook；最终整体文档；全量验收后才替换 main |

## 验证记录

- 基线：`uv run --frozen pytest tests/` → **55 passed in 8.14s**。
- 环境：macOS arm64，Python 3.12.14，uv 0.13.9，pytest 9.2.0，tinker 0.39.0，cookbook 0.6.2.dev1+g1991897cb，transformers 5.23.0，torch 2.15.0。
- 阶段 2：checkpoint pruning + cloud operations → **19 passed in 5.39s**；ruff 通过。删除测试仅操作 pytest 临时文件。
- 第三步暂停中间态：**66 passed in 6.00s**，当时尚无迁移专项验收。
- 第三步完成：`uv run --frozen pytest tests/ -q` → **130 passed in 17.21s**。新增 64 项，包含独立子进程中的导入阻断与网络阻断回归。
- 第四步完整测试：`uv run --frozen pytest tests/ -q` → **176 passed in 19.26s**（删除 4 项 bridge 测试，新增 50 项训练迁移测试）；迁入代码、入口、相关测试 ruff 和 `git diff --check` 通过。
- 第五步完整测试：`uv run --offline --frozen pytest tests/ -q` → **209 passed in 15.96s**。其中包括独立进程的 cookbook 导入和网络连接阻断。改动的 Python 文件通过 ruff，暂存差异通过 `git diff --cached --check`。
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
- 第三步曾保留 `training/cookbook_bridge.py` 训练适配并验证旧 runner。第四步已删除该文件和 4 项过渡测试，训练直接使用本地环境、类型和 rollout。

## 第四步交付内容

- `config.py` 保存普通/动态共用 CLI 参数；`training/config.py` 保存有效训练配置和支持范围校验。新增 `train_dynamic_sampling.py`，与 `train.py` 一起进入本地公共主循环。普通默认 batch_size=512，四类配方参数和历史 renderer 名均保留。
- `training/data.py`：组内 reward 减均值，不除标准差；合并前缀连续的多轮轨迹，非前缀另建 datum；左移 target、旧 logprob、action mask、parse-error mask 和 provenance 对齐。
- `training/update.py`：直接调用 SDK `forward_backward` / `optim_step` / `save_weights_and_get_sampling_client`。保留 importance_sampling 默认及 loss 配置透传，Adam beta1=0.9、beta2=0.95、eps=1e-8；发给 SDK 前移除本地 mask 元数据。普通非 streaming 按上游 linspace 规则拆分 substeps，更新后发布 snapshot。
- `training/sampling.py`：本地 fail-fast 组 rollout、组评分与 cleanup；原动态 Batch/collect_batch 从 train_dapo 移入。动态组内全同剔除，同一次更新内使用同一策略补新题；超额题组计入已消费游标，不跨更新复用；耗尽或达到 rounds 上限不满整批时抛错，不调用 optimizer、不提交 checkpoint。
- `training/loop.py`：普通固定题组、动态补齐及普通 streaming 共用更新后的指标、轨迹和检查点边界。每次迭代单独作用域释放轨迹/训练 datum；任务失败或取消会取消并等待剩余 rollout，避免后台采样遗留。
- 普通 streaming 依完成顺序收组，每够一个 minibatch 即提交 forward_backward；每个 substep 完成所有 minibatch 后只做一次 Adam。整轮 rollout 始终持有旧 snapshot，整轮更新后才发布下一轮策略。动态配置在创建服务前拒绝 streaming、多 substeps 和多 epochs。
- `training/metrics.py` / `records.py` / `usage.py` / `utils.py` 保存必要的指标、旧 JSONL schema、token 计数和辅助函数。来源映射、修改范围与 Apache-2.0 头已补充至现有 `third_party/tinker_cookbook/UPSTREAM.md`。
- `train_dapo.py` 保留旧阶段三的父状态加载/CLI 协议，但它的训练部分也调用相同 `run_updates`；不再有独立更新循环。删除 training/cookbook_bridge.py 及专用 bridge 测试。

## 第四步验收时的边界（恢复与日志部分已在第五步替换）

- **训练核心已脱离 cookbook，整个运行时尚未完全脱离。** `training/legacy_runtime.py` 和旧 `train_dapo.py` 暂留 checkpoint、日志、价格查询的上游辅助调用；`train.py` 真正启动时仍使用旧 logdir 行为；`wandb_compat.py` 和 SkyRL 创建客户端 patch 留到第五步处理。没有任何本地计算模块或训练入口再调用 cookbook 的 RL train/data/rollout。
- 新普通/动态 CLI 帮助、本地配置/数学/主循环可在 cookbook import 和 socket connect 被阻断时运行。真实启动仍有上述辅助依赖，不能删除 pyproject/uv.lock 的 cookbook 依赖；完全不安装 cookbook 的干净环境验收仍属第七步。
- `train_dynamic_sampling` 当前支持基座起跑和自己的游标恢复。通用 parent_run/parent_step、精确父步选择、owner 标记、完整旧 schema 兼容、普通 run 锁和 W&B 身份统一尚属第五步；旧 train_dapo 的 step100/64×8/seed 等限制暂留在其兼容加载入口，不存在于新公共训练循环。
- 暂保留旧 JSON 配置字段供读取；未迁移的 async/off-policy、KL penalty/post-KL、通用 termination/retry、rolling checkpoint、HTML/trace 图表等框架功能明确拒绝启用。项目没有内置 evaluator；评测继续使用独立 offline_eval。旧 num_groups_to_log 字段保留但不生成 cookbook HTML logtree/终端样例；结构化 JSONL 和环境/token/KL 指标保留。
- 无 action target 的 overflow/全 masked 数据不发给 SDK；整批没有可训练 datum 时记录 optimizer_steps=0，继续消费普通 batch。普通同步“过滤全恒定组时保留一组”和普通 streaming“全部丢弃”的旧差异仍保留，已分别测试。动态不足整批始终中止，不套用普通模式的行为。
- 为统一写盘边界，普通模式现在也先完成更新/发布、发布 rollout JSONL、写 metrics，最后提交完整 checkpoint。普通周期保存暂时会额外导出 checkpoint 的 sampler 文件（已有新策略 snapshot）；第五步可在新的 checkpoint 实现中整理。最终步只保存 final，避免同一步重复 periodic + final。
- 固定离线 tokenizer / token 轨迹和 fake SDK 仅证明格式、数据和调度逻辑；未验证真实 Qwen tokenizer、GPU 数值、SDK 服务或云端运行。用户不要求这些验证，不能把它们列为待索取云端权限的阻塞项。
- chz 的 `--help` 仍沿用退出码 1。旧 SkyRL 基座评测的活跃 LoRA 客户端模式留在第六步。

## 第四步后的职责关系

| 部分 | 当前实现位置与责任 |
|---|---|
| CLI / 配置 | train.py、train_dynamic_sampling.py、config.py、training/config.py：参数 → builder → 有效配置；旧 train_dapo 是兼容入口 |
| 外层训练循环 | training/loop.py：每轮收集、更新、发布、指标/轨迹、检查点时机，普通和动态共用 |
| 题组采样 | training/sampling.py：组 rollout/评分；动态筛除和补齐；普通 streaming 在 loop 中按完成顺序交付 minibatch |
| 训练数学和更新 | training/data.py：优势、datum/mask/logprob；update.py：SDK 梯度/Adam/权重发布 |
| 单条轨迹 / 环境 | env/rollout.py、message_env.py、agent_tool_env.py、search_tool_env.py：多轮采样、搜索规则、终止与奖励 |
| 格式 / 工具 | env/renderers/、tool_dispatch.py、tools.py：Qwen 格式、解析、BM25 与评分 |
| SDK 交互 | backends/completers.py 发采样请求；training/update.py 发更新/发布请求；实际模型计算仍由 SDK 所连接的服务执行 |
| 恢复 / 日志过渡层 | training/legacy_runtime.py、train_dapo.py、wandb_compat.py：暂用 cookbook 辅助设施，第五步统一 |
| 离线评测 | offline_eval.py 与 backends/tinker_backend.py：已在第三步独立迁移，第四步未改评测流程 |

## 固定对照 fixture 的来源与使用

- `tests/fixtures/cookbook_reference.json` 来自原项目 `8165fec` 和 pinned cookbook；输入与离线 tokenizer 定义在 `tests/migration_cases.py`。正常 pytest 只读 JSON，不能为了通过测试而从新实现重新生成期望结果。
- 固定 tokenizer 将普通 Unicode 字符映射为 `ord(character) + 1000`，五个 Qwen 特殊 token 使用专用 ID；模拟 backend 保留 Qwen3 区分内容内特殊 token 文本的行为。每个 prompt 保存文本、token 数及 `json.dumps(token_ids)` 的 SHA-256，避免文本相同但 token 不同的差异被漏掉。
- 7 renderer 的对照输入包括初始提示、多轮历史、prefill、连续工具响应、Unicode 参数、thinking 历史及 XML/JSON/错误/未闭合调用、缺少/重复停止 token。
- 12 类完整轨迹额外对照采样 tokens/logprobs、工具参数与结果、reward、metrics/logs、消息历史、最终 observation；覆盖上下文上限处正常回答仍保留真实奖励。
- `tests/test_component_migration.py`、`test_eval_backends.py`、`test_tokenizer_loading.py`、`test_reward.py` 可在 cookbook 导入/网络阻断下运行；阻断器与独立 CLI/数据目录检查在 `test_eval_isolation.py`。旧 `test_cookbook_bridge.py` 已在第四步随 bridge 删除。
- 如需有意更新参考结果：先 `git archive 8165fec` 到临时目录，在进程导入任何 search_r1 模块前将临时目录放到 sys.path 首位，并加入当前 `tests/`。从旧 `search_r1.renderers/search_tool_env/tools` 及 pinned cookbook 取得 renderer、环境、工具、token result、do_single_rollout；分别调用共享 `renderer_snapshot` / `rollout_snapshot`，记录完整上游和项目 commit。应使用独立环境并人工审查 fixture 差异，不在常规测试中自动重建。
- 上述编码器用于验证迁移等价性，不是实际 Qwen tokenizer；没有下载模型文件。所有采样/检索结果均为固定本地数据。

## 第四步参考与本地验收

- `tests/fixtures/training_reference.json` 来自 pinned cookbook 的真实 data_processing、metric_util、metrics、rollout_logging 与 `_train_step_and_publish`，使用 `tests/training_cases.py` 的固定 token 轨迹；没有调用服务或真实训练。输入涵盖部分奖励、恒定组、多轮前缀/工具 observation、非前缀 reset、masked turn、完整/部分 provenance、overflow、transition + final reward。
- fixture 保存完整优势、datum、metadata、环境/KL 指标、JSONL records，以及 1/3 substeps 的 fake SDK 提交顺序、loss 配置、Adam 参数和旧 logprob。常规 pytest 只读这些期望结果；不能从迁移后实现重新生成 golden。
- 如需更新：在 pinned cookbook 环境导入上述旧函数，对 `training_cases.cases()` 运行 compute_advantages → assemble_training_data、指标和 records；对 centered_partial_rewards 用仅返回本地 Ready/ForwardBackwardOutput 的假客户端执行旧 `_train_step_and_publish`，保存提交参数。非 JSON tensor/ndarray 以 tolist 序列化，记录源 commit。此过程只运行本地数学和假客户端。
- 新增 fake SDK 验证：普通/动态相同策略边界、补齐/筛除/超额游标、尾批无更新、streaming 首个 forward 先于剩余 rollout 完成、多 substeps 的 Adam 次数、loss 配置、零 action、失败/取消排空任务、错误后不提交完整状态、组评分/cleanup、四类配方和非法组合。
- 独立子进程同时禁止 cookbook import 和 socket connect，运行训练核心全部 fake 回归；普通/动态 CLI 的帮助也在阻断下验证。暂留运行时另用 fake ServiceClient/真实本地 checkpoint manager 验证“基座 → 更新 → 本 run 恢复”，并验证共享检索客户端关闭。旧 train_dapo 恢复/正式 run 测试继续通过。
- 第四步先完成全套 **172 passed in 15.62s**；随后补充普通/动态启动恢复与固定上游 SDK 合约对照，最终全套 **176 passed in 19.26s**（见上方验证记录）。
- 新迁入的 9 个上游源文件已与本地 pinned checkout 逐字节比对一致（含之前已核对的共享源）；没有额外版权/NOTICE 要求。项目中仅保留需要的函数和本地编排，没有复制整个 RL 框架。

## 第五步交付与验收

- 普通与动态 CLI 共用 `training/runtime.py`。不指定父 run 时从 step 0、游标 0 新建客户端。指定 `parent_run` 与 `parent_step` 从普通父 run 的该步分叉。同一步多条记录选最后一条带完整状态的记录，sampler-only 记录跳过。`max_steps` 始终使用全局步数。普通分叉须保持 batch_size，动态子 run 可改 batch_size。动态父 run 只支持在本 run 续跑。
- 新 run 的 `config.json` 保存首次有效配置及按实际题目内容与顺序计算的 SHA-256。分叉快照同时固定父目录、步数、状态路径、游标与累计 token。再次启动可省略两个父参数，父目录移走后仍可使用固定起点或本 run 的最新完整状态。显式更换已绑定的父目录或步数会报错。
- 普通子 run 在自己的索引一次性登记 `owner="parent"` 的记录。动态子 run 直接读父状态。两者的新保存均为 `owner="child"`。旧记录保留原归属，阶段 2 对未知归属跳过清理的规则仍生效。
- 历史配置按保存的模型、LoRA rank、renderer、tokenizer 与 seed 核对，缺 `limit_search` 时沿用不限搜索。旧普通记录缺游标时用父 batch_size 和步数还原，跨 epoch 沿用整批边界。旧动态记录必须有有效的 `next_prompt_offset`，并检查已有 `stage3_run` 与 `parent_state_path`。旧快照没有数据指纹，无法追溯当时数据文件的实际内容或顺序。读取旧配置和完整索引行保留原字节。
- SDK 创建和恢复显式走 backend。基座创建 LoRA 客户端，分叉和续跑加载模型及优化器。`load_checkpoint_path` 单独用于只加载权重，步数、游标和优化器从头开始。SkyRL 使用创建客户端再加载的两步协议，原生 SDK 模式使用对应的建客户端接口。已移除训练用的两个全局 monkey patch 和 `legacy_runtime.py`。
- 普通、动态与旧入口共同持有 run 目录锁，兼容旧 `.stage3.lock`。一次更新先发布策略，再发布轨迹和指标，随后依次保存完整状态、sampler，最后原子替换索引。周期保存沿用 TTL，final 永久保留。重放会裁掉最后完整状态之后的指标和轨迹，沿用该状态的真实游标与 token 计数。
- `training/logging.py` 直接保存在线 W&B 的 id、project 和 entity。分叉使用独立 ID，同目录重启沿用原 ID。重放事件以 `training/step` 表示模型步数。本地保留 `metrics.jsonl` 和 `logs.log`。累计 token 和费用随更新行记录。
- `train_dapo` 保留现有 argparse 调用形式。新参数 `--parent-step` 可指定任意有效父步，省略时兼容旧脚本的第 100 步。两个公共入口的分叉都须显式指定父步。无指定输出目录时，默认 run 名含采样模式与微秒时间戳。
- 专项测试用临时目录和 fake SDK 从第 7 步分叉，父索引另有第 11 步、重复记录及 sampler-only 记录。覆盖首次登记中断、父目录移走、锁竞争、数据重排、旧配置与错误游标。两种模式都在第 9 步注入保存失败，验证从子 run 的第 8 步恢复优化器、重放相同题目、最终每个更新仅留一条指标。W&B 使用内存假客户端。
- 独立子进程同时阻断 cookbook import 和 socket connect，覆盖训练入口、启动、日志、检查点、旧 CLI 与本地回归。保留安装依赖至第七步。全部验证只使用本地文件与模拟，未运行训练、在线 W&B、模型推理或云端操作。

## 下一次从第六步开始

1. 按总计划整合普通/动态云端驱动，将固定分叉起点和当前恢复步分别传递。用临时目录验证从 7 到 18 的评测与上传区间、空子索引和重跑补齐。
2. 整合评测脚本、stage3 兼容脚本与运维路径，模拟旧 SkyRL 基座评测和保存后关闭会话异常。继续使用本地假服务。
3. 第七步再处理安装依赖、干净环境验收和中英文展示文档。保持原分支指针，直至最终验收。

## 提交与原分支保护

- `d84c473`：完整计划和状态文档；新分支此前已 ff 合入 094bc2c、8165fec。
- `55101ee`：父归档保护及本地文件回归测试，阶段 2 完成。
- `805f753`：第三步中间态及暂停记录（WIP）。
- `f61152c`：第三步完成，130 项本地测试通过。
- 第四步完成提交：`Move ordinary and dynamic training into the local shared loop`；具体 hash 见当前分支 git log。
- 第五步完成提交：`Unify checkpoint forks, recovery and run logging`，具体 hash 见当前分支 git log。
- 本次核对原分支仍为 main=`2c53127`、codex/stage3-dapo=`094bc2c`、claude/r75-cap030=`8165fec`。

## 已确定的检查点迁移约定

- 新完整状态归属字段为 `owner="parent"` 或 `owner="child"`。第五步已为新记录写入归属。含无标记完整记录的历史索引仍保守跳过自动清理（包括无法证明独立的 run）。
- 清理先解析和验证全部路径再删除；保持原索引不变。不得凭 model ID 或 step 猜测归属。
