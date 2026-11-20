# Search-R1 自包含重构方案

创建：2026-11-20。本文与 `REFACTOR_STATUS.md` 是可被后续 session 直接接续的正式记录；不依赖已删除的本地 handoff 或未跟踪的 runs/local_docs。

## 目标与最新授权

将两条实验分支的成果整合为便于展示、阅读和学习的 Search-R1 实现。迁入必要的固定版本 cookbook 代码并按本项目职责整理，最终不安装、不 import `tinker_cookbook`，继续使用 Tinker SDK。普通与动态采样共用本仓库的训练循环、更新和恢复实现。

**实验已经结束：不再运行任何训练，不做云端验证，不启动/停止云端服务，不实际上传或清理云端归档。** 本次只做本地代码、模拟测试和文档工作；将来复现遇到的问题另行处理。历史云端记录只能作为背景，不能当作迁移后实现已通过 GPU 验证的证据。用户已另行备份未跟踪资料；重构不得依赖这些文件存在。

本次授权先完成阶段 1–3，后续阶段另行继续。全程在 `codex/self-contained-refactor` 上工作，保持原 `main`、`codex/stage3-dapo`、`claude/r75-cap030` 指针不变。最终全部验收后再将 refactor 成果替换本地 main；本轮不操作 main、不推送、不删除分支/工作树、不触碰用户的 `.codex-attachments/`。

**最新调度（2026-11-20）：用户在第三步进行中要求“不要继续第三步，完成当前就暂停”。已完成阶段 1–2；第三步仅保存中间态，尚未验收，等待继续指令。具体断点见 REFACTOR_STATUS.md。**

## 起始历史与工作方式

仓库：`/Users/tim77/PycharmProjects/search-r1-reproduce`。

```text
main                         2c53127
  └─ codex/stage3-dapo        094bc2c  同步动态补采样与游标恢复
       └─ claude/r75-cap030   8165fec  训练专用封顶未答惩罚
```

在 main 基线上创建 refactor 分支，再在新分支 `git merge --ff-only claude/r75-cap030`，即可保留两项实验提交。不要在 main 上合并或重复 cherry-pick。各阶段提交独立变更，并持续更新状态文档的证据、限制和下一步。

## 目标目录与职责

```text
search_r1/
  train.py                     # 普通训练 CLI
  train_dynamic_sampling.py    # 动态 CLI，进入相同主循环
  config.py                    # 公共参数、有效配置、旧配置兼容
  offline_eval.py
  eval_config.py
  training/
    loop.py                    # 公共主循环
    sampling.py                # 普通取样、动态筛选与补采样
    data.py                    # 优势、轨迹转 token 训练数据
    update.py                  # SDK 更新与发布采样权重
    checkpoints.py             # 分叉、恢复、完整状态及游标
    logging.py                 # 指标、轨迹、W&B 身份
  env/
    types.py                   # 必要消息、工具、环境、轨迹类型
    search_env.py              # 数据读取、排序、环境创建
    search_tool_env.py         # 搜索限制、终局与训练惩罚
    tools.py                   # BM25 与基础评分
    rollout.py                 # 训练/评测共用多轮作答
    renderers.py               # Qwen 格式与解析；必要代码可拆子模块
  backends/
    tinker_backend.py          # SDK 创建/采样、SkyRL 兼容
    local_completer.py          # 本地模型评测
scripts/
  check_render.py
  make_eval_sets.py
  diagnose_run.py
  cloud/
    train_with_dev_eval.sh
    eval_checkpoint.sh
    run_train_test.sh
    stage3.sh                  # 兼容入口，转通用流程
    stage3_ops.py               # 参数化磁盘/归属/轨迹检查
    ckpt_prune.py
    upload_checkpoint.py
    ...                        # 安装、服务、monitor
tests/
docs/
  REFACTOR_PLAN.md
  REFACTOR_STATUS.md
  training.md
  migration_validation.md
third_party/tinker_cookbook/
  LICENSE
  UPSTREAM.md                  # 提交、源文件映射、修改记录
  NOTICE                       # 如有适用上游 NOTICE
retrieval/                     # 独立检索服务
experiments/                   # 历史实验档案
```

目录可按实现需要小幅调整；不复制整个 cookbook 框架，不引入新的通用插件层。第三方目录只放许可和来源记录。阶段间允许有明确记录的临时兼容入口；最终应消除 cookbook 运行时依赖及 monkey patch。

## 训练与迁移原则

普通和动态入口共用 chz `key=value` 公共配置。主流程清晰可见：初始化/恢复 → 收集题组 → 算优势/整理训练数据 → SDK forward_backward/optim_step → 发布权重 → 日志与检查点。

普通采样取固定题数；动态采样丢弃组内 reward 全同的题组，用新题补满。在一次更新前始终使用同一策略，超额轨迹不跨更新复用；达到补采样上限或数据耗尽不足整批时不更新。删除历史固定父路径、step 100、64×8、seed、学习率要求。

普通入口原有 streaming 能力需明确迁移和测试；动态入口启动时拒绝不支持的 streaming 组合。无需搬入 cookbook 所有通用执行模式。

来源固定为 cookbook `1991897cba59ea9d9b0030d183d1867b28eba9d9`。复制/改编保留适用版权、Apache-2.0 许可、上游 NOTICE（如有），在文件中标明修改与来源，在 UPSTREAM.md 建立映射。优先保持实验行为再简化结构，核对 prompt tokens、stop 条件、工具 schema、解析、长度/终止规则、reward、组内减均值优势、action token 训练信号、旧 logprob 对齐、loss/Adam 参数、更新次数和保存语义。

原 cookbook W&B monkey patch 改成本仓库日志功能；SkyRL 旧协议适配集中到 backend，尽量显式调用而非修改第三方全局类。移除 cookbook 后显式声明原先间接引入的必要依赖并更新 uv.lock。

## 四种对外实验

文档保留以下四组入口（公共示例明确 64×8；普通默认 batch_size 仍为 512）：

```bash
uv run --frozen python -m search_r1.train batch_size=64 group_size=8
uv run --frozen python -m search_r1.train batch_size=64 group_size=8 limit_search=true
uv run --frozen python -m search_r1.train batch_size=64 group_size=8 limit_search=true max_turns_unanswered_reward=-0.3
uv run --frozen python -m search_r1.train_dynamic_sampling batch_size=64 group_size=8 limit_search=true
```

这些是未来使用说明，不是本次要执行的训练命令。调用方决定总步数、保存频率和起点，无需 preset 映射。限制搜索：每轮只执行第一个 search 的前 3 条 query。封顶未答：只在训练时对满足条件的终局 reward **赋值**；第五轮有效回答沿用基础评分。评测继承搜索规则和基础评分，不使用训练专用惩罚。旧配置缺 limit_search 时按原始不限搜索规则。

移动模块更新脚本和测试，修正数据根目录并支持 SEARCH_R1_DATA_DIR；保留 renderer 名 `search_r1/qwen3_5_disable_thinking_unicode` 及训练/评测注册路径。迁移后的普通/动态/评测 CLI 均须提供帮助输出。

## 检查点、分叉与父归档

建议 Python 公共参数 parent_run/parent_step，云端 PARENT_RUN/PARENT_STEP；max_steps/TOTAL_STEPS 表示全局终点。

| 情形 | 协议 |
|---|---|
| 基座 → 普通/动态 | 独立 run、新客户端、step 与游标均从 0 开始 |
| 普通 → 普通 | 精确取父步完整记录，子索引一次性登记带父来源标记的记录，恢复模型及优化器 |
| 普通 → 动态 | 直接读指定父完整状态；子检查点索引只保存子 run 状态 |
| 已有 run 重启 | 优先最新本 run 完整状态，保留 W&B 身份，恢复游标 |
| 动态 → 普通 | 本轮不支持 |

同一步多条记录取最后一条有效完整记录，不取父最新步、不用 sampler-only。核对模型、数据、seed 和顺序；普通 → 普通 batch_size 须一致。普通父缺游标可用父步×父 batch_size 推算；动态缺游标报错。动态游标计入筛掉/超额题组。分叉 W&B 独立，续训身份稳定，重复初始化不可覆盖已有状态或换父来源。区分带优化器恢复和仅权重初始化。

保留单 run 锁；模型/优化器/游标必须匹配同次已完成更新，保持日志、轨迹、保存及完整索引提交顺序，故障从最后完整状态重放。兼容旧 config.json 和索引字段 stage3_run、parent_state_path、next_prompt_offset，不改写历史文件。

**先修复父归档保护，再启用新分叉。** 新索引标记父/子归属，按实际归档路径保护（包括重复 URI）；未知归属旧分叉索引跳过自动清理。sampler-only 不当完整状态。临时目录真实文件测试保留/删除，不操作真实 runs。新旧索引均须避免父文件被子清理误删。

## 云端脚本的本地整合范围

- train_with_dev_eval.sh 选择普通/动态、转发参数，从分叉点排评测；固定分叉起点与当前恢复步分开。7 → 18、间隔 5 的目标是 12/17/18，首次上传 (7,12]；重跑补做缺失评测/上传。
- stage3.sh/stage3_ops.py 去固定 smoke/formal 步数及父 run；保留按实际 batch/group/目标/游标计算的磁盘估算、状态归属与轨迹检查；专用删除清单退出日常入口。
- 脚本相对自身目录定位工具，覆盖 env_common、restart→tmux→start、install→patch、prune/upload/monitor；移除 STAGE3_PARENT 专用 monitor 切换。部署运行同步的项目副本，凭据和数据路径仍从 Pod 环境读取。
- eval_checkpoint.sh 以 run/step/dev或test 选完整状态和题集，默认用 run 配置，允许另一 training_config_file；配置快照覆盖对应 CLI 字段，输出按权重/题集/配置区分，检查准确题数与错误数。
- 明确基座评测模式：旧 SkyRL combined runtime 需要活跃 LoRA 客户端及心跳，再以 base_model 采样。checkpoint 评测走明确加载状态。只实现/模拟验证，不连接云端。
- run_train_test.sh 旧调用保留，支持动态保存→重启→再更新；修复末尾 echo 掩盖失败，校验目标完整状态。仅兼容已知的保存成功后关闭会话错误，不能吞其他失败。现有 stop_train 匹配也须核对。

## 分阶段交付

1. **分支合并与基线记录**：新 refactor 分支合入两个实验提交，保存本计划/状态，运行完整本地 pytest，记录版本与已有失败。
2. **父归档保护**：修复 ckpt_prune 的归属和路径保护，用临时真实文件测试旧分叉、重复 URI、sampler-only。
3. **公共组件与离线评测**：迁入必要类型、renderer、消息/搜索环境、rollout 和评测使用的 backend/辅助函数；更新调用方，让离线评测导入链先脱离 cookbook。训练暂保留旧循环并保持本地测试可用，记录过渡边界。
4. **训练实现**：迁入优势/训练数据/更新逻辑，统一普通和动态主循环，保留必要 streaming 行为。
5. **起点与恢复**：落实上表协议、W&B、旧配置兼容、锁和写盘顺序。
6. **云端驱动与评测脚本**：按上节改造，只做本地模拟测试。
7. **依赖清除与最终验收**：删除 cookbook 依赖，干净环境测试，更新 README.md/README_zh.md/scripts/README.md 与架构/验证文档，最后按用户要求让 refactor 成果替换本地 main。

## 本地验收和证据

- 合并后、迁移完成后完整 pytest，必要阶段专项测试；保留测试结果，不将模拟训练说成实际训练。
- 固定输入对照旧实现的 tokens、stop、工具解析、reward、advantage、target/logprob/训练信号；最终用固定 fixtures，测试不能依赖 cookbook。
- 四组 CLI 配置与奖励边界；非历史 batch/group/seed/起点；动态筛选、补满、耗尽、无整批不更新。
- 优化器恢复 API、父状态非最新、游标、重放、单 run 锁、W&B 身份。
- shell 模拟工作区覆盖零起点、非整倍起点、空子索引、首次上传、重复启动、补评测/上传。
- 父归档测试必须检查临时实际文件，覆盖无标记旧父、同 URI 重复及 sampler-only。
- 旧资料若仍在只读兼容核对；没有则将关键旧 schema 写为测试 fixtures，不能要求用户恢复备份才能继续。
- 干净环境不安装 cookbook，模块导入/CLI 帮助/pytest；检查 shell 语法、改动文件 ruff。
- 本轮没有训练和云端验证要求；明确注明运行时验证限制即可，不将此作为阻塞或另行索取云端操作许可。
