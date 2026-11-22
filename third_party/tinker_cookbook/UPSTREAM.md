# Cookbook source provenance

Source repository: [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

Source commit: `1991897cba59ea9d9b0030d183d1867b28eba9d9`.

The adapted sources are licensed under Apache-2.0; see [LICENSE](LICENSE).
Source file headers identify their origins and local modifications.

## Environment and backends

| Local file (relative to `search_r1/`) | Upstream source (relative to `tinker_cookbook/`) | Project adaptations |
|---|---|---|
| `env/types.py` | `rl/types.py` | Project types, completer interfaces and strict sequence alignment |
| `env/limits.py` | `rl/rollout_limits.py` | Integration with project trajectory types |
| `env/message_env.py` | `rl/message_env.py` | Integration with project renderers, types and rollout limits |
| `env/agent_tool_env.py` | `tool_use/agent_tool_message_env.py` | Tool-based message environment for the search task |
| `env/tool_types.py` | `tool_use/types.py` | Tool types using project message types |
| `env/tool_dispatch.py` | `tool_use/tools.py` | Tool dispatch using project tool and message types |
| `env/renderers/base.py` | `renderers/base.py` | Text and thinking message types, parsing and generation prompts |
| `env/renderers/qwen3.py` | `renderers/qwen3.py` | Text-only Qwen formatting and tool-call parsing |
| `env/renderers/qwen3_5.py` | `renderers/qwen3_5.py` | Qwen3.5 formatting and parsing with project renderer classes |
| `env/tokenizer.py` | `tokenizer_utils.py` | Cached Hugging Face tokenizer loading |
| `env/rollout.py` | `rl/rollout_runner.py` | Episode execution and context overflow handling |
| `backends/completers.py` | `completers.py` | Token completion interface and result types |
| `backends/tinker_backend.py` | `checkpoint_utils.py` (renderer metadata helper) | Renderer metadata handling and SDK session lifecycle management |

## Search task

Project-specific extensions include Unicode-preserving tool argument rendering,
local BM25 retrieval, reward customization and evaluation records.

| Local file (relative to `search_r1/`) | Upstream source (relative to `tinker_cookbook/`) | Project adaptations |
|---|---|---|
| `env/search_env.py` | `recipes/search_tool/search_env.py` | Search environment configuration and dataset integration |
| `env/tools.py` | `recipes/search_tool/tools.py` | Local BM25 retrieval and reward customization |
| `offline_eval.py` | `recipes/search_tool/offline_eval.py` | Project evaluation backends, records and rollout limits |

## Training components

| Local file (relative to `search_r1/`) | Upstream source (relative to `tinker_cookbook/`) | Project adaptations |
|---|---|---|
| `training/config.py` | `rl/train.py` | Training configuration and validation of supported synchronous modes |
| `training/data.py` | `rl/data_processing.py`, `supervised/common.py` | Group-centered advantages and text trajectory processing with aligned targets, masks and logprobs |
| `training/update.py` | `rl/train.py` | SDK gradient updates, Adam steps and sampling snapshots for full and streaming batches |
| `training/loop.py` | `rl/train.py` | Shared ordinary and dynamic sampling loop with streaming minibatches and checkpoint coordination |
| `training/sampling.py` | `rl/rollouts.py` | Trajectory-group collection, grading and dynamic refill sampling |
| `training/metrics.py` | `rl/metric_util.py`, `rl/metrics.py`, `rl/train.py` | Trajectory metrics, sampling/training KL and token usage counts |
| `training/records.py` | `rl/rollout_logging.py` | JSON-safe rollout summary serialization using project trajectory types |
| `training/usage.py` | `utils/cost_estimation.py` | Token usage accounting and optional cost estimation |
| `training/utils.py` | `utils/misc_utils.py` | Sequence alignment, aggregation and batch partitioning helpers |

## Checkpointing and logging

| Local file (relative to `search_r1/`) | Upstream source (relative to `tinker_cookbook/`) | Project adaptations |
|---|---|---|
| `training/checkpoints.py`, `checkpoint_index.py` | `checkpoint_utils.py`, `cli_utils.py` | Run locking, immutable fork sources, resume cursor validation and atomic checkpoint indexing |
| `training/logging.py` | `utils/ml_log.py` | Run-scoped W&B logging, independent fork identities and local JSONL replay |
