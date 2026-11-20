# Cookbook source provenance — stages 3–4

Source repository: https://github.com/thinking-machines-lab/tinker-cookbook

Pinned source commit: `1991897cba59ea9d9b0030d183d1867b28eba9d9`.

The installed distribution used for extraction is `tinker-cookbook 0.6.2.dev1+g1991897cb`; its `direct_url.json` records that commit. File headers identify derived code and local modifications.

On 2026-11-20 the local Git checkout was verified at the full pinned commit. All 13 source files below matched the installed distribution byte for byte. A case-insensitive audit of the complete tracked tree for NOTICE/license/copying/copyright filenames found only the root `LICENSE`; none of the extracted files carried additional copyright or attribution notices. No upstream NOTICE file exists at this commit, so none is synthesized here. The retained Apache-2.0 `LICENSE` matches both the installed distribution and the pinned root license byte for byte (SHA-256 `d13899cd96d284a6d9f130dfcf19fb3ee5f7e3be3ec627cefa5ff9d9d6023136`).

| Local file under search_r1/ | Upstream file under tinker_cookbook/ | Local changes |
|---|---|---|
| env/types.py | rl/types.py | Local types/completer imports, strict zip, shortened documentation |
| env/limits.py | rl/rollout_limits.py | Local types, shortened documentation |
| env/message_env.py | rl/message_env.py | Local renderer/types/limits imports, shortened documentation |
| env/agent_tool_env.py | tool_use/agent_tool_message_env.py | Retain message environment; omit general environment factory and model preset lookup |
| env/tool_types.py | tool_use/types.py | Local message types |
| env/tool_dispatch.py | tool_use/tools.py | Local tool/message types, shortened documentation |
| env/renderers/base.py | renderers/base.py | Retain text/thinking message types, parsing and generation prompt methods; omit image/audio types, streaming parser, supervised example builder and pickle factory |
| env/renderers/qwen3.py | renderers/qwen3.py | Retain Qwen text formatting/parsing; rename text subset of Qwen3VLRenderer to Qwen3ToolRenderer; reject image/audio input; omit streaming and OpenAI conversion methods |
| env/renderers/qwen3_5.py | renderers/qwen3_5.py | Retain Qwen3.5 text formatting/parsing; local parent classes; omit streaming/OpenAI conversion overrides |
| env/tokenizer.py | tokenizer_utils.py | Focused cached Hugging Face tokenizer path; omit other model families and registry |
| env/rollout.py | rl/rollout_runner.py | Extract unconfigured episode loop and overflow handling; no generic hooks or runner-side budgets |
| backends/completers.py | completers.py | Retain token completer and token result; omit message completer |
| backends/tinker_backend.py | checkpoint_utils.py (renderer metadata helper) | Local helper/imports/logging; project code uses explicit SDK create/load calls and owns evaluation session cleanup |

The project-specific Unicode renderer and search task code were moved from this repository. Earlier recipe adaptations remain in `env/search_env.py` (from `recipes/search_tool/search_env.py`), `env/tools.py` (from `recipes/search_tool/tools.py`) and `offline_eval.py` (from `recipes/search_tool/offline_eval.py`): local BM25 retrieval, rewards, dataset paths, evaluation records and limits are project changes. The same root license covers those recipe sources; they carry no additional notices at the pinned commit.

Stage 4 also verified the installed bytes against the same pinned checkout for `rl/train.py`, `rl/data_processing.py`, `supervised/common.py`, `utils/misc_utils.py`, `rl/rollouts.py`, `rl/metrics.py`, `rl/metric_util.py`, `rl/rollout_logging.py` and `utils/cost_estimation.py`. These sources have no additional file-level copyright or attribution notices. The root license audit above applies unchanged.

| Local file under search_r1/ | Upstream source | Local changes |
|---|---|---|
| training/config.py | rl/train.py | Retain configuration field vocabulary; validate project-supported synchronous modes; omit generic dispatch and presets |
| training/data.py | rl/data_processing.py; supervised/common.py | Retain centered advantages, prefix merge/split, targets, masks, logprobs and provenance; text-only input/target shift; local types/helpers |
| training/update.py | rl/train.py | Focused SDK gradient/Adam/snapshot API for full and streaming batches, remove mask before SDK calls; explicit pending-task cleanup; omit KL/reference/async framework |
| training/loop.py | rl/train.py; existing project train_dapo.py | One ordinary/dynamic loop with iteration-scoped data, completion-ordered minibatches and post-log checkpoint boundary; no HTML/trace framework |
| training/sampling.py | rl/rollouts.py; existing project train_dapo.py | Local fail-fast group creation/grading/cleanup and task draining; preserve project dynamic refill selection/cursor |
| training/metrics.py | rl/metric_util.py; rl/metrics.py; rl/train.py | Extract trajectory metrics, sample/train KL and usage counts; omit evaluators and reference/post-KL requests |
| training/records.py | rl/rollout_logging.py | Retain JSON-safe rollout serialization and historical schema; local types; omit store/deprecated write wrappers |
| training/usage.py | utils/cost_estimation.py | Retain token/rate value types and accounting; omit pricing network helper from local computation modules |
| training/utils.py | utils/misc_utils.py | Retain strict zip, equality, means and linspace split; omit unrelated helpers |

No complete cookbook package is vendored. Evaluation and training computation now use local implementations. A temporary startup/checkpoint/logging adapter still imports cookbook auxiliary modules; complete dependency removal is deferred. See [REFACTOR_STATUS.md](../../docs/REFACTOR_STATUS.md) for the tested scope, deliberate behavior differences and remaining work.
