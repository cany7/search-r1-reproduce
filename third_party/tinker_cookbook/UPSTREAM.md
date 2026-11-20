# Cookbook source provenance — partial migration

Source repository: https://github.com/thinking-machines-lab/tinker-cookbook

Pinned source commit: `1991897cba59ea9d9b0030d183d1867b28eba9d9`.

The installed distribution used for extraction is `tinker-cookbook 0.6.2.dev1+g1991897cb`; its `direct_url.json` records that commit. `LICENSE` is the Apache-2.0 license shipped with that distribution. File headers identify derived code and local modifications. The complete upstream tree's additional attribution/NOTICE audit is still pending: stage 3 was paused at the user's request before acceptance.

| Local file under search_r1/ | Upstream file under tinker_cookbook/ | Local changes |
|---|---|---|
| env/types.py | rl/types.py | Local types/completer imports, strict zip, shortened documentation |
| env/limits.py | rl/rollout_limits.py | Local types, shortened documentation |
| env/message_env.py | rl/message_env.py | Local renderer/types/limits imports, shortened documentation |
| env/agent_tool_env.py | tool_use/agent_tool_message_env.py | Retain message environment; omit general environment factory and model preset lookup |
| env/tool_types.py | tool_use/types.py | Local message types |
| env/tool_dispatch.py | tool_use/tools.py | Local tool/message types, shortened documentation |
| env/renderers/base.py | renderers/base.py | Retain message types, parsing and generation prompt methods; omit streaming parser, supervised example builder, image processing and pickle factory |
| env/renderers/qwen3.py | renderers/qwen3.py | Retain Qwen text formatting/parsing; reject image rendering; omit streaming and OpenAI conversion methods |
| env/renderers/qwen3_5.py | renderers/qwen3_5.py | Retain Qwen3.5 text formatting/parsing; local parent classes; omit streaming/OpenAI conversion overrides |
| env/tokenizer.py | tokenizer_utils.py | Focused cached Hugging Face tokenizer path; omit other model families and registry |
| env/rollout.py | rl/rollout_runner.py | Extract unconfigured episode loop and overflow handling; no generic hooks or runner-side budgets |
| backends/completers.py | completers.py | Retain token completer and token result; omit message completer |
| backends/tinker_backend.py | checkpoint_utils.py (renderer metadata helper) | Local helper/imports/logging; other code uses explicit SDK create/load calls |

The project-specific Unicode renderer and search task code were moved from this repository. No complete cookbook package is vendored. The old training paths still depend on the installed cookbook through a temporary training-only bridge; the eventual dependency removal belongs to later stages.

This snapshot is **not a completed or fully behavior-validated migration**. See `docs/REFACTOR_STATUS.md` for the exact completed checks and remaining work.
