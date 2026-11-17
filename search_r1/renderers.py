"""Project renderers, registered before training or evaluation creates them.

Re-export the cookbook factory so callers also register the project renderer on
import. Keep the upstream name available for historical runs and checkpoints.
"""

import json

from tinker_cookbook.renderers import get_renderer as get_renderer
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5DisableThinkingRenderer

DEFAULT_RENDERER_NAME = "search_r1/qwen3_5_disable_thinking_unicode"


class Qwen3_5DisableThinkingUnicodeRenderer(Qwen3_5DisableThinkingRenderer):
    """Match the HF template's Unicode handling for structured tool arguments.

    Transformers' template ``tojson`` preserves non-ASCII characters. The pinned
    cookbook escapes them when rendering lists such as ``query_list``, which can
    break the sampled token prefix. Other normalization can still break prefixes,
    so the inherited ``has_extension_property=False`` remains appropriate.
    """

    def _format_tool_call_argument(self, value: object) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return super()._format_tool_call_argument(value)


register_renderer(DEFAULT_RENDERER_NAME, Qwen3_5DisableThinkingUnicodeRenderer)
