"""Local Qwen text renderers and the stable renderer names saved in run configs."""

import json
from functools import partial

from .base import Message, Renderer, get_text_content  # noqa: F401
from .qwen3 import Qwen3DisableThinkingRenderer, Qwen3InstructRenderer, Qwen3Renderer
from .qwen3_5 import Qwen3_5DisableThinkingRenderer, Qwen3_5Renderer

DEFAULT_RENDERER_NAME = "search_r1/qwen3_5_disable_thinking_unicode"


class Qwen3_5DisableThinkingUnicodeRenderer(Qwen3_5DisableThinkingRenderer):
    """Preserve Unicode in structured tool arguments, matching the HF template."""

    def _format_tool_call_argument(self, value: object) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return super()._format_tool_call_argument(value)


_RENDERERS = {
    DEFAULT_RENDERER_NAME: Qwen3_5DisableThinkingUnicodeRenderer,
    "qwen3": Qwen3Renderer,
    "qwen3_disable_thinking": Qwen3DisableThinkingRenderer,
    "qwen3_instruct": Qwen3InstructRenderer,
    "qwen3_5": Qwen3_5Renderer,
    "qwen3_5_disable_thinking": Qwen3_5DisableThinkingRenderer,
    "qwen3_5_preserve_thinking": partial(Qwen3_5Renderer, strip_thinking_from_history=False),
}


def get_renderer(name, tokenizer):
    try:
        factory = _RENDERERS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported Search-R1 renderer {name!r}; choose from {list(_RENDERERS)}"
        ) from None
    return factory(tokenizer)


def get_recommended_renderer_name(model_name: str) -> str:
    name = model_name.lower()
    if "qwen3.5" in name:
        return "qwen3_5"
    if "qwen3" in name and "instruct" in name:
        return "qwen3_instruct"
    if "qwen3" in name:
        return "qwen3"
    raise ValueError(f"Specify a supported renderer explicitly for model {model_name!r}")
