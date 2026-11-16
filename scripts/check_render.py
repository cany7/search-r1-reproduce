"""环境自检：加载两个模型的 tokenizer 和 renderer，渲染一条带工具的初始 prompt。

不需要推理引擎和检索服务。用法：uv run python scripts/check_render.py
"""

from typing import Annotated

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

CONFIGS = [
    ("Qwen/Qwen3.5-4B", "qwen3_5_disable_thinking"),
    ("Qwen/Qwen3.5-4B", "qwen3_5"),
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct"),
]


class DummyTool:
    @tool
    async def search(
        self,
        query_list: Annotated[list[str], "A list of fully-formed semantic queries."],
    ) -> ToolResult:
        """Search Wikipedia for relevant information based on the given query."""
        return simple_tool_result("")


def main() -> None:
    spec = DummyTool().search.to_spec()
    for model_name, renderer_name in CONFIGS:
        tokenizer = tokenizer_utils.get_tokenizer(model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        messages = renderer.create_conversation_prefix_with_tools(
            tools=[spec], system_prompt="You are a helpful assistant."
        ) + [{"role": "user", "content": "Who wrote Hamlet?"}]
        model_input = renderer.build_generation_prompt(messages)
        tokens = model_input.to_ints()
        print("=" * 80)
        print(f"{model_name} | renderer={renderer_name} | recommended="
              f"{model_info.get_recommended_renderer_name(model_name)}")
        print(f"prompt tokens: {len(tokens)} | stop: {renderer.get_stop_sequences()}")
        print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
