# Derived from tinker-cookbook/renderers/qwen3_5.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local imports and a focused text-search API.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.

from __future__ import annotations

import json
import re

from .base import (
    UNTERMINATED_TOOL_BLOCK_ERROR,
    ContentPart,
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Role,
    TextPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    has_thinking,
)
from .qwen3 import Qwen3ToolRenderer

_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>\s*$",
    re.DOTALL,
)


_PARAM_BLOCK_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def _strip_visible_content(content: list[ContentPart]) -> list[ContentPart]:
    """Trim like the HF template's ``content|trim``, skipping thinking parts."""
    parts = list(content)
    visible = [i for i, p in enumerate(parts) if p["type"] != "thinking"]
    for indices, strip in ((visible, str.lstrip), (visible[::-1], str.rstrip)):
        for i in indices:
            part = parts[i]
            if part["type"] != "text":
                break
            text = strip(part["text"])
            parts[i] = TextPart(type="text", text=text)
            if text:
                break
    return parts


class Qwen3_5Renderer(Qwen3ToolRenderer):
    """Renderer for Qwen3.5 models."""

    # The Qwen3.5-family templates wrap a run of consecutive tool responses in a
    # single <|im_start|>user block (gated on loop.previtem/nextitem).
    groups_consecutive_tool_responses = True

    @property
    def has_extension_property(self) -> bool:
        """Qwen3.5 cannot claim the extension property, even with thinking preserved."""
        return False

    def _trim_content(self, content: str | list[ContentPart]) -> str | list[ContentPart]:
        """Trim like the HF template's ``content|trim``."""
        if isinstance(content, str):
            return content.strip()
        return _strip_visible_content(content)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        message = {**message, "content": self._trim_content(message["content"])}
        return super().render_message(message, ctx)

    def _generation_suffix_str(self, role: Role, ctx: RenderContext) -> str:
        """The header the generation prompt ends with, including the `<think>` prefill."""
        maybe_newline = "\n" if ctx.idx > 0 else ""
        return f"{maybe_newline}<|im_start|>{role}\n<think>\n"

    def _generation_suffix_in_header(self, message: Message, ctx: RenderContext) -> str:
        """The produced turn is sampled after the prefill, so the prefill is observed."""
        if message["role"] != "assistant" or not ctx.is_last:
            return ""
        return self._generation_suffix_str(self._get_qwen_role_for_message(message), ctx)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """The full generation suffix: the role header plus the `<think>` prefill."""
        return self.tokenizer.encode(
            self._generation_suffix_str(role, ctx), add_special_tokens=False
        )

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Insert empty think block for assistant messages after the last user query."""
        if ctx.idx <= ctx.last_user_index:
            return ""

        content = message.get("content", "")
        has_think = has_thinking(content)

        return "" if has_think else "<think>\n\n</think>\n\n"

    def _format_thinking_text(self, thinking: str) -> str:
        """Qwen3.5 uses newline-padded think blocks."""
        return f"<think>\n{thinking}\n</think>\n\n"

    def _parse_qwen3_5_tool_call_xml(self, raw_text: str) -> ToolCall | UnparsedToolCall:
        """Parse Qwen3.5 XML-style tool calls from a raw <tool_call> block."""
        match = _FUNCTION_BLOCK_RE.match(raw_text)
        if not match:
            return UnparsedToolCall(raw_text=raw_text, error="Malformed Qwen3.5 tool call XML")

        function_name = match.group("name").strip()
        body = match.group("body")
        if not function_name:
            return UnparsedToolCall(raw_text=raw_text, error="Missing function name")

        arguments: dict[str, object] = {}
        pos = 0
        for param in _PARAM_BLOCK_RE.finditer(body):
            if body[pos : param.start()].strip():
                return UnparsedToolCall(
                    raw_text=raw_text,
                    error="Unexpected non-parameter content inside <function> block",
                )

            param_name = param.group("name").strip()
            param_value_text = param.group("value").strip("\n")
            if not param_name:
                return UnparsedToolCall(raw_text=raw_text, error="Empty parameter name")

            try:
                param_value: object = json.loads(param_value_text)
            except json.JSONDecodeError:
                param_value = param_value_text

            arguments[param_name] = param_value
            pos = param.end()

        if body[pos:].strip():
            return UnparsedToolCall(
                raw_text=raw_text,
                error="Unexpected trailing content inside <function> block",
            )

        return ToolCall(
            function=ToolCall.FunctionBody(
                name=function_name,
                arguments=json.dumps(arguments),
            )
        )

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the prefilled <think>\n before parsing sampled tokens."""
        think_prefix_tokens = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        think_suffix_token = self.tokenizer.encode("</think>", add_special_tokens=False)
        assert len(think_suffix_token) == 1

        starts_with_think = (
            len(response) >= len(think_prefix_tokens)
            and response[: len(think_prefix_tokens)] == think_prefix_tokens
        )
        if not starts_with_think and think_suffix_token[0] in response:
            return think_prefix_tokens + response
        return response

    def _postprocess_parsed_message(self, message: Message) -> None:
        """Apply Qwen3.5-specific post-processing to a parsed message in-place."""
        content = message.get("content")
        if isinstance(content, list):
            first_text_after_thinking: TextPart | None = None
            seen_thinking = False
            for p in content:
                if p["type"] == "thinking":
                    p["thinking"] = p["thinking"].strip()
                    seen_thinking = True
                elif seen_thinking and p["type"] == "text":
                    first_text_after_thinking = p
                    break

            # Template inserts exactly two separator newlines between </think> and text.
            if first_text_after_thinking is not None and first_text_after_thinking[
                "text"
            ].startswith("\n\n"):
                first_text_after_thinking["text"] = first_text_after_thinking["text"][2:]

        # Qwen3 parent parser assumes JSON inside <tool_call>; convert XML blocks here.
        converted_xml_calls: list[ToolCall] = []
        remaining_unparsed: list[UnparsedToolCall] = []
        for unparsed in message.get("unparsed_tool_calls", []):
            if (
                unparsed.error.startswith(UNTERMINATED_TOOL_BLOCK_ERROR)
                or "<function=" not in unparsed.raw_text
            ):
                # Unterminated blocks stay unparsed as-is (the XML converter
                # would replace the informative error with a generic one).
                remaining_unparsed.append(unparsed)
                continue
            parsed = self._parse_qwen3_5_tool_call_xml(unparsed.raw_text)
            if isinstance(parsed, ToolCall):
                converted_xml_calls.append(parsed)
            else:
                remaining_unparsed.append(parsed)

        if converted_xml_calls:
            message["tool_calls"] = message.get("tool_calls", []) + converted_xml_calls
        if remaining_unparsed:
            message["unparsed_tool_calls"] = remaining_unparsed
        else:
            message.pop("unparsed_tool_calls", None)

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse response with Qwen3.5-specific post-processing."""
        message, termination = super().parse_response(response)
        if not termination.is_clean:
            return message, termination

        self._postprocess_parsed_message(message)
        return message, termination

    def _format_tool_call_argument(self, value: object) -> str:
        """Serialize one tool-call argument value the way the template does."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _format_tool_call_xml(self, tool_call: ToolCall) -> str:
        """Format a single tool call in Qwen3.5's XML parameter format."""
        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        lines = [f"<tool_call>\n<function={tool_call.function.name}>"]
        for param_name, param_value in args.items():
            value_str = self._format_tool_call_argument(param_value)
            lines.append(f"<parameter={param_name}>\n{value_str}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        return "\n".join(lines)

    def _format_tool_calls_chunks(self, message: Message) -> list[TextPart]:
        """Format tool_calls using Qwen3.5's XML parameter format."""
        assert "tool_calls" in message, "tool_calls are required to format tool calls"
        content = message.get("content", "")
        if isinstance(content, str):
            has_visible_content = bool(content.strip())
        else:
            # Thinking parts are rendered separately by the template, so only text
            # parts count as content for the separator.
            has_visible_content = any(p["type"] == "text" and p["text"].strip() for p in content)
        separator = "\n\n" if has_visible_content else ""
        return [
            TextPart(
                type="text",
                text=separator
                + "\n".join(self._format_tool_call_xml(tc) for tc in message["tool_calls"]),
            )
        ]

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Qwen3.5 tool specifications."""
        tools_text = ""
        if tools:
            tool_lines = "\n".join(json.dumps(tool) for tool in tools)
            tools_text = (
                "# Tools\n\n"
                "You have access to the following functions:\n\n"
                "<tools>\n"
                f"{tool_lines}\n"
                "</tools>\n\n"
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n"
                "<function=example_function_name>\n"
                "<parameter=example_parameter_1>\n"
                "value_1\n"
                "</parameter>\n"
                "<parameter=example_parameter_2>\n"
                "This is the value for the second parameter\n"
                "that can span\n"
                "multiple lines\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n\n"
                "<IMPORTANT>\n"
                "Reminder:\n"
                "- Function calls MUST follow the specified format: "
                "an inner <function=...></function> block must be nested within "
                "<tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language "
                "BEFORE the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with "
                "your current knowledge and do not tell the user about function calls\n"
                "</IMPORTANT>"
            )

        if tools_text:
            content = tools_text + "\n\n" + system_prompt if system_prompt else tools_text
        else:
            content = system_prompt

        return [Message(role="system", content=content)]


class Qwen3_5DisableThinkingRenderer(Qwen3_5Renderer):
    """Renderer for Qwen3.5 models with thinking disabled."""

    disables_thinking = True

    def _generation_suffix_str(self, role: Role, ctx: RenderContext) -> str:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        return f"{maybe_newline}<|im_start|>{role}\n<think>\n\n</think>\n\n"
