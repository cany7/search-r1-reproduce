# Derived from tinker-cookbook/renderers/base.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local imports and a focused text-search API.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

import pydantic
import tinker
from PIL import Image

from search_r1.env.tokenizer import Tokenizer, encode_without_special_tokens


class RendererError(ValueError):
    """The response or message cannot be rendered consistently."""

logger = logging.getLogger(__name__)


class ParseTermination(StrEnum):
    """How a sampled response ended, as observed by ``Renderer.parse_response``."""

    STOP_SEQUENCE = "stop_sequence"
    """The renderer's expected stop signal (``\\n\\nUser:``, ``<|im_end|>``, ...) fired."""

    EOS = "eos"
    """Termination via the model's EOS token, accepted as a fallback by some renderers."""

    MALFORMED = "malformed"
    """Response did not terminate cleanly (truncated, multiple stop signals, ...)."""

    @property
    def is_clean(self) -> bool:
        """True for any clean termination — what eval grading should gate on."""
        return self != ParseTermination.MALFORMED

    @property
    def is_stop_sequence(self) -> bool:
        """True only for the renderer's expected stop signal — strict format-reward gate."""
        return self == ParseTermination.STOP_SEQUENCE


class StrictBase(pydantic.BaseModel):
    """Pydantic base class that's immutable and doesn't silently ignore extra fields."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)


class ToolCall(StrictBase):
    """Structured tool invocation following OpenAI/kosong format."""

    class FunctionBody(pydantic.BaseModel):
        """Tool call function body containing the tool name and arguments."""

        name: str
        """The name of the tool to be called."""
        arguments: str
        """Arguments of the tool call in JSON string format."""

    type: Literal["function"] = "function"
    """Tool call type, must be 'function' for compatibility."""

    id: str | None = None
    """Optional unique identifier for tracking this specific tool call."""

    function: FunctionBody
    """The function body containing tool name and arguments."""


class UnparsedToolCall(StrictBase):
    """Represents a tool call that failed to parse from model output."""

    raw_text: str
    """The original text from the model that failed to parse."""

    error: str
    """Description of what went wrong during parsing."""


class TextPart(TypedDict):
    """A chunk of text content in a message, usually meant to be visible to the user"""

    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    """A chunk of image content in a message."""

    type: Literal["image"]
    image: str | Image.Image


class AudioPart(TypedDict):
    """A chunk of input audio in a message."""

    type: Literal["audio"]
    audio: str | bytes
    format: NotRequired[Literal["wav", "mp3", "flac"]]
    num_frames: NotRequired[int]
    sample_rate: NotRequired[int]


class ThinkingPart(TypedDict):
    """Model's internal reasoning (chain-of-thought) as a content part."""

    type: Literal["thinking"]
    thinking: str  # The thinking/reasoning content


ContentPart = TextPart | ImagePart | AudioPart | ThinkingPart




_REPLACEMENT_CHAR = "\ufffd"


_THINK_OPEN_TAG = "<think>"


_THINK_CLOSE_TAG = "</think>"


Role = str


Content = str | list[ContentPart]


class Message(TypedDict):
    """Container for a single turn in a multi-turn conversation."""

    role: Role
    content: Content

    tool_calls: NotRequired[list[ToolCall]]
    unparsed_tool_calls: NotRequired[list[UnparsedToolCall]]
    trainable: NotRequired[bool]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


@dataclass
class RenderContext:
    """Context passed to render_message for rendering a single message."""

    idx: int
    """Index of the message in the conversation (0-based)."""

    is_last: bool
    """Whether this is the last message in the conversation."""

    prev_message: Message | None = None
    """The previous message in the conversation, if any."""

    next_message: Message | None = None
    """The next message in the conversation, if any."""

    last_user_index: int = -1
    """Index of the last user message in the conversation. -1 if no user messages."""

    in_last_assistant_turn: bool = False
    """Whether this message belongs to the turn the model is being asked to produce."""


class ToolSpec(TypedDict):
    """Tool specification following the OpenAI function calling format."""

    name: str
    """The name of the tool."""
    description: str
    """A description of what the tool does."""
    parameters: dict
    """JSON Schema object describing the tool's parameters."""


def ensure_text(content: Content) -> str:
    """Assert that content is text-only and return it as a string."""
    if isinstance(content, str):
        return content
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    raise RendererError(f"Expected text content, got multimodal content with {len(content)} parts")


def ensure_list(content: Content) -> list[ContentPart]:
    """Normalize content to list form. Wraps string content in a TextPart."""
    if isinstance(content, str):
        return [TextPart(type="text", text=content)]
    return content


def has_thinking(content: Content) -> bool:
    """Whether content carries reasoning, in either shape it can arrive in."""
    if isinstance(content, list):
        return any(p["type"] == "thinking" for p in content)
    return "<think>" in content


def remove_thinking(parts: list[ContentPart]) -> list[ContentPart]:
    """Filter out ThinkingPart elements from a content part list."""
    return [p for p in parts if p["type"] != "thinking"]


def get_text_content(message: Message) -> str:
    """Extract text content from message, stripping thinking parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def format_content_as_string(content: Content, separator: str = "\n") -> str:
    """Format message content as a string, handling all content part types."""
    if isinstance(content, str):
        return content

    parts = []
    for p in content:
        if p["type"] == "thinking":
            parts.append(f"<think>{p['thinking']}</think>")
        elif p["type"] == "text":
            parts.append(p["text"])
        elif p["type"] == "image":
            img = p["image"]
            if isinstance(img, str):
                parts.append(f"<image>{img}</image>")
            else:
                parts.append(f"<image>Image({img.size[0]}x{img.size[1]}, {img.mode})</image>")
        else:
            raise RendererError(f"Unknown content part type: {p['type']}")
    return separator.join(parts)


def _parse_tool_call_json(tool_call_str: str, raw_text: str) -> ToolCall | UnparsedToolCall:
    """Parse tool call JSON. Returns UnparsedToolCall on failure."""
    try:
        tool_call = json.loads(tool_call_str.strip())
    except json.JSONDecodeError as e:
        return UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")

    if not isinstance(tool_call, dict):
        return UnparsedToolCall(raw_text=raw_text, error="Tool call is not a JSON object")

    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    tool_id = tool_call.get("id")

    if not isinstance(name, str):
        return UnparsedToolCall(raw_text=raw_text, error="Missing or invalid 'name' field")
    if not isinstance(arguments, dict):
        return UnparsedToolCall(raw_text=raw_text, error="Missing or invalid 'arguments' field")

    if tool_id is not None and not isinstance(tool_id, str):
        tool_id = None

    # TODO: arguments is already a dict from json.loads above, but ToolCall.FunctionBody.arguments
    # expects a JSON string. This round-trip (loads then dumps) is wasteful. Consider changing
    # FunctionBody.arguments to accept dict directly, or parse tool calls more lazily.
    # We may want to revisit the decision to store arguments as unparsed JSON string.
    return ToolCall(
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
        id=tool_id,
    )


def parse_content_blocks(
    content: str,
) -> tuple[list[ContentPart], list[ToolCall | UnparsedToolCall]] | None:
    """Parse a string with <think>...</think> and <tool_call>...</tool_call> tags."""
    if "<think>" not in content and "<tool_call>" not in content:
        return None  # No special blocks, caller should use original string

    parts: list[ContentPart] = []
    tool_calls: list[ToolCall | UnparsedToolCall] = []
    pos = 0

    # Pattern to find both <think>...</think> and <tool_call>...</tool_call> blocks
    pattern = re.compile(r"<think>(.*?)</think>|<tool_call>(.*?)</tool_call>", re.DOTALL)

    for match in pattern.finditer(content):
        # Add any text before this block (preserve whitespace for identity roundtrip)
        text_before = content[pos : match.start()]
        if text_before:  # Skip only truly empty strings
            parts.append(TextPart(type="text", text=text_before))

        if match.group(1) is not None:
            # This is a <think> block
            thinking = match.group(1)
            if thinking:  # Skip empty thinking blocks
                parts.append(ThinkingPart(type="thinking", thinking=thinking))
        else:
            # This is a <tool_call> block — goes into separate tool_calls list
            tool_call_json = match.group(2)
            raw_text = match.group(0)  # Full match including tags
            tool_calls.append(_parse_tool_call_json(tool_call_json, raw_text))

        pos = match.end()

    # Add any remaining text after the last block
    remaining = content[pos:]
    if remaining:  # Skip only truly empty strings
        parts.append(TextPart(type="text", text=remaining))

    return parts, tool_calls


def parse_think_blocks(content: str) -> list[ContentPart] | None:
    """Parse a string with only <think>...</think> tags into ThinkingPart/TextPart list."""
    if "<think>" not in content:
        return None

    parts: list[ContentPart] = []
    pos = 0
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    for match in pattern.finditer(content):
        text_before = content[pos : match.start()]
        if text_before:  # Skip only truly empty strings
            parts.append(TextPart(type="text", text=text_before))

        thinking = match.group(1)
        if thinking:  # Skip empty thinking blocks
            parts.append(ThinkingPart(type="thinking", thinking=thinking))

        pos = match.end()

    remaining = content[pos:]
    if remaining:  # Skip only truly empty strings
        parts.append(TextPart(type="text", text=remaining))

    return parts


class ParseFailureKind(StrEnum):
    """What kind of parse failure a sampled response exhibits."""

    STRUCTURAL = "structural"
    """Broken framing: the response did not terminate cleanly.  Not"""

    CONTENT = "content"
    """Clean framing with unparsable content (tool-call JSON errors,"""


PARSE_FAILURE_DETAIL_MAX_CHARS = 16384


UNTERMINATED_TOOL_BLOCK_ERROR = "unterminated tool block"


def classify_parse_failure(
    message: Message, termination: ParseTermination
) -> tuple[ParseFailureKind, str] | None:
    """Classify the parse failure (if any) in a ``parse_response`` result."""
    if not termination.is_clean:
        return (
            ParseFailureKind.STRUCTURAL,
            "response did not terminate cleanly "
            f"(termination={termination.value}): expected the renderer's stop signal "
            "or EOS, so the message framing is broken",
        )
    unparsed = message.get("unparsed_tool_calls") or []
    if unparsed:
        detail = "\n".join(tc.error for tc in unparsed)
        return (ParseFailureKind.CONTENT, detail[:PARSE_FAILURE_DETAIL_MAX_CHARS])
    return None


def detect_unterminated_tool_block(
    content: str, open_marker: str, close_marker: str
) -> UnparsedToolCall | None:
    """Detect a tool block that was opened but never closed."""
    opens = content.count(open_marker)
    if opens == 0 or opens <= content.count(close_marker):
        return None
    return UnparsedToolCall(
        raw_text=content[content.rfind(open_marker) :][:PARSE_FAILURE_DETAIL_MAX_CHARS],
        error=(
            f"{UNTERMINATED_TOOL_BLOCK_ERROR}: '{open_marker}' opened without a "
            f"matching '{close_marker}'"
        ),
    )


def _tool_call_payload(tool_call: ToolCall) -> dict[str, object]:
    """Minimal JSON payload for embedding in <tool_call> blocks."""
    # Convert from nested structure to flat format for compatibility
    return {
        "name": tool_call.function.name,
        "arguments": json.loads(tool_call.function.arguments),
    }


@dataclass(frozen=True)
class RenderedMessage:
    """Container for parts of a rendered message, structured for loss masking."""

    output: list[tinker.ModelInputChunk]
    """What the model generates for this turn."""

    header: tinker.EncodedTextChunk | None = None
    """Role identifier and delimiters that introduce the turn."""

    stop_overlap: tinker.EncodedTextChunk | None = None
    """Tokens that overlap between stop sequence and next message's header."""


class Renderer(ABC):
    """Abstract base class for rendering message lists into training and sampling prompts."""

    tokenizer: Tokenizer

    disables_thinking: bool = False
    """Whether this renderer's generation prompt closes the think block."""

    # Pickle metadata — set by get_renderer() via _stamp_pickle_metadata().
    # Class-level defaults ensure these exist even when subclasses bypass super().__init__().
    _renderer_name: str | None = None
    _model_name: str | None = None
    _has_image_processor: bool = False

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer


    @property
    def has_extension_property(self) -> bool:
        """Whether this renderer satisfies the sequence extension property."""
        return False

    @property
    def _bos_tokens(self) -> list[int]:
        return []

    @abstractmethod
    def get_stop_sequences(self) -> list[str] | list[int]:
        """Return stop token IDs or strings that signal end-of-generation for the model."""
        ...

    @abstractmethod
    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a single message into its header/output/stop_overlap components."""
        ...

    @abstractmethod
    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled tokens back into a Message."""
        ...

    supports_streaming: bool = False
    """Whether this renderer supports streaming response parsing."""

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Normalize sampled response tokens before parsing."""
        return response

    @property
    def _end_message_token(self) -> int:
        """The token ID that marks the end of a message."""
        raise NotImplementedError(
            f"{type(self).__name__} must define _end_message_token to support streaming"
        )




    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create message(s) with tool specifications to prepend to conversations."""
        raise NotImplementedError

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Return tokens to append to the prompt for generation."""
        # Default: render an empty message and use its header tokens
        rendered = self.render_message(Message(role=role, content=""), ctx)
        if rendered.header:
            return list(rendered.header.tokens)
        return []

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        """Convert a message list to a token prompt for sampling."""

        chunks: list[tinker.types.ModelInputChunk] = []
        if self._bos_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=self._bos_tokens))

        last_user_idx = max(
            (idx for idx, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )

        turn_start = self._last_assistant_turn_start_index(messages)

        for idx, message in enumerate(messages):
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(messages) - 1),
                prev_message=messages[idx - 1] if idx > 0 else None,
                next_message=messages[idx + 1] if idx + 1 < len(messages) else None,
                last_user_index=last_user_idx,
                in_last_assistant_turn=idx >= turn_start,
            )
            rendered_message = self.render_message(message, ctx)
            header_chunk = rendered_message.header
            output_chunks = rendered_message.output
            # Skip empty EncodedTextChunks (header or output): the sampler rejects them with
            # a 400 "Chunk N has empty tokens list". An empty header happens e.g. for the
            # second of several consecutive tool messages, which share one turn header.
            if header_chunk and (
                not isinstance(header_chunk, tinker.EncodedTextChunk) or header_chunk.tokens
            ):
                chunks.append(header_chunk)
            chunks.extend(
                [x for x in output_chunks if not isinstance(x, tinker.EncodedTextChunk) or x.tokens]
            )

        suffix_ctx = RenderContext(
            idx=len(messages),
            is_last=True,
            prev_message=messages[-1] if messages else None,
            last_user_index=last_user_idx,
            in_last_assistant_turn=True,
        )
        suffix_tokens = self._get_generation_suffix(role, suffix_ctx)
        if suffix_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=suffix_tokens))

        if prefill:
            chunks.append(
                tinker.types.EncodedTextChunk(
                    tokens=encode_without_special_tokens(self.tokenizer, prefill)
                )
            )
        return tinker.ModelInput(chunks=chunks)





    def _last_assistant_turn_start_index(self, messages: list[Message]) -> int:
        """Index of the first message in the turn the model is being asked to produce."""
        return max((idx for idx, m in enumerate(messages) if m["role"] == "user"), default=-1) + 1



def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, ParseTermination]:
    """Parse a sampled token sequence by splitting on a stop token."""
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = str(tokenizer.decode(response))
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return Message(role="assistant", content=str_response), ParseTermination.MALFORMED
    elif emt_count == 1:
        str_response = str(tokenizer.decode(response[: response.index(stop_token)]))
        return Message(role="assistant", content=str_response), ParseTermination.STOP_SEQUENCE
    else:
        raise RendererError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )
