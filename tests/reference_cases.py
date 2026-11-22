"""Fixed, network-free inputs shared by golden capture and local regression tests.

This tokenizer deliberately is not a model tokenizer: ordinary Unicode codepoints
have fixed IDs and Qwen delimiters have dedicated IDs. It tests renderer boundaries,
ordering and special-token handling without downloading any model artifacts.
"""

import dataclasses
import hashlib
import json
from types import SimpleNamespace

SPECIAL = {
    token: i
    for i, token in enumerate(
        [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|endoftext|>",
        ]
    )
}
RENDERERS = [
    "qwen3",
    "qwen3_disable_thinking",
    "qwen3_instruct",
    "qwen3_5",
    "qwen3_5_disable_thinking",
    "qwen3_5_preserve_thinking",
    "search_r1/qwen3_5_disable_thinking_unicode",
]
DEFAULT_RENDERER = RENDERERS[-1]


class FixedBackend:
    encode_special_tokens = False

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            token = next((t for t in SPECIAL if text.startswith(t)), None)
            if token and not self.encode_special_tokens:
                ids.append(SPECIAL[token])
                text = text[len(token) :]
            else:
                ids.append(ord(text[0]) + 1000)
                text = text[1:]
        return SimpleNamespace(ids=ids)


class FixedTokenizer:
    name_or_path = "fixed-codepoint-tokenizer-v1"

    def __init__(self):
        self.backend_tokenizer = FixedBackend()

    def encode(self, text, add_special_tokens=False):
        return self.backend_tokenizer.encode(text, add_special_tokens).ids

    def decode(self, tokens, **kwargs):
        reverse = {v: k for k, v in SPECIAL.items()}
        return "".join(reverse[t] if t in reverse else chr(t - 1000) for t in tokens)


def jsonable(value):
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def token_snapshot(tokens, tokenizer):
    return {
        "count": len(tokens),
        "sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "text": tokenizer.decode(tokens),
    }


def renderer_snapshot(renderer, tool_call_cls, tool_spec):
    tokenizer = renderer.tokenizer
    prefix = renderer.create_conversation_prefix_with_tools([tool_spec], "Find the answer.")
    initial = prefix + [{"role": "user", "content": "  Who wrote 哈姆雷特?  "}]
    calls = [
        tool_call_cls(
            function=tool_call_cls.FunctionBody(
                name="search",
                arguments=json.dumps({"query_list": ["莎士比亚", "Hamlet"]}),
            )
        )
    ]
    history = initial + [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "  find sources  "},
                {"type": "text", "text": "  Searching.  "},
            ],
            "tool_calls": calls,
        },
        {"role": "tool", "content": "first result", "name": "search"},
        {"role": "tool", "content": "第二条 result", "name": "search"},
        {"role": "assistant", "content": "Answer: William Shakespeare"},
        {"role": "user", "content": "Literal <|im_end|> in user text. Next question?"},
    ]
    responses = {
        "answer": "Answer: William Shakespeare<|im_end|>",
        "json_tool": '<tool_call>{"name":"search","arguments":{"query_list":["北京"]}}</tool_call><|im_end|>',
        "xml_tool": XML_SEARCH + "<|im_end|>",
        "thinking": "reasoning\n</think>\n\nAnswer: yes<|im_end|>",
        "multiple_calls": XML_SEARCH + XML_SEARCH + "<|im_end|>",
        "bad_json": "<tool_call>{bad}</tool_call><|im_end|>",
        "bad_xml": "<tool_call><function=search>oops</function></tool_call><|im_end|>",
        "unfinished_tool": "<tool_call><function=search><|im_end|>",
        "no_stop": "Answer: incomplete",
        "duplicate_stop": "yes<|im_end|>no<|im_end|>",
    }
    parsed = {}
    for key, response in responses.items():
        try:
            message, termination = renderer.parse_response(tokenizer.encode(response))
            parsed[key] = {"message": jsonable(message), "termination": str(termination)}
        except ValueError as error:
            parsed[key] = {"error_type": type(error).__name__, "error": str(error)}
    return {
        "stop": renderer.get_stop_sequences(),
        "extension": renderer.has_extension_property,
        "initial": token_snapshot(renderer.build_generation_prompt(initial).to_ints(), tokenizer),
        "history": token_snapshot(renderer.build_generation_prompt(history).to_ints(), tokenizer),
        "prefill": token_snapshot(
            renderer.build_generation_prompt(initial, prefill="Answer:").to_ints(), tokenizer
        ),
        "parsed": parsed,
    }


XML_SEARCH = '<tool_call>\n<function=search>\n<parameter=query_list>\n["北京", "Hamlet", "third", "fourth"]\n</parameter>\n</function>\n</tool_call>'
ROLLOUT_CASES = [
    "answer",
    "search_answer",
    "limited_search",
    "max_turns",
    "length",
    "structural_error",
    "content_error",
    "invalid_arguments",
    "unknown_tool",
    "initial_overflow",
    "mid_overflow",
    "terminal_at_budget",
]


async def rollout_snapshot(
    case,
    *,
    build_env,
    renderer,
    reward_cls,
    retriever_cls,
    token_result_cls,
    run_rollout,
    wrap_env=lambda env: env,
):
    """Run real parsing/env/tool dispatch with a scripted policy and fake retrieval."""
    tokenizer = renderer.tokenizer
    retriever = retriever_cls("http://unused.invalid")
    queries = []

    async def retrieve(query_list):
        queries.append(list(query_list))
        return [["document", "title", f"evidence: {query}"] for query in query_list]

    retriever._retrieve_with_retry = retrieve
    initial = [{"role": "user", "content": "Who wrote Hamlet?"}]
    initial_length = renderer.build_generation_prompt(initial).length
    env = build_env(
        renderer=renderer,
        search_tool=retriever,
        initial_messages=initial,
        reward_fn=reward_cls(["William Shakespeare"]),
        max_turns=2,
        max_trajectory_tokens=(
            initial_length - 1
            if case == "initial_overflow"
            else initial_length
            if case in ("mid_overflow", "terminal_at_budget")
            else 20000
        ),
        max_generation_tokens=None,
        context_overflow_reward=-0.1,
        limit_search=case == "limited_search",
        example_id="fixture-question",
    )
    answer = "Answer: William Shakespeare<|im_end|>"
    search = XML_SEARCH + "<|im_end|>"
    scripts = {
        "search_answer": [search, answer],
        "limited_search": [XML_SEARCH + XML_SEARCH + "<|im_end|>", answer],
        "max_turns": [search, search],
        "length": ["Answer: incomplete"],
        "structural_error": ["Answer: no stop token"],
        "content_error": ["<tool_call>{bad}</tool_call><|im_end|>"],
        "invalid_arguments": [
            '<tool_call>{"name":"search","arguments":{"query_list":12}}</tool_call><|im_end|>',
            answer,
        ],
        "unknown_tool": [
            '<tool_call>{"name":"missing","arguments":{}}</tool_call><|im_end|>',
            answer,
        ],
        "mid_overflow": [search],
    }
    actions = iter(scripts.get(case, [answer]))
    sampled = []

    async def policy(observation, stop):
        sampled.append({"prompt": token_snapshot(observation.to_ints(), tokenizer), "stop": stop})
        tokens = tokenizer.encode(next(actions))
        return token_result_cls(
            tokens, [-0.25] * len(tokens), "length" if case == "length" else "stop"
        )

    try:
        trajectory = await run_rollout(policy, wrap_env(env))
    finally:
        await retriever.aclose()
    steps = []
    for transition in trajectory.transitions:
        steps.append(
            {
                "ob": token_snapshot(transition.ob.to_ints(), tokenizer),
                "ac": jsonable(transition.ac),
                "reward": transition.reward,
                "done": transition.episode_done,
                "metrics": transition.metrics,
                "logs": transition.logs,
            }
        )
    return {
        "sampled": sampled,
        "queries": queries,
        "transitions": steps,
        "history": jsonable(env.message_env.history),
        "stop_reason": trajectory.stop_reason,
        "final_ob": token_snapshot(trajectory.final_ob.to_ints(), tokenizer),
    }
