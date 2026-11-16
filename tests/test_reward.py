"""F1+ reward and the recipe's original reward on hand-built histories."""

import asyncio

import pytest
from tinker_cookbook.renderers.base import Message, ToolCall

from search_r1.tools import TextAnswerReward, compute_f1, extract_search_queries

GOLD = ["Wilhelm Conrad Röntgen", "W. C. Röntgen"]


def history(final: str, queries: list[list[str]] | None = None) -> list[Message]:
    """A conversation with one search call per entry of ``queries``, then ``final``."""
    messages: list[Message] = [{"role": "user", "content": "question"}]
    for query_list in queries if queries is not None else [["first nobel prize physics"]]:
        arguments = '{"query_list": [' + ", ".join(f'"{q}"' for q in query_list) + "]}"
        call = ToolCall(function=ToolCall.FunctionBody(name="search", arguments=arguments))
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "content": "Query: ...\nDocument 1:\n..."})
    messages.append({"role": "assistant", "content": final})
    return messages


def grade(messages: list[Message], **kwargs) -> tuple[float, dict[str, float]]:
    return asyncio.run(TextAnswerReward(gold_answers=GOLD, **kwargs)(messages))


@pytest.mark.parametrize(
    ("final", "searched", "f1_plus", "original"),
    [
        ("Answer: Wilhelm Conrad Röntgen", True, 1.0, 1.0),
        ("Answer: the wilhelm conrad röntgen.", True, 1.0, 1.0),  # normalization
        ("Answer: Wilhelm Conrad Röntgen", False, 0.9, 1.0),  # no search: -0.1
        ("Answer: Röntgen", True, 0.5, 0.0),  # partial credit only under F1
        ("Answer: Einstein", True, 0.0, 0.0),
        ("It was Röntgen.", True, -0.1, -0.1),  # no "Answer:"
        ("It was Röntgen.", False, -0.2, -0.1),  # both penalties
        ("Answer: a Answer: Wilhelm Conrad Röntgen", True, -0.1, -0.1),  # two "Answer:"
    ],
)
def test_reward(final: str, searched: bool, f1_plus: float, original: float) -> None:
    messages = history(final, None if searched else [])
    assert grade(messages)[0] == pytest.approx(f1_plus)
    assert grade(messages, answer_score="em", no_search_coef=0.0)[0] == pytest.approx(original)


def test_metrics_report_em_and_f1_under_either_reward() -> None:
    messages = history("Answer: Röntgen")
    for kwargs in ({}, {"answer_score": "em", "no_search_coef": 0.0}):
        _, metrics = grade(messages, **kwargs)
        assert metrics == {"format": 1.0, "correct": 0.0, "f1": 0.5, "searched": 1.0}


def test_f1_takes_best_gold_answer() -> None:
    # "W. C. Röntgen" normalizes to three tokens; "W C Röntgen" matches it exactly.
    assert grade(history("Answer: W C Röntgen"))[1]["f1"] == pytest.approx(1.0)


def test_compute_f1() -> None:
    assert compute_f1("new york city", "New York") == pytest.approx(0.8)
    assert compute_f1("", "New York") == 0.0
    assert compute_f1("paris", "london") == 0.0


def test_extract_search_queries() -> None:
    messages = history("Answer: x", [["a", "b"], ["a"]])
    assert extract_search_queries(messages) == (2, ["a", "b", "a"])
