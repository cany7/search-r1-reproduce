"""Training-only terminal reward for episodes that hit max_turns without a valid answer."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import chz
import pytest
import tinker

from search_r1 import offline_eval, train
from search_r1.backends.completers import TokensWithLogprobs
from search_r1.env import search_env
from search_r1.env.renderers.base import Message, ParseTermination, ToolCall, UnparsedToolCall
from search_r1.env.search_tool_env import MAX_TURNS_UNANSWERED_LOG, build_search_env
from search_r1.env.tools import BM25Tool, TextAnswerReward
from search_r1.env.types import Trajectory, Transition
from search_r1.eval_config import eval_settings_from_training_config

CAP = -0.3


def search(queries: list[str], content: str = "") -> Message:
    call = ToolCall(
        function=ToolCall.FunctionBody(name="search", arguments=json.dumps({"query_list": queries}))
    )
    return {"role": "assistant", "content": content, "tool_calls": [call]}


@pytest.fixture
def retriever(monkeypatch):
    tool = BM25Tool("http://unused")
    monkeypatch.setattr(
        tool, "_retrieve_with_retry", AsyncMock(side_effect=lambda qs: [["p", "p", q] for q in qs])
    )
    return tool


def env_for(retriever, reward=CAP, *, renderer=None, max_trajectory_tokens=32768):
    if renderer is None:
        renderer = Mock()
        renderer.get_stop_sequences.return_value = [2]
    return build_search_env(
        renderer=renderer,
        search_tool=retriever,
        initial_messages=[{"role": "user", "content": "question"}],
        reward_fn=TextAnswerReward(["yes"]),
        max_turns=5,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=None,
        context_overflow_reward=-0.1,
        limit_search=True,
        example_id="q",
        max_turns_unanswered_reward=reward,
    )


async def play(env, final: Message, turns: int = 5):
    """Four searches, then ``final`` as the fifth (or ``turns``-th) message."""
    message_env = env.message_env
    await message_env.initial_observation()
    for i in range(turns - 1):
        step = await message_env.step(search([f"q{i}"]))
        assert not step.episode_done and step.reward == 0.0
    return await message_env.step(final)


def test_fifth_search_without_answer_gets_the_absolute_reward_once(retriever):
    last = asyncio.run(play(env_for(retriever), search(["q4"])))
    assert last.episode_done and last.metrics["stop/max_turns"] == 1.0
    assert last.metrics["format"] == 0.0
    assert last.reward == CAP
    assert last.logs[MAX_TURNS_UNANSWERED_LOG] == pytest.approx(-0.1)


def test_default_and_evaluation_keep_the_shared_score(retriever):
    last = asyncio.run(play(env_for(retriever, reward=None), search(["q4"])))
    assert last.metrics["stop/max_turns"] == 1.0
    assert last.reward == pytest.approx(-0.1)
    assert MAX_TURNS_UNANSWERED_LOG not in last.logs


@pytest.mark.parametrize(
    "final, stop, reward",
    [
        # A fifth turn that answers stops as completed and keeps F1 and format terms.
        ({"role": "assistant", "content": "Answer: yes"}, "stop/completed", 1.0),
        ({"role": "assistant", "content": "Answer: no"}, "stop/completed", 0.0),
        # max_turns with a tool call next to a valid answer is graded normally.
        (search(["q4"], "Answer: yes"), "stop/max_turns", 1.0),
        # Ordinary end without an answer keeps the format penalty.
        ({"role": "assistant", "content": "I do not know."}, "stop/completed", -0.1),
    ],
)
def test_answers_and_ordinary_endings_are_unchanged(retriever, final, stop, reward):
    last = asyncio.run(play(env_for(retriever), final))
    assert last.episode_done and last.metrics[stop] == 1.0
    assert last.reward == pytest.approx(reward)
    assert MAX_TURNS_UNANSWERED_LOG not in last.logs


def test_early_ending_without_answer_is_unchanged(retriever):
    last = asyncio.run(play(env_for(retriever), {"role": "assistant", "content": "?"}, turns=2))
    assert last.metrics["stop/completed"] == 1.0 and last.reward == pytest.approx(-0.1)


def test_fifth_turn_content_parse_failure_is_unchanged(retriever):
    bad = {
        "role": "assistant",
        "content": "",
        "unparsed_tool_calls": [UnparsedToolCall(raw_text="<tool_call>{", error="bad json")],
    }
    env = env_for(retriever)
    last = asyncio.run(play(env, bad))
    assert last.episode_done and last.metrics["max_turns"] == 1.0
    assert last.metrics["stop/parse_error"] == 1.0 and "stop/max_turns" not in last.metrics
    assert last.reward == env.message_env.failed_parse_reward
    assert MAX_TURNS_UNANSWERED_LOG not in last.logs


def token_env(retriever, parsed, *, max_trajectory_tokens=32768, prompt_len=1):
    renderer = Mock()
    renderer.get_stop_sequences.return_value = [2]
    renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1] * prompt_len)
    renderer.parse_response.side_effect = parsed
    return env_for(retriever, renderer=renderer, max_trajectory_tokens=max_trajectory_tokens)


def test_truncation_structural_failure_and_overflow_are_unchanged(retriever):
    async def run(env, extra=None, turns=5):
        await env.initial_observation()
        for _ in range(turns - 1):
            assert not (await env.step([3])).episode_done
        return await env.step([3], extra=extra)

    four = [(search([f"q{i}"]), ParseTermination.STOP_SEQUENCE) for i in range(4)]
    truncated = asyncio.run(run(token_env(retriever, four), extra={"stop_reason": "length"}))
    assert truncated.metrics["stop/max_tokens"] == 1.0 and truncated.reward == -0.1

    broken = four + [({"role": "assistant", "content": ""}, ParseTermination.MALFORMED)]
    env = token_env(retriever, broken)
    malformed = asyncio.run(run(env))
    assert malformed.metrics["stop/parse_error"] == 1.0
    assert malformed.reward == env.failed_parse_reward == pytest.approx(-0.1)

    env = token_env(retriever, four, max_trajectory_tokens=100, prompt_len=50)
    env.renderer.build_generation_prompt.side_effect = [
        tinker.ModelInput.from_ints([1] * 50),
        tinker.ModelInput.from_ints([1] * 200),
    ]
    overflow = asyncio.run(run(env, turns=1))
    assert overflow.metrics["stop/context_overflow"] == 1.0 and overflow.reward == -0.1


def trajectory(last_logs):
    ac = TokensWithLogprobs([3], [0.0])
    ob = tinker.ModelInput.from_ints([1])
    steps = [Transition(ob=ob, ac=ac, reward=0.0, episode_done=False)] * 2
    steps.append(Transition(ob=ob, ac=ac, reward=CAP, episode_done=True, logs=last_logs))
    return Trajectory(transitions=steps, final_ob=ob)


def test_group_metric_uses_every_trajectory_as_denominator():
    datum = {"id": "q", "question": "Who?", "answer": ["yes"], "data_source": "nq"}
    builder = search_env.SearchEnvGroupBuilder(
        datum, "m", None, 5, 4, Mock(), limit_search=True, max_turns_unanswered_reward=CAP
    )
    group = [trajectory({MAX_TURNS_UNANSWERED_LOG: -0.1})] + [trajectory({})] * 3
    rewards = asyncio.run(builder.compute_group_rewards(group, []))
    assert [r for r, _ in rewards] == [0.0] * 4
    assert [m["max_turns_unanswered"] for _, m in rewards] == [1.0, 0.0, 0.0, 0.0]
    default = search_env.SearchEnvGroupBuilder(datum, "m", None, 5, 4, Mock(), limit_search=True)
    assert asyncio.run(default.compute_group_rewards(group, [])) == [(0.0, {})] * 4


def test_unlimited_search_rejects_the_option(retriever):
    with pytest.raises(ValueError, match="limit_search"):
        build_search_env(
            renderer=Mock(),
            search_tool=retriever,
            initial_messages=[],
            reward_fn=TextAnswerReward(["yes"]),
            max_turns=5,
            max_trajectory_tokens=32768,
            max_generation_tokens=None,
            context_overflow_reward=-0.1,
            limit_search=False,
            example_id="q",
            max_turns_unanswered_reward=CAP,
        )


def test_config_records_the_value_and_eval_keeps_the_original_score(tmp_path, monkeypatch, retriever):
    run_training = AsyncMock()
    monkeypatch.setattr("search_r1.training.legacy_runtime.main", run_training)
    monkeypatch.setattr(
        search_env,
        "download_search_r1_dataset",
        lambda split: [{"id": "q", "question": "Who?", "answer": ["yes"], "data_source": "nq"}],
    )
    monkeypatch.setattr(search_env.tokenizer_utils, "get_tokenizer", Mock())
    renderer = Mock()
    renderer.create_conversation_prefix_with_tools.return_value = []
    monkeypatch.setattr(search_env, "get_renderer", lambda *args: renderer)
    asyncio.run(
        train.cli_main(
            train.CLIConfig(
                log_path=str(tmp_path / "run"),
                limit_search=True,
                max_turns_unanswered_reward=CAP,
                batch_size=1,
                group_size=2,
                skyrl_compat=False,
                wandb_project=None,
            )
        )
    )
    config = run_training.await_args.args[0]
    snapshot = chz.asdict(config)
    assert snapshot["dataset_builder"]["max_turns_unanswered_reward"] == CAP
    dataset, _ = asyncio.run(config.dataset_builder())
    envs = asyncio.run(dataset.get_batch(0)[0].make_envs())
    assert {e.message_env.max_turns_unanswered_reward for e in envs} == {CAP}
    settings = eval_settings_from_training_config(snapshot)
    assert not any("unanswered" in key for key in settings)
    assert settings["limit_search"] is True
    assert "max_turns_unanswered_reward" not in chz.asdict(offline_eval.CLIConfig())

    # Offline evaluation of a five-search trajectory keeps the shared -0.1 score.
    eval_renderer = Mock()
    eval_renderer.get_stop_sequences.return_value = [2]
    eval_renderer.create_conversation_prefix_with_tools.return_value = []
    eval_renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1])
    eval_renderer.parse_response.side_effect = [
        (search([f"q{i}"]), ParseTermination.STOP_SEQUENCE) for i in range(5)
    ]
    record = asyncio.run(
        offline_eval.evaluate_single_item(
            {"id": "q", "question": "Who?", "answer": ["yes"], "data_source": "nq"},
            retriever,
            AsyncMock(return_value=TokensWithLogprobs([2], [0.0])),
            eval_renderer,
            Mock(),
            offline_eval.CLIConfig(limit_search=True),
            asyncio.Semaphore(1),
        )
    )
    assert record["termination"] == "max_turns"
    assert record["reward"] == pytest.approx(-0.1)
