import asyncio
import json
from unittest.mock import AsyncMock, Mock, call

import chz
import pytest
import tinker

from search_r1 import offline_eval, train
from search_r1.backends.completers import TokensWithLogprobs
from search_r1.env import search_env
from search_r1.env.renderers.base import Message, ParseTermination, ToolCall
from search_r1.env.rollout import do_single_rollout
from search_r1.env.search_tool_env import build_search_env, search_tool_spec
from search_r1.env.tools import (
    BM25Tool,
    RetrievalUnavailable,
    TextAnswerReward,
    count_tool_errors,
)
from search_r1.eval_config import eval_settings_from_training_config


def search(*lists: object) -> Message:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="search", arguments=json.dumps({"query_list": q})
                )
            )
            for q in lists
        ],
    }


@pytest.fixture
def backend(monkeypatch):
    retriever = BM25Tool("http://unused")

    async def passages(queries):
        await asyncio.sleep(0)  # Let other trajectories start their tool calls.
        return [["shared passage", "shared passage", q] for q in queries]

    fetch = AsyncMock(side_effect=passages)
    monkeypatch.setattr(retriever, "_retrieve_with_retry", fetch)
    return retriever, fetch


def make_env(retriever, limit_search=True, example_id="q"):
    return build_search_env(
        renderer=Mock(),
        search_tool=retriever,
        initial_messages=[{"role": "user", "content": "question"}],
        reward_fn=TextAnswerReward(["yes"]),
        max_turns=5,
        max_trajectory_tokens=32768,
        max_generation_tokens=None,
        context_overflow_reward=-0.1,
        limit_search=limit_search,
        example_id=example_id,
    ).message_env


def test_concurrent_trajectories_keep_requests_and_execute_prefixes(backend):
    retriever, fetch = backend

    async def episode(label):
        env = make_env(retriever, example_id=label)
        await env.initial_observation()
        requested = [label, "重复", "重复", "unused"]
        message = search(requested, ["later"])
        step = await env.step(message)
        assert env.history[1] == message
        assert len(env.history) == 3
        expected = "".join(
            f"Query: {q}\nDocument 1:\nshared passage\nDocument 2:\nshared passage\nDocument 3:\n{q}\n"
            for q in requested[:3]
        )
        assert env.history[-1]["content"] == expected
        assert not step.episode_done and step.reward == 0
        assert step.logs["example_id"] == label
        assert step.metrics["first_turn_queries_requested"] == 5
        assert json.loads(step.logs["search_calls"]) == [
            {
                "requested_queries": requested,
                "executed_queries": requested[:3],
                "status": "dispatched",
            },
            {"requested_queries": ["later"], "executed_queries": [], "status": "skipped"},
        ]
        second = await env.step(search([label], ["still unused"]))
        assert second.metrics["search_calls_executed"] == 1
        assert second.metrics["search_queries_executed"] == 1
        assert "first_turn_queries_requested" not in second.metrics
        answer = await env.step({"role": "assistant", "content": "Answer: yes"})
        assert answer.episode_done and answer.reward == 1.0
        assert json.loads(answer.logs["search_calls"]) == []

    async def run_both():
        await asyncio.gather(episode("one"), episode("two"))

    asyncio.run(run_both())
    assert fetch.await_count == 4
    fetch.assert_has_awaits(
        [
            call(["one", "重复", "重复"]),
            call(["two", "重复", "重复"]),
            call(["one"]),
            call(["two"]),
        ],
        any_order=True,
    )


@pytest.mark.parametrize("bad", ['["stringified"]', ["a", "b", "c", 4]])
def test_invalid_first_search_keeps_validation_and_never_falls_back(backend, bad):
    retriever, fetch = backend
    env = make_env(retriever)
    step = asyncio.run(env.step(search(bad, ["also unused"])))
    fetch.assert_not_awaited()
    assert not step.episode_done
    assert count_tool_errors(env.history) == 1
    assert step.metrics["search_calls_requested"] == 2
    assert step.metrics["search_calls_executed"] == 0
    assert [r["status"] for r in json.loads(step.logs["search_calls"])] == [
        "invalid_arguments",
        "skipped",
    ]


def test_phase_one_executes_all_queries_and_schema_only_loses_one_sentence(backend):
    retriever, fetch = backend
    env = make_env(retriever, limit_search=False)
    asyncio.run(env.step(search(["a", "b", "c", "d"], ["e"])))
    fetch.assert_has_awaits([call(["a", "b", "c", "d"]), call(["e"])])
    old_spec = search_tool_spec(False)
    assert old_spec == env._tool_dict["search"].to_spec()
    param = old_spec["parameters"]["properties"]["query_list"]
    assert param["description"] == (
        "A list of fully-formed semantic queries. The tool will return search results for each query."
    )
    param["description"] = "A list of fully-formed semantic queries."
    assert search_tool_spec(True) == old_spec


def test_eval_records_requested_and_dispatched_calls_without_changing_the_score(backend):
    retriever, fetch = backend
    item = {"id": "q", "question": "Who?", "answer": ["yes"], "data_source": "nq"}
    renderer = Mock()
    renderer.get_stop_sequences.return_value = [2]
    renderer.create_conversation_prefix_with_tools.return_value = []
    renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1])
    renderer.parse_response.side_effect = [
        (search(["first"], ["unused"]), ParseTermination.STOP_SEQUENCE),
        ({"role": "assistant", "content": "Answer: yes"}, ParseTermination.STOP_SEQUENCE),
    ]
    record = asyncio.run(
        offline_eval.evaluate_single_item(
            item,
            retriever,
            AsyncMock(return_value=TokensWithLogprobs([2], [0.0])),
            renderer,
            Mock(),
            offline_eval.CLIConfig(limit_search=True),
            asyncio.Semaphore(1),
        )
    )
    fetch.assert_awaited_once_with(["first"])
    assert record["reward"] == record["em"] == 1.0
    assert record["num_search_calls"] == 2 and record["queries"] == ["first", "unused"]
    assert record["num_tool_errors"] == 0
    first, last = record["turns"]
    assert [c["executed_queries"] for c in first["search_calls"]] == [["first"], []]
    assert last["search_calls"] == []


def test_training_entrypoint_and_saved_eval_settings(tmp_path, monkeypatch):
    run_training = AsyncMock()
    monkeypatch.setattr("search_r1.training.runtime.main", run_training)
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
                batch_size=1,
                group_size=2,
                skyrl_compat=False,
                wandb_project=None,
            )
        )
    )
    config = run_training.await_args.args[0]
    dataset, _ = asyncio.run(config.dataset_builder())
    envs = asyncio.run(dataset.get_batch(0)[0].make_envs())
    assert envs[0].message_env is not envs[1].message_env
    assert envs[0].message_env.retriever is envs[1].message_env.retriever
    renderer.create_conversation_prefix_with_tools.assert_called_with(
        tools=[search_tool_spec(True)], system_prompt=search_env.SEARCH_TASK_INSTRUCTIONS_FAST
    )
    snapshot = chz.asdict(config)
    assert eval_settings_from_training_config(snapshot)["limit_search"] is True
    del snapshot["dataset_builder"]["limit_search"]
    assert eval_settings_from_training_config(snapshot)["limit_search"] is False


class FailingSession:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def post(self, *args, **kwargs):
        self.calls += 1
        raise OSError("Too many open files")


def test_exhausted_retries_raise_retrieval_unavailable():
    retriever = BM25Tool("http://unused", max_retries=3, initial_retry_delay=0)
    session = FailingSession()
    retriever._session = session
    with pytest.raises(RetrievalUnavailable, match="after 3 attempts") as caught:
        asyncio.run(retriever.retrieve(["q"]))
    assert session.calls == 3
    assert isinstance(caught.value.__cause__, OSError)


def outage(retriever, monkeypatch):
    fetch = AsyncMock(side_effect=RetrievalUnavailable("Retrieval failed after 10 attempts"))
    monkeypatch.setattr(retriever, "_retrieve_with_retry", fetch)
    return fetch


@pytest.mark.parametrize("limit_search", [False, True])
def test_retrieval_outage_raises_from_step(monkeypatch, limit_search):
    retriever = BM25Tool("http://unused")
    fetch = outage(retriever, monkeypatch)
    env = make_env(retriever, limit_search=limit_search)
    with pytest.raises(RetrievalUnavailable):
        asyncio.run(env.step(search(["a"])))
    fetch.assert_awaited_once_with(["a"])


def test_invalid_arguments_still_reach_the_model_as_a_tool_error(monkeypatch):
    retriever = BM25Tool("http://unused")
    fetch = outage(retriever, monkeypatch)
    env = make_env(retriever, limit_search=False)
    result = asyncio.run(env.step(search("not a list")))
    fetch.assert_not_awaited()
    assert not result.episode_done
    assert count_tool_errors(env.history) == 1


def test_training_rollout_fails_on_retrieval_outage(monkeypatch):
    retriever = BM25Tool("http://unused")
    outage(retriever, monkeypatch)
    renderer = Mock()
    renderer.get_stop_sequences.return_value = [2]
    renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1])
    renderer.parse_response.return_value = (search(["a"]), ParseTermination.STOP_SEQUENCE)
    env = build_search_env(
        renderer=renderer,
        search_tool=retriever,
        initial_messages=[{"role": "user", "content": "question"}],
        reward_fn=TextAnswerReward(["yes"]),
        max_turns=5,
        max_trajectory_tokens=32768,
        max_generation_tokens=None,
        context_overflow_reward=-0.1,
        limit_search=True,
        example_id="q",
    )
    policy = AsyncMock(return_value=TokensWithLogprobs([2], [0.0]))
    with pytest.raises(RetrievalUnavailable):
        asyncio.run(do_single_rollout(policy, env))
    # No further turn is sampled on the failed search.
    policy.assert_awaited_once()


def test_eval_records_a_retrieval_outage_as_an_error(monkeypatch):
    retriever = BM25Tool("http://unused")
    outage(retriever, monkeypatch)
    item = {"id": "q", "question": "Who?", "answer": ["yes"], "data_source": "nq"}
    renderer = Mock()
    renderer.get_stop_sequences.return_value = [2]
    renderer.create_conversation_prefix_with_tools.return_value = []
    renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1])
    renderer.parse_response.return_value = (search(["a"]), ParseTermination.STOP_SEQUENCE)
    record = asyncio.run(
        offline_eval.evaluate_single_item(
            item,
            retriever,
            AsyncMock(return_value=TokensWithLogprobs([2], [0.0])),
            renderer,
            Mock(),
            offline_eval.CLIConfig(limit_search=False),
            asyncio.Semaphore(1),
        )
    )
    assert record["id"] == "q" and "RetrievalUnavailable" in record["error"]
    assert "reward" not in record
