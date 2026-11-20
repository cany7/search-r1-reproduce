"""SDK boundaries and evaluation cleanup, using in-memory clients only."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
from migration_cases import FixedTokenizer

from search_r1 import offline_eval
from search_r1.backends.completers import TinkerTokenCompleter, TokensWithLogprobs
from search_r1.backends.tinker_backend import (
    get_renderer_name_from_checkpoint_async,
    load_training_client,
)


@pytest.fixture
def sdk(monkeypatch):
    sampler = SimpleNamespace(sample=AsyncMock())
    training = SimpleNamespace(
        load_state=AsyncMock(),
        load_state_with_optimizer=AsyncMock(),
        save_weights_and_get_sampling_client=AsyncMock(return_value=sampler),
    )
    rest = SimpleNamespace(
        get_training_run_by_tinker_path=AsyncMock(
            return_value=SimpleNamespace(user_metadata={"renderer_name": "qwen3_5"}),
        )
    )
    service = SimpleNamespace(
        create_lora_training_client=AsyncMock(return_value=training),
        create_sampling_client=AsyncMock(return_value=sampler),
        create_rest_client=Mock(return_value=rest),
        close=AsyncMock(),
    )
    factory = Mock(return_value=service)
    monkeypatch.setattr(offline_eval.tinker, "ServiceClient", factory)
    return SimpleNamespace(
        service=service, training=training, sampler=sampler, rest=rest, factory=factory
    )


@pytest.mark.parametrize("with_optimizer", [False, True])
def test_load_state_uses_explicit_sdk_sequence(sdk, with_optimizer):
    calls = []

    async def create(*args, **kwargs):
        calls.append("create")
        return sdk.training

    async def load(path):
        calls.append(path)

    sdk.service.create_lora_training_client.side_effect = create
    method = sdk.training.load_state_with_optimizer if with_optimizer else sdk.training.load_state
    method.side_effect = load
    actual = asyncio.run(
        load_training_client(
            sdk.service, "tinker://m/weights/7", "Qwen/test", 16, with_optimizer=with_optimizer
        )
    )
    assert actual is sdk.training and calls == ["create", "tinker://m/weights/7"]
    sdk.service.create_lora_training_client.assert_awaited_once_with("Qwen/test", rank=16)
    other = sdk.training.load_state if with_optimizer else sdk.training.load_state_with_optimizer
    other.assert_not_called()


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"renderer_name": "qwen3_5"}, "qwen3_5"),
        ({}, None),
        (None, None),
    ],
)
def test_renderer_metadata(sdk, metadata, expected):
    sdk.rest.get_training_run_by_tinker_path.return_value = SimpleNamespace(user_metadata=metadata)
    assert (
        asyncio.run(get_renderer_name_from_checkpoint_async(sdk.service, "tinker://m/weights/7"))
        == expected
    )


@pytest.mark.parametrize("error", [ValueError("old server"), tinker.TinkerError("unsupported")])
def test_renderer_metadata_falls_back_for_old_protocol(sdk, error):
    sdk.rest.get_training_run_by_tinker_path.side_effect = error
    assert (
        asyncio.run(get_renderer_name_from_checkpoint_async(sdk.service, "tinker://m/weights/7"))
        is None
    )


@pytest.mark.parametrize("mode", ["base", "sampler", "state"])
def test_policy_owns_clients_until_session_ends(sdk, mode):
    config = offline_eval.CLIConfig(
        tinker_state_url="tinker://m/weights/7" if mode == "state" else None,
        tinker_checkpoint_url="tinker://m/sampler_weights/7" if mode == "sampler" else None,
    )

    async def run():
        async with offline_eval.policy_session(config) as (policy, renderer):
            assert policy.service_client is sdk.service
            assert policy.training_client is (sdk.training if mode == "state" else None)
            assert policy.sampling_client is sdk.sampler
            assert renderer == (config.renderer_name if mode == "base" else "qwen3_5")
            sdk.service.close.assert_not_called()

    asyncio.run(run())
    sdk.service.close.assert_awaited_once_with(status="success")
    if mode == "state":
        sdk.training.load_state.assert_awaited_once_with(config.tinker_state_url)
        sdk.training.load_state_with_optimizer.assert_not_called()
        sdk.training.save_weights_and_get_sampling_client.assert_awaited_once()
        sdk.service.create_sampling_client.assert_not_called()
    else:
        sdk.service.create_lora_training_client.assert_not_called()
        sdk.service.create_sampling_client.assert_awaited_once_with(
            **(
                {"base_model": config.base_model}
                if mode == "base"
                else {"model_path": config.tinker_checkpoint_url}
            )
        )


def test_conflicting_paths_fail_before_creating_service(sdk):
    config = offline_eval.CLIConfig(tinker_state_url="state", tinker_checkpoint_url="sampler")
    with pytest.raises(ValueError, match="only one"):
        asyncio.run(offline_eval.build_policy(config))
    sdk.factory.assert_not_called()


def test_failed_load_closes_service_and_preserves_original_error(sdk, caplog):
    sdk.training.load_state.side_effect = ValueError("load failed")
    sdk.service.close.side_effect = RuntimeError("old server close failed")
    with pytest.raises(ValueError, match="load failed"):
        asyncio.run(offline_eval.build_policy(offline_eval.CLIConfig(tinker_state_url="state")))
    sdk.service.close.assert_awaited_once_with(status="errored")
    assert "Could not finish" in caplog.text


@pytest.mark.parametrize(
    "error, status",
    [(ValueError("render failed"), "errored"), (asyncio.CancelledError(), "interrupted")],
)
def test_policy_session_cleanup_on_error_or_cancel(sdk, error, status):
    async def run():
        async with offline_eval.policy_session(offline_eval.CLIConfig()):
            raise error

    with pytest.raises(type(error)):
        asyncio.run(run())
    sdk.service.close.assert_awaited_once_with(status=status)


def test_sampling_budget_and_provenance_are_preserved(sdk):
    sdk.sampler.sample.return_value = SimpleNamespace(
        sequences=[
            SimpleNamespace(
                tokens=[3, 4], logprobs=[-0.2, -0.3], stop_reason="length", sequence_id="seq"
            )
        ],
        prompt_cache_hit_tokens=7,
    )
    policy = TinkerTokenCompleter(sdk.sampler, max_tokens=10, temperature=0.7, context_window=12)
    prompt = tinker.ModelInput.from_ints([1] * 8)
    result = asyncio.run(policy(prompt, [2], max_tokens=6))
    args = sdk.sampler.sample.await_args.kwargs
    assert args["prompt"] is prompt and args["num_samples"] == 1
    assert (
        args["sampling_params"].max_tokens,
        args["sampling_params"].stop,
        args["sampling_params"].temperature,
    ) == (4, [2], 0.7)
    assert result == TokensWithLogprobs([3, 4], [-0.2, -0.3], "length", "seq", 7)
    with pytest.raises(ValueError, match="No room"):
        asyncio.run(policy(tinker.ModelInput.from_ints([1] * 12), [2]))
    assert sdk.sampler.sample.await_count == 1


def test_eval_writes_records_with_local_renderer_and_closes_clients(tmp_path, monkeypatch, sdk):
    tokenizer = FixedTokenizer()
    monkeypatch.setattr(offline_eval, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(offline_eval.tokenizer_utils, "get_tokenizer", lambda _: tokenizer)
    item = {"id": "q", "data_source": "nq", "question": "Who?", "answer": ["yes"]}
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps(item) + "\n")
    tokens = tokenizer.encode("Answer: yes<|im_end|>")
    sdk.sampler.sample.return_value = SimpleNamespace(
        sequences=[
            SimpleNamespace(
                tokens=tokens, logprobs=[-0.1] * len(tokens), stop_reason="stop", sequence_id=None
            )
        ],
        prompt_cache_hit_tokens=0,
    )
    tool = offline_eval.BM25Tool("http://unused.invalid")
    tool.aclose = AsyncMock()
    monkeypatch.setattr(offline_eval, "BM25Tool", lambda **kwargs: tool)
    asyncio.run(
        offline_eval.cli_main(offline_eval.CLIConfig(eval_file=str(source), run_name="test"))
    )
    output = tmp_path / "runs/eval/test"
    record = json.loads((output / "trajectories.jsonl").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert record["termination"] == "answered" and record["em"] == 1.0
    assert summary["run"]["num_errors"] == 0 and summary["by_source"]["OVERALL"]["n"] == 1
    tool.aclose.assert_awaited_once()
    sdk.service.close.assert_awaited_once_with(status="success")


def test_cancelled_eval_drains_rollouts_before_closing_resources(tmp_path, monkeypatch, sdk):
    monkeypatch.setattr(offline_eval, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(offline_eval.tokenizer_utils, "get_tokenizer", lambda _: FixedTokenizer())
    item = {"id": "q", "data_source": "nq", "question": "Who?", "answer": ["yes"]}
    monkeypatch.setattr(offline_eval, "load_eval_items", lambda _: [item, item])
    events = []

    async def run():
        both_started = asyncio.Event()
        started = 0

        async def evaluate(*args):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append("rollout_done")

        async def close_tool():
            events.append("tool_closed")

        tool = SimpleNamespace(aclose=close_tool)
        monkeypatch.setattr(offline_eval, "BM25Tool", lambda **kwargs: tool)
        monkeypatch.setattr(offline_eval, "evaluate_single_item", evaluate)
        sdk.service.close.side_effect = lambda **kwargs: events.append("sdk_closed")
        task = asyncio.create_task(offline_eval.cli_main(offline_eval.CLIConfig(run_name="cancel")))
        await asyncio.wait_for(both_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert events == ["rollout_done", "rollout_done", "tool_closed", "sdk_closed"]
    sdk.service.close.assert_awaited_once_with(status="interrupted")
