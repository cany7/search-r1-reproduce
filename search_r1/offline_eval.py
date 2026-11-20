# Adapted from tinker-cookbook/recipes/search_tool/offline_eval.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local components, BM25 search and project evaluation/training rules.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.
"""Offline evaluation for the search agent.

Extends the recipe's ``offline_eval.py``:

- two sampling backends: Tinker (cloud, the numbers that go into the report) and a
  local OpenAI-compatible server (oMLX; debugging and base-model selection only);
- a fixed evaluation set can be passed with ``eval_file``;
- every trajectory is written to disk with its termination reason, search queries,
  token counts and transcript, so a low score can be traced to its cause.

Outputs go to ``runs/eval/<run_name>/``: ``config.json``, ``trajectories.jsonl``
and ``summary.json``.
"""

import asyncio
import dataclasses
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import chz
import tinker
from dotenv import load_dotenv

from search_r1.backends import tinker_backend as checkpoint_utils
from search_r1.backends.completers import TokenCompleter
from search_r1.backends.local_completer import OpenAICompletionsTokenCompleter
from search_r1.backends.tinker_backend import (
    TinkerEvaluationPolicy,
    close_service,
    load_training_client,
)
from search_r1.env import tokenizer as tokenizer_utils
from search_r1.env.renderers import DEFAULT_RENDERER_NAME, Renderer, get_renderer, get_text_content
from search_r1.env.renderers.base import Message
from search_r1.env.rollout import do_single_rollout
from search_r1.env.search_env import (
    DEFAULT_RETRIEVAL_URL,
    PromptName,
    SearchR1Datum,
    download_search_r1_dataset,
    get_task_instructions,
    search_initial_messages,
)
from search_r1.env.search_tool_env import build_search_env, search_tool_spec
from search_r1.env.tokenizer import Tokenizer
from search_r1.env.tools import (
    BM25Tool,
    RetrievalConfig,
    TextAnswerReward,
    count_tool_errors,
    extract_search_queries,
    normalize_answer,
)
from search_r1.env.types import StopReason, Trajectory
from search_r1.eval_config import eval_settings_from_training_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Termination = Literal[
    "answered",  # finished on its own with a well-formed answer
    "no_valid_answer",  # finished on its own, but no well-formed "Answer:"
    "max_turns",  # still calling the tool when the turn cap was reached
    "turn_truncated",  # a single turn hit the per-turn generation cap
    "context_overflow",  # the conversation outgrew the trajectory token budget
    "parse_error",  # the response or its tool calls could not be parsed
    "unknown",
]


@chz.chz
class CLIConfig:
    training_config_file: str | None = chz.field(
        default=None,
        doc="Saved training config.json. Its model, renderer, sampling, reward, retrieval and "
        "rollout settings override the corresponding CLI values; eval data and concurrency do not change.",
    )
    # Sampling backend
    backend: Literal["tinker", "omlx"] = chz.field(
        default="tinker", doc="tinker: Tinker / SkyRL sampling client. omlx: local server."
    )
    base_model: str = chz.field(default="Qwen/Qwen3.5-4B", doc="HF name, for tokenizer")
    renderer_name: str = chz.field(
        default=DEFAULT_RENDERER_NAME,
        doc="Renderer to use (Tinker: only if the checkpoint does not record one)",
    )
    tinker_checkpoint_url: str | None = chz.field(
        default=None,
        doc="Tinker sampler checkpoint to evaluate; None evaluates the base model. "
        "Not reliable on self-hosted SkyRL (FSDP): it ignores the checkpoint and samples the "
        "model's current weights, or fails after a restart. Use tinker_state_url there.",
    )
    skyrl_base_eval: bool = chz.field(
        default=False,
        doc="Keep a LoRA client and its SDK session heartbeat alive while sampling base_model "
        "on a fresh SkyRL combined runtime.",
    )
    tinker_state_url: str | None = chz.field(
        default=None,
        doc="Full training state to evaluate (tinker://<model>/weights/<name>). Loads it into a "
        "new LoRA training client and syncs the weights to the sampler. Use this on SkyRL; "
        "the server must have no other model loaded.",
    )
    lora_rank: int = chz.field(default=32, doc="LoRA rank of the state loaded by tinker_state_url")
    tinker_base_url: str | None = chz.field(
        default=None, doc="Tinker API base URL, e.g. a self-hosted SkyRL service"
    )
    omlx_model: str | None = chz.field(
        default=None, doc="Model id as the local server lists it (backend=omlx)"
    )

    # Prompt and reward. Defaults are the training configuration; the recipe's
    # originals are prompt_name=original answer_score=em no_search_coef=0.0.
    # EM and F1 are recorded either way; these only change the reported reward.
    prompt_name: PromptName = chz.field(default="fast", doc="System prompt variant")
    answer_score: Literal["f1", "em"] = chz.field(default="f1", doc="Answer term of the reward")
    format_coef: float = chz.field(default=0.1, doc="Penalty for a missing well-formed answer")
    no_search_coef: float = chz.field(default=0.1, doc="Penalty when no search was made")

    # Evaluation data
    eval_file: str | None = chz.field(
        default=None, doc="JSONL of a fixed evaluation set; overrides the sampling options below"
    )
    limit_per_source: int | None = chz.field(
        default=None, doc="With eval_file: evaluate only the first k items of each data source"
    )
    max_eval_samples: int = chz.field(
        default=100, doc="Maximum number of samples to evaluate per data source"
    )
    seed: int = chz.field(default=42, doc="Random seed for sampling")
    split: Literal["train", "test"] = chz.field(default="test", doc="Dataset split to use")

    # Rollout limits, matching train.py
    max_tokens: int = chz.field(default=1024, doc="Maximum number of tokens to generate per turn")
    max_turns: int = chz.field(default=5, doc="Maximum number of assistant turns")
    max_trajectory_tokens: int = chz.field(
        default=32 * 1024,
        doc="Token budget for the full multi-turn trajectory, matching train.py; "
        "episodes that would exceed it end instead of overflowing the model's context window",
    )
    max_generation_tokens: int | None = chz.field(
        default=None, doc="Optional generation reserve for the environment's context budget"
    )
    context_overflow_reward: float = chz.field(default=-0.1, doc="Reward on context overflow")
    temperature: float = chz.field(default=1.0, doc="Sampling temperature")

    # Infrastructure
    retrieval_url: str | None = chz.field(
        default=None, doc="BM25 retrieval server; defaults to $RETRIEVAL_URL"
    )
    n_results: int = chz.field(default=3, doc="Retrieved documents per search query")
    limit_search: bool = chz.field(
        default=False, doc="Run only the first search per turn and its first three queries."
    )
    concurrency: int = chz.field(default=8, doc="Maximum rollouts in flight")
    run_name: str | None = chz.field(default=None, doc="Output directory name under runs/eval")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def split_data_by_source(data: list[SearchR1Datum]) -> dict[str, list[SearchR1Datum]]:
    """Split data by data source."""
    data_by_source = defaultdict(list)
    for item in data:
        data_by_source[item["data_source"]].append(item)
    return dict(data_by_source)


def sample_k_from_each_source(
    data_by_source: dict[str, list[SearchR1Datum]], k: int, seed: int = 42
) -> dict[str, list[SearchR1Datum]]:
    """Sample K items from each data source."""
    random.seed(seed)
    sampled_data = {}
    total_samples = 0

    for source, items in data_by_source.items():
        if len(items) <= k:
            sampled_data[source] = items
        else:
            sampled_data[source] = random.sample(items, k)
        total_samples += len(sampled_data[source])
        print(f"{source}: {len(items)} -> {len(sampled_data[source])} samples")

    print(f"Total samples: {total_samples}")
    return sampled_data


def load_eval_items(config: CLIConfig) -> list[SearchR1Datum]:
    if config.eval_file is not None:
        with open(config.eval_file) as f:
            items: list[SearchR1Datum] = [json.loads(line) for line in f if line.strip()]
        if config.limit_per_source is not None:
            by_source = split_data_by_source(items)
            items = [i for group in by_source.values() for i in group[: config.limit_per_source]]
        print(f"Loaded {len(items)} items from {config.eval_file}")
        return items

    print(f"Loading {config.split} split...")
    data = download_search_r1_dataset(config.split)
    print(f"Sampling up to {config.max_eval_samples} samples from each source...")
    sampled = sample_k_from_each_source(
        split_data_by_source(data), config.max_eval_samples, config.seed
    )
    return [item for items in sampled.values() for item in items]


# ---------------------------------------------------------------------------
# Per-trajectory record
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    return str(obj)


def classify_termination(
    stop_reason: str | None, last_turn_stop_reason: str | None, has_answer: bool
) -> Termination:
    if stop_reason == StopReason.COMPLETED:
        return "answered" if has_answer else "no_valid_answer"
    if stop_reason == StopReason.MAX_TURNS:
        return "max_turns"
    if stop_reason == StopReason.MAX_TOKENS:
        # MAX_TOKENS covers both a turn cut off mid-generation and a conversation
        # that no longer fits the trajectory budget; the sampler's own stop
        # reason tells them apart.
        return "turn_truncated" if last_turn_stop_reason == "length" else "context_overflow"
    if stop_reason == StopReason.CONTEXT_OVERFLOW:
        return "context_overflow"
    if stop_reason == StopReason.PARSE_ERROR:
        return "parse_error"
    return "unknown"


def build_record(
    item: SearchR1Datum,
    num_initial_messages: int,
    history: list[Message],
    trajectory: Trajectory,
    reward_fn: TextAnswerReward,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    transitions = trajectory.transitions
    generated = history[num_initial_messages:]
    final_metrics = transitions[-1].metrics if transitions else {}
    last_turn_stop_reason = transitions[-1].ac.stop_reason if transitions else None

    num_search_calls, queries = extract_search_queries(generated)
    normalized_queries = [normalize_answer(q) for q in queries]

    # The final answer, if the episode ended with a plain assistant message.
    answer = None
    assistant_messages = [m for m in generated if m.get("role") == "assistant"]
    if assistant_messages and not assistant_messages[-1].get("tool_calls"):
        answer = reward_fn._extract_answer(get_text_content(assistant_messages[-1]))

    turns = []
    for t in transitions:
        turn = {
            "prompt_tokens": t.ob.length,
            "sampled_tokens": len(t.ac.tokens),
            "stop_reason": t.ac.stop_reason,
        }
        if "search_calls" in t.logs:
            turn["search_calls"] = json.loads(t.logs["search_calls"])
        turns.append(turn)

    has_answer = final_metrics.get("format", 0.0) == 1.0
    return {
        "id": item.get("id", ""),
        "data_source": item["data_source"],
        "question": item["question"],
        "gold_answers": item["answer"],
        "answer": answer,
        "has_answer": has_answer,
        "em": final_metrics.get("correct", 0.0),
        "f1": final_metrics.get("f1", 0.0),
        "reward": sum(t.reward for t in transitions),
        "termination": classify_termination(
            trajectory.stop_reason, last_turn_stop_reason, has_answer
        ),
        "stop_reason": trajectory.stop_reason,
        "num_turns": len(transitions),
        "num_search_calls": num_search_calls,
        # Calls answered with an error (bad arguments); no search ran for these.
        "num_tool_errors": count_tool_errors(generated),
        "queries": queries,
        "repeated_query": len(set(normalized_queries)) < len(normalized_queries),
        "parse_error": any(t.metrics.get("parse_error", 0.0) > 0 for t in transitions),
        "sampled_tokens": sum(len(t.ac.tokens) for t in transitions),
        "turns": turns,
        # Messages produced during the rollout. The system prompt and tool
        # declaration are the same for every item and live in config.json.
        "messages": _to_jsonable(generated),
        # Raw text of the last sampled turn. A truncated or unparseable turn is
        # not in ``messages``, so this is the only place to see it.
        "last_turn_text": tokenizer.decode(transitions[-1].ac.tokens) if transitions else "",
    }


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


async def evaluate_single_item(
    item: SearchR1Datum,
    search_tool: BM25Tool,
    policy: TokenCompleter,
    renderer: Renderer,
    tokenizer: Tokenizer,
    config: CLIConfig,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    initial_messages = search_initial_messages(
        item, renderer, config.prompt_name, config.limit_search
    )
    reward_fn = TextAnswerReward(
        gold_answers=item["answer"],
        format_coef=config.format_coef,
        answer_score=config.answer_score,
        no_search_coef=config.no_search_coef,
    )

    env = build_search_env(
        renderer=renderer,
        search_tool=search_tool,
        initial_messages=initial_messages,
        reward_fn=reward_fn,
        max_turns=config.max_turns,
        max_trajectory_tokens=config.max_trajectory_tokens,
        max_generation_tokens=config.max_generation_tokens,
        context_overflow_reward=config.context_overflow_reward,
        limit_search=config.limit_search,
        example_id=item["id"],
    )
    async with semaphore:
        try:
            trajectory = await do_single_rollout(policy, env)
        except Exception as e:
            return {
                "id": item.get("id", ""),
                "data_source": item["data_source"],
                "question": item["question"],
                "error": repr(e),
            }

    # AgentToolMessageEnv keeps the parsed conversation.
    history: list[Message] = env.message_env.history  # type: ignore[attr-defined]
    return build_record(item, len(initial_messages), history, trajectory, reward_fn, tokenizer)


async def build_policy(config: CLIConfig) -> tuple[TokenCompleter, str]:
    """Return the sampling policy and the renderer name to use."""
    if config.skyrl_base_eval and (
        config.backend != "tinker" or config.tinker_state_url or config.tinker_checkpoint_url
    ):
        raise ValueError("skyrl_base_eval requires the Tinker backend and base weights")
    if config.backend == "omlx":
        if config.omlx_model is None:
            raise ValueError("backend=omlx requires omlx_model")
        tokenizer = tokenizer_utils.get_tokenizer(config.base_model)
        policy: TokenCompleter = OpenAICompletionsTokenCompleter(
            base_url=os.environ["OMLX_BASE_URL"],
            api_key=os.environ["OMLX_API_KEY"],
            model=config.omlx_model,
            tokenizer=tokenizer,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_concurrency=config.concurrency,
        )
        return policy, config.renderer_name

    if config.tinker_state_url is not None and config.tinker_checkpoint_url is not None:
        raise ValueError("Pass only one of tinker_state_url and tinker_checkpoint_url")
    service_client = tinker.ServiceClient(
        user_metadata={"recipe_name": "eval_search_tool_offline"},
        base_url=config.tinker_base_url,
    )
    try:
        renderer_name = None
        training_client = None
        if config.tinker_state_url is not None:
            # Explicit create/load supports the older SkyRL protocol without patching SDK classes.
            training_client = await load_training_client(
                service_client, config.tinker_state_url, config.base_model, config.lora_rank
            )
            sampling_client = await training_client.save_weights_and_get_sampling_client()
            renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(
                service_client, config.tinker_state_url
            )
        elif config.tinker_checkpoint_url is not None:
            sampling_client = await service_client.create_sampling_client(
                model_path=config.tinker_checkpoint_url
            )
            renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(
                service_client, config.tinker_checkpoint_url
            )
        else:
            if config.skyrl_base_eval:
                # Creating a training client starts the pinned SDK's session heartbeat.
                # Retain both clients until policy_session has drained all rollouts.
                training_client = await service_client.create_lora_training_client(
                    config.base_model, rank=config.lora_rank
                )
            sampling_client = await service_client.create_sampling_client(
                base_model=config.base_model
            )
        policy = TinkerEvaluationPolicy(
            sampling_client,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            service_client=service_client,
            training_client=training_client,
        )
        return policy, config.renderer_name if config.training_config_file else (
            renderer_name or config.renderer_name
        )
    except BaseException as error:
        status = (
            "interrupted"
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
            else "errored"
        )
        await close_service(service_client, status)
        raise


@asynccontextmanager
async def policy_session(config: CLIConfig):
    """Close clients after all rollouts, also on setup failure or cancellation."""
    policy, renderer_name = await build_policy(config)
    status = "success"
    try:
        yield policy, renderer_name
    except (asyncio.CancelledError, KeyboardInterrupt):
        status = "interrupted"
        raise
    except BaseException:
        status = "errored"
        raise
    finally:
        if isinstance(policy, TinkerEvaluationPolicy):
            await policy.aclose(status)
        elif isinstance(policy, OpenAICompletionsTokenCompleter):
            try:
                await policy.aclose()
            except Exception:
                logger.warning("Could not close local evaluation client", exc_info=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate records. Splits the score into answer rate and accuracy when answered."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    answered = [r for r in records if r["has_answer"]]

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / n

    return {
        "n": n,
        "em": mean("em"),
        "f1": mean("f1"),
        "answer_rate": len(answered) / n,
        "em_when_answered": (sum(r["em"] for r in answered) / len(answered)) if answered else None,
        "f1_when_answered": (sum(r["f1"] for r in answered) / len(answered)) if answered else None,
        "reward": mean("reward"),
        "termination": dict(Counter(r["termination"] for r in records)),
        "search_calls": mean("num_search_calls"),
        "tool_error_rate": sum(r["num_tool_errors"] > 0 for r in records) / n,
        "no_search_rate": sum(r["num_search_calls"] == 0 for r in records) / n,
        "repeated_query_rate": mean("repeated_query"),
        "parse_error_rate": mean("parse_error"),
        "turns": mean("num_turns"),
        "sampled_tokens": mean("sampled_tokens"),
    }


def print_summary(summary_by_source: dict[str, dict[str, Any]]) -> None:
    header = (
        f"{'Dataset':<17}{'N':>5}{'EM':>8}{'F1':>8}{'Answered':>10}{'EM|ans':>8}"
        f"{'Searches':>10}{'NoSearch':>10}{'Tokens':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for source, s in summary_by_source.items():
        if source == "OVERALL":
            print("-" * len(header))
        em_ans = f"{s['em_when_answered']:.3f}" if s.get("em_when_answered") is not None else "-"
        print(
            f"{source:<17}{s['n']:>5}{s['em']:>8.3f}{s['f1']:>8.3f}{s['answer_rate']:>10.3f}{em_ans:>8}"
            f"{s['search_calls']:>10.2f}{s['no_search_rate']:>10.3f}{s['sampled_tokens']:>8.0f}"
        )
    print("=" * len(header))
    print("Termination:", summary_by_source["OVERALL"]["termination"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def cli_main(config: CLIConfig) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    training_config = None
    if config.training_config_file is not None:
        training_config = json.loads(Path(config.training_config_file).read_text())
        config = chz.replace(config, **eval_settings_from_training_config(training_config))
    items = load_eval_items(config)

    async with policy_session(config) as (policy, renderer_name):
        await _evaluate_with_policy(config, items, training_config, policy, renderer_name)


async def _evaluate_with_policy(config, items, training_config, policy, renderer_name):
    tokenizer = tokenizer_utils.get_tokenizer(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)
    print(f"Using renderer: {renderer_name}")

    retrieval_url = config.retrieval_url or os.environ.get("RETRIEVAL_URL", DEFAULT_RETRIEVAL_URL)
    search_tool = BM25Tool(
        retrieval_url=retrieval_url,
        retrieval_config=RetrievalConfig(n_results=config.n_results),
    )

    async with aclosing(search_tool):
        model_tag = (config.omlx_model or config.base_model).replace("/", "-")
        run_name = config.run_name or f"{model_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = PROJECT_ROOT / "runs" / "eval" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(
                {
                    **chz.asdict(config),
                    # Snapshot the source before the next training segment rewrites config.json.
                    "training_config": training_config,
                    "resolved_renderer_name": renderer_name,
                    "retrieval_url": retrieval_url,
                    "system_prompt": get_task_instructions(config.prompt_name),
                    "tools": [search_tool_spec(config.limit_search)],
                    "num_items": len(items),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Evaluating {len(items)} items (concurrency {config.concurrency})")
        semaphore = asyncio.Semaphore(config.concurrency)
        start = time.monotonic()
        records: list[dict[str, Any]] = []
        tasks = [
            asyncio.create_task(
                evaluate_single_item(
                    item, search_tool, policy, renderer, tokenizer, config, semaphore
                )
            )
            for item in items
        ]
        try:
            with open(output_dir / "trajectories.jsonl", "w") as f:
                for done_count, task in enumerate(asyncio.as_completed(tasks), start=1):
                    record = await task
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    if done_count % 10 == 0 or done_count == len(tasks):
                        print(f"  {done_count}/{len(tasks)} done ({time.monotonic() - start:.0f}s)")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start

        errors = [r for r in records if "error" in r]
        ok = [r for r in records if "error" not in r]
        summary_by_source = {
            source: summarize([r for r in ok if r["data_source"] == source])
            for source in sorted({r["data_source"] for r in ok})
        }
        summary_by_source["OVERALL"] = summarize(ok)

        total_sampled = sum(r["sampled_tokens"] for r in ok)
        run_stats: dict[str, Any] = {
            "elapsed_seconds": elapsed,
            "num_errors": len(errors),
            "total_sampled_tokens": total_sampled,
            "sampled_tokens_per_second": total_sampled / elapsed if elapsed > 0 else None,
        }
        if isinstance(policy, OpenAICompletionsTokenCompleter):
            run_stats["sampling_calls"] = policy.num_calls
            run_stats["token_count_mismatches"] = policy.token_count_mismatches
        with open(output_dir / "summary.json", "w") as f:
            json.dump(
                {"by_source": summary_by_source, "run": run_stats}, f, ensure_ascii=False, indent=2
            )

        if ok:
            print_summary(summary_by_source)
        print(f"Run: {run_stats}")
        if errors:
            print(f"WARNING: {len(errors)} rollouts raised an error; first: {errors[0]['error']}")
        print(f"Output: {output_dir}")


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))
