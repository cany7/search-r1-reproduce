# Adapted from tinker-cookbook/recipes/search_tool/search_env.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local components, BM25 search and project evaluation/training rules.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.
from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

import chz
import pandas as pd
from huggingface_hub import hf_hub_download

from search_r1.env import renderers as model_info
from search_r1.env import tokenizer as tokenizer_utils
from search_r1.env.renderers import get_renderer
from search_r1.env.renderers.base import Message, Renderer
from search_r1.env.search_tool_env import (
    MAX_TURNS_UNANSWERED_LOG,
    build_search_env,
    search_tool_spec,
)
from search_r1.env.tools import BM25Tool, RetrievalConfig, TextAnswerReward
from search_r1.env.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)

# Project data directory (datasets, index, corpus). Override with SEARCH_R1_DATA_DIR.
DATA_DIR = Path(
    os.environ.get("SEARCH_R1_DATA_DIR", Path(__file__).resolve().parents[2] / "data")
)
DEFAULT_RETRIEVAL_URL = "http://127.0.0.1:8001"

SEARCH_TASK_INSTRUCTIONS = """You are an expert assistant who solves tasks using a Wikipedia search tool.

Here are instructions for how to solve a problem:
1. Think step by step before calling the tool and after you receive the result of the tool call. Decide what queries to call the tool with.
2. Call the tool with the queries you have decided on.
3. Think step by step again after you receive the result of the tool call. If you have the information you need, you can stop here.
4. Otherwise, come up with new queries that combine information from the previous results.
5. Include your final answer after the "Answer:" prefix. The answer should be one to five words.

Here is an example of solving a real question:
"What was the population of San Francisco in the year that New York City first recorded a census population above seven million?"

1. Think step by step: In order to answer this question, I first need to find the census year in which New York City's population first exceeded seven million. I will search for New York City's historical census populations.
2. Calling search tool: <tool_call>{"name": "search", "arguments": {"query_list": ["New York City historical population census"]}}</tool_call> (Output omitted for brevity)
3. Think step by step again: The 1930 census recorded 6,930,446 people and the 1940 census recorded 7,454,995, so 1940 was the first census above seven million. Now I need the population of San Francisco in 1940. I will search for it.
<tool_call>{"name": "search", "arguments": {"query_list": ["San Francisco population 1940 census"]}}</tool_call> (Output omitted for brevity)
4. Answer: 634,536
"""


# "Fast thinking" variant (Search-R1++): the instructions no longer ask the model to
# think step by step, and the example shows only tool calls and the answer. The tool
# call and answer formats are unchanged.
SEARCH_TASK_INSTRUCTIONS_FAST = """You are an expert assistant who solves tasks using a Wikipedia search tool.

Here are instructions for how to solve a problem:
1. Decide what queries to call the tool with.
2. Call the tool with the queries you have decided on.
3. If you have the information you need after you receive the result of the tool call, you can stop here.
4. Otherwise, come up with new queries that combine information from the previous results.
5. Include your final answer after the "Answer:" prefix. The answer should be one to five words.

Here is an example of solving a real question:
"What was the population of San Francisco in the year that New York City first recorded a census population above seven million?"

1. Calling search tool: <tool_call>{"name": "search", "arguments": {"query_list": ["New York City historical population census"]}}</tool_call> (Output omitted for brevity)
2. Calling search tool: <tool_call>{"name": "search", "arguments": {"query_list": ["San Francisco population 1940 census"]}}</tool_call> (Output omitted for brevity)
3. Answer: 634,536
"""

PromptName = Literal["original", "fast"]


def get_task_instructions(prompt_name: PromptName) -> str:
    if prompt_name == "original":
        return SEARCH_TASK_INSTRUCTIONS
    if prompt_name == "fast":
        return SEARCH_TASK_INSTRUCTIONS_FAST
    raise ValueError(f"Unknown prompt_name: {prompt_name}")


class SearchR1Datum(TypedDict):
    id: str
    question: str
    answer: list[str]
    data_source: str


def process_single_row(row_series: pd.Series) -> SearchR1Datum:
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row_series: DataFrame row containing the original data

    Returns:
        SearchR1Datum: Processed row data in the required format
    """
    import numpy as np

    row = row_series.to_dict()
    question: str = row.get("question", "")

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # NOTE(tianyi)
    # I hate datasets with mixed types but it is what it is.
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    assert isinstance(ground_truth, list)
    for item in ground_truth:
        assert isinstance(item, str)
    ground_truth = cast(list[str], ground_truth)
    return {
        "id": str(row.get("id", "")),
        "question": question,
        "answer": ground_truth,
        "data_source": row["data_source"],
    }


def download_search_r1_dataset(split: Literal["train", "test"]) -> list[SearchR1Datum]:
    hf_repo_id: str = "PeterJinGo/nq_hotpotqa_train"
    parquet_filename: str = f"{split}.parquet"
    download_dir = DATA_DIR / "nq_hotpotqa_train"
    local_parquet_filepath = download_dir / parquet_filename
    if not local_parquet_filepath.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=parquet_filename,
            repo_type="dataset",
            local_dir=download_dir,
        )

    df_raw = pd.read_parquet(local_parquet_filepath)

    return df_raw.apply(process_single_row, axis=1).tolist()


def search_initial_messages(
    datum: SearchR1Datum,
    renderer: Renderer,
    prompt_name: PromptName,
    limit_search: bool,
) -> list[Message]:
    """Build initial messages with tool schemas and task question."""
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[search_tool_spec(limit_search)],
        system_prompt=get_task_instructions(prompt_name),
    )
    return prefix + [{"role": "user", "content": datum["question"]}]


class SearchEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder that creates search environments with a shared BM25Tool."""

    def __init__(
        self,
        datum: SearchR1Datum,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        search_tool: BM25Tool,
        format_coef: float = 0.1,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
        prompt_name: PromptName = "fast",
        answer_score: Literal["f1", "em"] = "f1",
        no_search_coef: float = 0.1,
        limit_search: bool = False,
        max_turns_unanswered_reward: float | None = None,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.search_tool = search_tool
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.prompt_name = prompt_name
        self.answer_score = answer_score
        self.no_search_coef = no_search_coef
        self.limit_search = limit_search
        self.max_turns_unanswered_reward = max_turns_unanswered_reward

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        # Tool, initial_messages, reward_fn are all stateless - can share
        initial_messages = search_initial_messages(
            self.datum, renderer, self.prompt_name, self.limit_search
        )
        reward_fn = TextAnswerReward(
            gold_answers=self.datum["answer"],
            format_coef=self.format_coef,
            answer_score=self.answer_score,
            no_search_coef=self.no_search_coef,
        )

        return [
            build_search_env(
                renderer=renderer,
                search_tool=self.search_tool,
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                limit_search=self.limit_search,
                example_id=self.datum["id"],
                max_turns_unanswered_reward=self.max_turns_unanswered_reward,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        if self.max_turns_unanswered_reward is None:
            return await super().compute_group_rewards(trajectory_group, env_group)
        # A 0/1 value for every trajectory, so env/*/max_turns_unanswered is a fraction
        # of all trajectories (transition metrics only average over steps that set them).
        return [
            (
                0.0,
                {
                    "max_turns_unanswered": float(
                        bool(traj.transitions)
                        and MAX_TURNS_UNANSWERED_LOG in traj.transitions[-1].logs
                    )
                },
            )
            for traj in trajectory_group
        ]

    def logging_tags(self) -> list[str]:
        return [self.datum.get("data_source", "unknown")]


class SearchRLDataset(RLDataset):
    """Dataset that processes search EnvGroupBuilders once per epoch."""

    def __init__(
        self,
        env_group_builders: list[SearchEnvGroupBuilder],
        batch_size: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return len(self.env_group_builders) // self.batch_size


@chz.chz
class SearchR1DatasetBuilder(RLDatasetBuilder):
    """Build an RL dataset over SearchR1 tasks with BM25Tool."""

    model_name_for_tokenizer: str
    # BM25 retrieval server
    retrieval_url: str = DEFAULT_RETRIEVAL_URL
    retrieval_config: RetrievalConfig = RetrievalConfig()
    limit_search: bool = False
    # Dataset params
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 5
    format_coef: float = 0.1
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    # Prompt and reward. Defaults are the training configuration (fast prompt, F1+);
    # prompt_name="original", answer_score="em", no_search_coef=0.0 is the recipe's.
    prompt_name: PromptName = "fast"
    answer_score: Literal["f1", "em"] = "f1"
    no_search_coef: float = 0.1
    # Training only (offline evaluation keeps the shared scorer): terminal reward for an
    # episode that reaches max_turns without a valid answer. None keeps the scorer's value.
    max_turns_unanswered_reward: float | None = None
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Create shared BM25Tool
        search_tool = BM25Tool(
            retrieval_url=self.retrieval_url,
            retrieval_config=self.retrieval_config,
        )

        data = download_search_r1_dataset("train")
        rng = random.Random(self.seed)
        rng.shuffle(data)

        env_builders = [
            SearchEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                search_tool=search_tool,
                format_coef=self.format_coef,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                prompt_name=self.prompt_name,
                answer_score=self.answer_score,
                no_search_coef=self.no_search_coef,
                limit_search=self.limit_search,
                max_turns_unanswered_reward=self.max_turns_unanswered_reward,
            )
            for datum in data
        ]
        dataset = SearchRLDataset(
            env_group_builders=env_builders,
            batch_size=self.batch_size,
        )
        return dataset, None
