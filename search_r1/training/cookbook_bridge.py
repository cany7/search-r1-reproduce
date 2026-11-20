"""Temporary training-only boundary while stages 4–5 migrate the old train loop.

The evaluation path must never import this module. Remove it once training uses
the local rollout and checkpoint implementation. Only the initial-overflow
sentinel requires conversion: the old runner checks its concrete class.
"""

import chz
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.rl.types import InitialObservationOverflow as CookbookOverflow

from search_r1.env.renderers import DEFAULT_RENDERER_NAME, Qwen3_5DisableThinkingUnicodeRenderer
from search_r1.env.search_env import SearchR1DatasetBuilder as LocalDatasetBuilder
from search_r1.env.types import InitialObservationOverflow

register_renderer(DEFAULT_RENDERER_NAME, Qwen3_5DisableThinkingUnicodeRenderer)


class _Environment:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    async def initial_observation(self):
        result = await self.env.initial_observation()
        if isinstance(result, InitialObservationOverflow):
            return CookbookOverflow(result.reward, result.metrics, result.logs)
        return result


class _Group:
    def __init__(self, builder):
        self.builder = builder

    def __getattr__(self, name):
        return getattr(self.builder, name)

    async def make_envs(self):
        return [_Environment(env) for env in await self.builder.make_envs()]


@chz.chz
class SearchR1DatasetBuilder(LocalDatasetBuilder):
    async def __call__(self):
        dataset, test_dataset = await super().__call__()
        dataset.env_group_builders = [_Group(b) for b in dataset.env_group_builders]
        return dataset, test_dataset
