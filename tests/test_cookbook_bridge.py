"""Temporary integration check; remove with the old training loop in stages 4–5.

This is deliberately separate from the cookbook-free evaluation regression suite.
It executes the old single-episode runner, never a training loop or SDK service.
"""

import asyncio

import pytest
from test_component_migration import REFERENCE, local_rollout
from tinker_cookbook.rl.rollouts import do_single_rollout

from search_r1.training.cookbook_bridge import _Environment


@pytest.mark.parametrize("case", ["initial_overflow", "search_answer", "length", "content_error"])
def test_old_runner_accepts_local_environment_through_bridge(case):
    actual = asyncio.run(local_rollout(case, run_rollout=do_single_rollout, wrap_env=_Environment))
    assert actual == REFERENCE["rollouts"][case]
