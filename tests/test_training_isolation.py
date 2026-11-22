"""The math, loop and public CLI help must work with cookbook and network blocked."""

import pytest
from test_eval_isolation import isolated


def test_core_imports_and_fake_sdk_regressions_without_cookbook():
    result = isolated("""
import importlib
for name in ["config", "data", "loop", "metrics", "records", "sampling", "update", "usage", "utils"]:
    importlib.import_module("search_r1.training." + name)
import search_r1.train
import search_r1.train_dynamic_sampling
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_training_data.py", "tests/test_training_loop.py",
    "tests/test_training_runtime.py", "tests/test_training_checkpoints.py", "tests/test_dynamic_sampling.py",
    "tests/test_training_logging.py"]))
""")
    assert "passed" in result


@pytest.mark.parametrize("module", ["search_r1.train", "search_r1.train_dynamic_sampling"])
def test_training_cli_help_is_offline(module):
    output = isolated(
        f"""
import runpy
sys.argv = [{module!r}, "--help"]
runpy.run_module({module!r}, run_name="__main__")
""",
        expected_returncode=1,
    )
    for key in (
        "max_sampling_rounds",
        "stream_minibatch",
        "parent_run",
        "parent_step",
        "load_checkpoint_path",
    ):
        assert key in output
