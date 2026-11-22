"""Training CLI help must work without network access or model downloads."""

import pytest
from test_eval_isolation import isolated


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
