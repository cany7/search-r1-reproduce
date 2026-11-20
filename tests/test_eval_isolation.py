"""Fresh-process checks: evaluation cannot import cookbook or access the network."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GUARD = """
import importlib.abc
import socket
import sys
class NoCookbook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tinker_cookbook" or fullname.startswith("tinker_cookbook."):
            raise AssertionError("Forbidden cookbook import: " + fullname)
sys.meta_path.insert(0, NoCookbook())
def no_network(*args, **kwargs):
    raise AssertionError("Network access is forbidden in offline migration checks")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
"""


def isolated(code, *, expected_returncode=0, **env):
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        **env,
    }
    # chz changes CLI errors into raised exceptions inside pytest; exercise the
    # normal CLI contract in the subprocess instead of inheriting that marker.
    environment.pop("PYTEST_VERSION", None)
    result = subprocess.run(
        [sys.executable, "-c", GUARD + code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected_returncode, result.stdout + result.stderr
    if expected_returncode:
        assert not result.stderr, result.stderr
    return result.stdout


def test_eval_and_helpers_import_without_cookbook():
    isolated(
        """
import importlib
import pkgutil
import runpy
from pathlib import Path
import search_r1.env
import search_r1.backends
for package in [search_r1.env, search_r1.backends]:
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module.name)
import search_r1.offline_eval
import search_r1.eval_config
for script in ["check_render.py", "diagnose_run.py", "make_eval_sets.py"]:
    runpy.run_path(str(Path("scripts") / script), run_name="import_only")
from search_r1.env.search_env import DATA_DIR
assert DATA_DIR == Path.cwd() / "data", DATA_DIR
assert not any(n.startswith("tinker_cookbook") for n in sys.modules)
""",
        SEARCH_R1_DATA_DIR=str(ROOT / "data"),
    )
    # A separate process with the variable absent checks the relocated default.
    isolated("""
import os
from pathlib import Path
os.environ.pop("SEARCH_R1_DATA_DIR", None)
from search_r1.env.search_env import DATA_DIR
assert DATA_DIR == Path.cwd() / "data", DATA_DIR
""")


def test_eval_cli_help_does_not_load_models_or_cookbook():
    output = isolated(
        """
import runpy
sys.argv = ["search_r1.offline_eval", "--help"]
runpy.run_module("search_r1.offline_eval", run_name="__main__")
""",
        expected_returncode=1,
    )  # The pinned chz entrypoint exits 1 even for --help.
    assert "training_config_file" in output and "tinker_state_url" in output


def test_overridden_data_directory_is_actually_read(tmp_path):
    dataset = tmp_path / "nq_hotpotqa_train"
    dataset.mkdir()
    pd.DataFrame(
        [{"id": 8, "question": "Who?", "golden_answers": ["yes"], "data_source": "nq"}]
    ).to_parquet(dataset / "test.parquet")
    isolated(
        """
import os
from pathlib import Path
from search_r1.env.search_env import DATA_DIR, download_search_r1_dataset
assert DATA_DIR == Path(os.environ["SEARCH_R1_DATA_DIR"])
assert download_search_r1_dataset("test") == [
    {"id": "8", "question": "Who?", "answer": ["yes"], "data_source": "nq"}
]
""",
        SEARCH_R1_DATA_DIR=str(tmp_path),
    )


def test_local_regressions_pass_with_cookbook_imports_blocked():
    output = isolated("""
import pytest
raise SystemExit(pytest.main([
    "-q", "tests/test_component_migration.py", "tests/test_eval_backends.py", "tests/test_reward.py",
    "tests/test_tokenizer_loading.py",
]))
""")
    assert "passed" in output
