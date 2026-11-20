#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env_common.sh" || exit 1
cd "$PROJ"
uv sync --locked
.venv/bin/python -c "import tinker, wandb; print('tinker', tinker.__version__)"
cd retrieval && uv sync --locked
.venv/bin/python -c "import pyserini, faiss; print('pyserini ok')"
echo PROJ_DONE
