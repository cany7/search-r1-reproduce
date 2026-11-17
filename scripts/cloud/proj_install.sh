set -e
source /workspace/.secrets/env.sh
export PATH=/home/ray/.local/bin:$PATH
export UV_PYTHON=/home/ray/anaconda3/bin/python3.12
cd /workspace/search-r1/project
uv sync --locked
.venv/bin/python -c "import tinker, tinker_cookbook, wandb; print('tinker', tinker.__version__)"
cd retrieval && uv sync --locked
.venv/bin/python -c "import pyserini, faiss; print('pyserini ok')"
echo PROJ_DONE
