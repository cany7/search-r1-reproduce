#!/bin/bash
# Install SkyRL (pinned commit + search-r1 local patch) with the tinker and fsdp extras.
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env_common.sh" || exit 1
[ -d "$SKYRL" ] || git clone -q https://github.com/NovaSky-AI/SkyRL.git "$SKYRL"
cd "$SKYRL"
git checkout -q 79d4c8946bba23c2e0617e96ac1095c596093d83
# Trim padding and add length-based checkpointing (see the matching .patch file).
python3 "$CLOUD_DIR/apply_patch.py" "$SKYRL"
git diff --stat
date
[ -d .venv ] || uv venv --python "$UV_PYTHON" .venv
uv sync --extra tinker --extra fsdp
date
.venv/bin/python -c "import torch, vllm, tinker; print(torch.__version__, torch.version.cuda, vllm.__version__, tinker.__version__, torch.cuda.is_available())"
echo INSTALL_DONE
