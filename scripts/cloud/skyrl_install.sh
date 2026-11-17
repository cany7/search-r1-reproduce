#!/bin/bash
# Install SkyRL (pinned commit + search-r1 local patch) with the tinker and fsdp extras.
# Expects this directory to be copied to /workspace/bin on the pod.
set -e
source /workspace/.secrets/env.sh
export PATH=/home/ray/.local/bin:$PATH
SKYRL=/workspace/search-r1/SkyRL
[ -d $SKYRL ] || git clone -q https://github.com/NovaSky-AI/SkyRL.git $SKYRL
cd $SKYRL
git checkout -q 79d4c8946bba23c2e0617e96ac1095c596093d83
# Trim shared left padding per policy microbatch (see skyrl-79d4c89-trim-microbatch-padding.patch).
python3 /workspace/bin/apply_patch.py $SKYRL
git diff --stat
date
[ -d .venv ] || uv venv --python /home/ray/anaconda3/bin/python3.12 .venv
uv sync --extra tinker --extra fsdp
date
.venv/bin/python -c "import torch, vllm, tinker; print(torch.__version__, torch.version.cuda, vllm.__version__, tinker.__version__, torch.cuda.is_available())"
echo INSTALL_DONE
