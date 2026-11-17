#!/bin/bash
# Restart the SkyRL server with optional EXTRA_CFG and wait until the API is up.
# usage: EXTRA_CFG='"trainer.micro_train_batch_size_per_gpu": 2' restart_skyrl.sh <logname>
NAME=${1:-skyrl_server}
tmux kill-session -t skyrl 2>/dev/null
pkill -f "skyrl.tinker" 2>/dev/null; sleep 3
source /workspace/bin/env_common.sh
(cd $SKYRL && uv run --frozen --extra tinker --extra fsdp ray stop --force >/dev/null 2>&1)
pkill -9 -f "skyrl.tinker|raylet|gcs_server|vllm" 2>/dev/null; sleep 5
rm -rf "$RAY_TMPDIR"/ray/session_* /tmp/ray 2>/dev/null  # stale Ray sessions each hold a venv copy
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / /workspace | tail -2
tmux new -d -s skyrl "EXTRA_CFG='${EXTRA_CFG:-}' bash /workspace/bin/start_skyrl.sh 2>&1 | tee /workspace/logs/$NAME.log"
for i in $(seq 1 60); do curl -sf localhost:8000/api/v1/healthz >/dev/null 2>&1 && break; grep -q "Uvicorn running" /workspace/logs/$NAME.log 2>/dev/null && break; sleep 2; done
grep -a "backend-config" /workspace/logs/$NAME.log | head -1
echo "server up: $(date)"
