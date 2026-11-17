#!/bin/bash
# Restart the SkyRL server with optional EXTRA_CFG and wait until the API is up.
# usage: EXTRA_CFG='"trainer.micro_train_batch_size_per_gpu": 2' restart_skyrl.sh <logname>
NAME=${1:-skyrl_server}
HEALTH_TIMEOUT_SECONDS=${HEALTH_TIMEOUT_SECONDS:-120}
GC_SHORT_SEQ_THRESHOLD=${GC_SHORT_SEQ_THRESHOLD:-8000}
if ! [[ "$GC_SHORT_SEQ_THRESHOLD" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "GC_SHORT_SEQ_THRESHOLD must be a nonnegative integer" >&2
  exit 1
fi
tmux kill-session -t skyrl 2>/dev/null
pkill -f "skyrl.tinker" 2>/dev/null; sleep 3
source /workspace/bin/env_common.sh
(cd $SKYRL && uv run --frozen --extra tinker --extra fsdp ray stop --force >/dev/null 2>&1)
pkill -9 -f "skyrl.tinker|raylet|gcs_server|vllm" 2>/dev/null; sleep 5
rm -rf "$RAY_TMPDIR"/ray/session_* /tmp/ray 2>/dev/null  # stale Ray sessions each hold a venv copy
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / /workspace | tail -2
tmux new -d -s skyrl "GC_SHORT_SEQ_THRESHOLD=$GC_SHORT_SEQ_THRESHOLD EXTRA_CFG='${EXTRA_CFG:-}' bash /workspace/bin/start_skyrl.sh 2>&1 | tee /workspace/logs/$NAME.log" || exit 1
deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -sf --max-time 2 "${TINKER_BASE_URL}/api/v1/healthz" >/dev/null 2>&1; then
    echo "server up: $(date)"
    exit 0
  fi
  sleep 2
done
echo "ERROR: API health check timed out after ${HEALTH_TIMEOUT_SECONDS}s; see /workspace/logs/$NAME.log" >&2
exit 1
