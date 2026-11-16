#!/usr/bin/env bash
# Local base-model evaluation (step 2 of the reproduction plan). All five
# configurations use the same questions, retrieval setup, and length limit.
#
# The script reuses a running retrieval server. If it starts the server itself,
# it waits for readiness and stops the server on exit, including early exits.
# Start oMLX before running this script. Configurations run one at a time.
# Each configuration unloads the other model to avoid holding both models
# (about 9 GB each) and the retrieval server (about 13.5 GB) in memory at once.
#
# Usage
#   ./scripts/run_base_eval.sh            # Run all five configurations
#   ./scripts/run_base_eval.sh q35-fast   # Run one configuration
#   LIMIT=50 ./scripts/run_base_eval.sh   # Use the first 50 questions per source (default 100)
#   TAG=run2 ./scripts/run_base_eval.sh   # Set an output directory suffix for a repeat run
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${LIMIT:-100}"
TAG="${TAG:-n$LIMIT}"

Q35_ID=Qwen3.5-4B-bf16
Q3I_ID=Qwen3-4B-Instruct-2507-bf16
Q35=(omlx_model=$Q35_ID base_model=Qwen/Qwen3.5-4B)
Q3I=(omlx_model=$Q3I_ID base_model=Qwen/Qwen3-4B-Instruct-2507 renderer_name=qwen3_instruct)

set -a; . ./.env; set +a
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8001}"

# ---------- Retrieval server ----------
retrieval_ready() {
  curl -s -o /dev/null -m 5 -f -X POST "$RETRIEVAL_URL/retrieve" \
    -H 'content-type: application/json' \
    -d '{"queries":["test"],"topk":1,"return_scores":true}'
}

STARTED_RETRIEVAL=0
stop_retrieval() {
  if [ "$STARTED_RETRIEVAL" = 1 ]; then
    echo "stopping retrieval server"
    ./retrieval/stop.sh || true
  fi
}
trap stop_retrieval EXIT

if retrieval_ready; then
  echo "retrieval server already running at $RETRIEVAL_URL"
else
  echo "starting retrieval server (log: runs/retrieval.log)"
  mkdir -p runs
  ./retrieval/start.sh > runs/retrieval.log 2>&1 &
  STARTED_RETRIEVAL=1
  # Startup takes about a minute with a cached corpus and longer on the first run.
  # Wait at most 20 minutes.
  for _ in $(seq 240); do
    if retrieval_ready; then break; fi
    if ! pgrep -f "retrieval/retrieval_server.py" > /dev/null; then
      echo "retrieval server exited during startup; see runs/retrieval.log" >&2
      exit 1
    fi
    sleep 5
  done
  retrieval_ready || { echo "retrieval server not ready after 20 minutes" >&2; exit 1; }
  echo "retrieval server ready"
fi

# ---------- oMLX ----------
# Unload the specified model. Ignore the 400 response if the model is already unloaded.
unload() {
  curl -s -o /dev/null -m 120 -X POST -H "Authorization: Bearer $OMLX_API_KEY" \
    "$OMLX_BASE_URL/models/$1/unload" || true
}

# ---------- Evaluation ----------
run() {
  local name="$1"; shift
  echo "===== $name ====="
  uv run python -m search_r1.offline_eval backend=omlx \
    eval_file=eval_sets/dev.jsonl limit_per_source="$LIMIT" \
    run_name="base-$name-$TAG" "$@"
}

config() {
  case "$1" in
    q35-*) unload "$Q3I_ID" ;;
    q3i-*) unload "$Q35_ID" ;;
  esac
  case "$1" in
    q35-original)       run "$1" "${Q35[@]}" renderer_name=qwen3_5_disable_thinking prompt_name=original ;;
    q35-fast)           run "$1" "${Q35[@]}" renderer_name=qwen3_5_disable_thinking prompt_name=fast ;;
    q35-think-fast)     run "$1" "${Q35[@]}" renderer_name=qwen3_5 prompt_name=fast ;;
    q3i-original)       run "$1" "${Q3I[@]}" prompt_name=original ;;
    q3i-fast)           run "$1" "${Q3I[@]}" prompt_name=fast ;;
    *) echo "unknown config: $1" >&2; exit 1 ;;
  esac
}

if [ "$#" -gt 0 ]; then
  for name in "$@"; do config "$name"; done
else
  for name in q35-original q35-fast q35-think-fast q3i-original q3i-fast; do config "$name"; done
fi
