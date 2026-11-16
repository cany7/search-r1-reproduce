#!/usr/bin/env bash
# 本地基座评测（计划第 2 步）：5 个配置，同一批题、同一检索、同一长度限制。
#
# 脚本自己管检索服务：没在运行就启动它并等它就绪，跑完（或中途退出）时把它停掉。
# 如果检索服务本来就在运行，脚本直接用它，结束时不动它。
# oMLX 需要事先开着。配置是一个接一个跑的，同一时间只用一个模型；切换模型前会让
# oMLX 卸载另一个，免得两个模型（各约 9 GB）和检索服务（约 13.5 GB）一起占内存。
#
# 用法：
#   ./scripts/run_base_eval.sh            # 跑全部 5 个配置
#   ./scripts/run_base_eval.sh q35-fast   # 只跑一个配置
#   LIMIT=50 ./scripts/run_base_eval.sh   # 每个来源只取前 50 题（默认 100，即整个开发集）
#   TAG=run2 ./scripts/run_base_eval.sh   # 输出目录名后缀，重复跑时用来区分
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

# ---------- 检索服务 ----------
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
  # 语料缓存已生成时约 1 分钟；第一次要转换语料，会久一些。最多等 20 分钟。
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
# 卸载指定模型。没加载时 oMLX 返回 400，忽略即可。
unload() {
  curl -s -o /dev/null -m 120 -X POST -H "Authorization: Bearer $OMLX_API_KEY" \
    "$OMLX_BASE_URL/models/$1/unload" || true
}

# ---------- 评测 ----------
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
