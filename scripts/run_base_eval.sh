#!/usr/bin/env bash
# 本地基座评测（计划第 2 步）：5 个配置，同一批题、同一检索、同一长度限制。
# 需要先起检索服务（./retrieval/start.sh）和 oMLX，并在 oMLX 里能加载两个模型。
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

Q35=(omlx_model=Qwen3.5-4B-bf16 base_model=Qwen/Qwen3.5-4B)
Q3I=(omlx_model=Qwen3-4B-Instruct-2507-bf16 base_model=Qwen/Qwen3-4B-Instruct-2507 renderer_name=qwen3_instruct)

run() {
  local name="$1"; shift
  echo "===== $name ====="
  uv run python -m search_r1.offline_eval backend=omlx \
    eval_file=eval_sets/dev.jsonl limit_per_source="$LIMIT" \
    run_name="base-$name-$TAG" "$@"
}

config() {
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
