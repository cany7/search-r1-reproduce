#!/usr/bin/env bash
# 启动 BM25 检索服务。数据目录和端口可用环境变量覆盖：
#   DATA_DIR（默认 <项目>/data）、RETRIEVAL_PORT（默认 8001）、RETRIEVAL_HOST（默认 127.0.0.1）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"

# pyserini 需要 JDK 21。优先级：RETRIEVAL_JAVA_HOME > Homebrew 的 openjdk@21 > 已有的 JAVA_HOME。
# 不直接沿用已有的 JAVA_HOME，是因为它可能指向别的 JDK（比如 Android Studio 自带的，缺
# jdk.incubator.vector 模块，pyserini 起不来）。
BREW_JDK="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if [ -n "${RETRIEVAL_JAVA_HOME:-}" ]; then
  export JAVA_HOME="$RETRIEVAL_JAVA_HOME"
elif [ -d "$BREW_JDK" ]; then
  export JAVA_HOME="$BREW_JDK"
fi
echo "JAVA_HOME=${JAVA_HOME:-<unset>}"

exec uv run --project "$ROOT/retrieval" python "$ROOT/retrieval/retrieval_server.py" \
  --index_path "$DATA/bm25-index/bm25" \
  --corpus_path "$DATA/corpus/wiki-18.jsonl" \
  --topk 3 --retriever_name bm25 \
  --host "${RETRIEVAL_HOST:-127.0.0.1}" --port "${RETRIEVAL_PORT:-8001}"
