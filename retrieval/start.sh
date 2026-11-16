#!/usr/bin/env bash
# 启动 BM25 检索服务。数据目录和端口可用环境变量覆盖：
#   DATA_DIR（默认 <项目>/data）、RETRIEVAL_PORT（默认 8001）、RETRIEVAL_HOST（默认 127.0.0.1）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"

# pyserini 需要 JDK 21。没设 JAVA_HOME 时，在 macOS 上用 Homebrew 的 openjdk@21。
BREW_JDK="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if [ -z "${JAVA_HOME:-}" ] && [ -d "$BREW_JDK" ]; then
  export JAVA_HOME="$BREW_JDK"
fi

exec uv run --project "$ROOT/retrieval" python "$ROOT/retrieval/retrieval_server.py" \
  --index_path "$DATA/bm25-index/bm25" \
  --corpus_path "$DATA/corpus/wiki-18.jsonl" \
  --topk 3 --retriever_name bm25 \
  --host "${RETRIEVAL_HOST:-127.0.0.1}" --port "${RETRIEVAL_PORT:-8001}"
