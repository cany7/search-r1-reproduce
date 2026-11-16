#!/usr/bin/env bash
# Start the BM25 retrieval server. Environment variables override these defaults.
#   DATA_DIR=<project>/data, RETRIEVAL_PORT=8001, RETRIEVAL_HOST=127.0.0.1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"

# Pyserini requires JDK 21. Prefer RETRIEVAL_JAVA_HOME, then Homebrew's openjdk@21,
# then the existing JAVA_HOME. Some JDKs, such as the one bundled with Android Studio,
# lack the jdk.incubator.vector module that Pyserini needs at startup.
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
