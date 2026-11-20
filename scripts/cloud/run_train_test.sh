#!/usr/bin/env bash
# usage: run_train_test.sh NAME BATCH_SIZE MAX_STEPS [key=value ...]
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then echo "usage: $0 NAME BATCH_SIZE MAX_STEPS [key=value ...]"; exit 0; fi
[ "$#" -ge 3 ] || { echo "usage: $0 NAME BATCH_SIZE MAX_STEPS [key=value ...]" >&2; exit 2; }
source "$(dirname -- "${BASH_SOURCE[0]}")/env_common.sh"
cd "$PROJ"
exec python3 -m scripts.cloud.driver smoke "$@"
