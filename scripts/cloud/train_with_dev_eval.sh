#!/usr/bin/env bash
# usage: train_with_dev_eval.sh RUN [key=value ...]
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then echo "usage: $0 RUN [key=value ...]"; exit 0; fi
[ "$#" -ge 1 ] || { echo "usage: $0 RUN [key=value ...]" >&2; exit 2; }
source "$(dirname -- "${BASH_SOURCE[0]}")/env_common.sh"
cd "$PROJ"
exec python3 -m scripts.cloud.driver train "$@"
