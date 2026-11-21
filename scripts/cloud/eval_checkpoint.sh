#!/usr/bin/env bash
# usage: eval_checkpoint.sh RUN STEP|base dev|test [training_config_file=FILE] [key=value ...]
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then echo "usage: $0 RUN STEP|base dev|test [training_config_file=FILE] [key=value ...]"; exit 0; fi
[ "$#" -ge 3 ] || { echo "usage: $0 RUN STEP|base dev|test [training_config_file=FILE] [key=value ...]" >&2; exit 2; }
source "$(dirname -- "${BASH_SOURCE[0]}")/env_common.sh"
cd "$PROJ"
exec python3 -m scripts.cloud.workflows.driver eval "$@"
