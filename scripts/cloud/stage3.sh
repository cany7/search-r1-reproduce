#!/usr/bin/env bash
# Dynamic compatibility entry. Supply the run and targets as arguments or Pod variables.
set -eo pipefail
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SAMPLING_MODE=dynamic
case "${1:-}" in
  formal)
    shift
    if [ "$#" -eq 0 ]; then set -- "${RUN_NAME:?set RUN_NAME or pass the run name}"; fi
    exec bash "$dir/train_with_dev_eval.sh" "$@"
    ;;
  smoke)
    shift
    if [ "$#" -eq 0 ]; then
      set -- "${RUN_NAME:?set RUN_NAME}" "${BATCH_SIZE:?set BATCH_SIZE}" "${TOTAL_STEPS:?set TOTAL_STEPS}"
    fi
    export RESTART_BEFORE_TRAIN=1
    exec bash "$dir/run_train_test.sh" "$@" behavior_if_log_dir_exists=resume
    ;;
  *) echo "usage: $0 formal RUN [key=value ...] | smoke NAME BATCH_SIZE MAX_STEPS [key=value ...]" >&2; exit 2 ;;
esac
