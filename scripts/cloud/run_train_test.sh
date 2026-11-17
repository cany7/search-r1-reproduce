#!/bin/bash
# Run a short train.py test. usage: run_train_test.sh <name> <batch_size> <max_steps> [extra train.py args...]
NAME=$1; BS=$2; STEPS=$3; shift 3
source /workspace/bin/env_common.sh
cd $PROJ
echo "START $NAME $(date +%s)" >> /workspace/logs/train_tests.times
uv run --frozen python -m search_r1.train batch_size=$BS group_size=8 max_steps=$STEPS \
  log_path=/workspace/runs/tests/$NAME wandb_name=test-$NAME behavior_if_log_dir_exists=delete "$@" \
  > /workspace/logs/train_$NAME.log 2>&1
RC=$?
echo "END $NAME $(date +%s) rc=$RC" >> /workspace/logs/train_tests.times
