#!/usr/bin/env bash
# Call from the project checkout after the stage-two driver and new-rule eval finish.
set -eo pipefail
source /workspace/bin/env_common.sh
set -u
unset WANDB_RUN_ID WANDB_RESUME
cd "$PROJ"
parent=/workspace/runs/train/search-r1-q35-64x8-s2
ops="$PROJ/scripts/cloud/stage3_ops.py"
# --close keeps tmux/Ray/background upload processes from inheriting the lock.
if [ "${1:-}" != --locked ]; then
  exec flock -n --close /workspace/runs/train/.stage3-driver.lock bash "$PROJ/scripts/cloud/stage3.sh" --locked "$@"
fi
shift

case "${1:-}" in
  smoke)
    name=search-r1-q35-64x8-s3-smoke
    out=/workspace/runs/train/$name
    python3 "$ops" disk-plan "$out" --targets 102 103 --save-every 0
    for t in 102 103; do
      if python3 "$ops" verify "$out" --target "$t" >/dev/null 2>&1; then continue; fi
      /workspace/bin/restart_skyrl.sh "${name}_server_$t"
      rc=0
      uv run --frozen python -m search_r1.train_dapo --parent "$parent" --output "$out" \
        --target "$t" --save-every 0 >"/workspace/logs/${name}_train_$t.log" 2>&1 || rc=$?
      # SkyRL may return 404 while closing a session after a successful final save.
      python3 "$ops" verify "$out" --target "$t" || { echo "Smoke failed (trainer rc=$rc)"; exit 1; }
    done
    ;;
  formal)
    name=search-r1-q35-64x8-s3
    python3 "$ops" verify /workspace/runs/train/search-r1-q35-64x8-s3-smoke --target 103
    export STAGE3_PARENT="$parent" PRUNE_STATES=0 UPLOAD=1
    export BATCH_SIZE=64 GROUP_SIZE=8 SAVE_EVERY=${SAVE_EVERY:-5}
    python3 "$ops" disk-plan "/workspace/runs/train/$name" --targets 112 124 --save-every "$SAVE_EVERY"
    exec bash "$PROJ/scripts/cloud/train_with_dev_eval.sh" "$name"
    ;;
  *) echo "usage: $0 smoke|formal" >&2; exit 2 ;;
esac
