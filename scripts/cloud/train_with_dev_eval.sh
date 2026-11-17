#!/bin/bash
# Formal training with a fixed dev-set eval every EVAL_EVERY steps (approach 2).
#
# Self-hosted SkyRL cannot evaluate a sampler checkpoint (it samples the loaded model's
# current weights, or fails after a restart), so training runs in segments:
#   train to step T (cookbook saves a `final` checkpoint) -> restart the server ->
#   evaluate that checkpoint's full state with offline_eval.py tinker_state_url=... ->
#   restart the server -> resume training to step T + EVAL_EVERY -> ...
# After each eval, unneeded full states are pruned and the eval step's checkpoint is uploaded
# to W&B Artifacts in a background process (upload_checkpoint.py); training does not wait for
# it. At the end the script waits for all uploads, then says the pod can be terminated.
#
# The script is idempotent: rerunning it skips segments whose checkpoint already exists
# and evals whose summary.json already exists, so it also resumes after an interruption
# and extends a run (e.g. rerun with TOTAL_STEPS=200 after a 100-step run).
#
# usage: train_with_dev_eval.sh <run_name> [extra train.py args...]
# env:   TOTAL_STEPS (100), EVAL_EVERY (25), BATCH_SIZE (64), GROUP_SIZE (8),
#        SAVE_EVERY (5; a crash loses at most this many steps),
#        EVAL_CONCURRENCY (32),
#        PRUNE_STATES (1: after each eval, delete full states other than the eval steps' and
#                      the latest, and duplicate saves of one step; sampler weights are kept;
#                      see ckpt_prune.py),
#        UPLOAD (1: upload each eval step's full state, the segment's sampler weights and the
#                eval summary to W&B; see upload_checkpoint.py)
# Disk: about 240 GB at the peak of a 200-step run; use a 300 GB volume.
#
# Needs: the BM25 retrieval server running, SkyRL with the local patch, /workspace/bin scripts.
# Run it in its own tmux session, not one named "skyrl" (restart_skyrl.sh kills that session):
#   tmux new -d -s formal "/workspace/bin/train_with_dev_eval.sh search-r1-64x8"
# The base model is not evaluated here; run the cloud base eval (dev and test) separately first.
# Each dev eval inherits model and rollout settings from the saved training config.json.
# train.py exits with rc=1 on SkyRL even after a clean run (404 on session finish), so
# success is judged from checkpoints.jsonl and the eval summary.json, not exit codes.
set -uo pipefail

NAME=${1:?usage: train_with_dev_eval.sh <run_name> [extra train.py args...]}
shift
TOTAL_STEPS=${TOTAL_STEPS:-100}
EVAL_EVERY=${EVAL_EVERY:-25}
BATCH_SIZE=${BATCH_SIZE:-64}
GROUP_SIZE=${GROUP_SIZE:-8}
SAVE_EVERY=${SAVE_EVERY:-5}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-32}
PRUNE_STATES=${PRUNE_STATES:-1}
UPLOAD=${UPLOAD:-1}

source /workspace/bin/env_common.sh
LOG_PATH=/workspace/runs/train/$NAME
DRIVER_LOG=/workspace/logs/${NAME}_driver.log
CKPT_BASE=/workspace/ckpts/skyrl
UPLOAD_MARKERS=$LOG_PATH/uploads
mkdir -p /workspace/runs/train

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }
fail() { log "FAILED: $*"; exit 1; }

# Print "<batch> <state_path>" of the last checkpoint record, or "0 -" when there is none.
last_checkpoint() {
  python3 - "$LOG_PATH/checkpoints.jsonl" <<'EOF'
import json, sys
batch, state = 0, "-"
try:
    for line in open(sys.argv[1]):
        if line.strip():
            rec = json.loads(line)
            if rec.get("state_path"):
                batch, state = rec["batch"], rec["state_path"]
except FileNotFoundError:
    pass
print(batch, state)
EOF
}

# Print the state_path saved at exactly <batch>, or "-".
state_at() {
  python3 - "$LOG_PATH/checkpoints.jsonl" "$1" <<'EOF'
import json, sys
state = "-"
try:
    for line in open(sys.argv[1]):
        if line.strip():
            rec = json.loads(line)
            if rec.get("batch") == int(sys.argv[2]) and rec.get("state_path"):
                state = rec["state_path"]
except FileNotFoundError:
    pass
print(state)
EOF
}

eval_summary_ok() {
  python3 - "$1" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if s["run"]["num_errors"] == 0 and s["by_source"]["OVERALL"]["n"] > 0 else 2)
EOF
}

curl -sf -X POST "$RETRIEVAL_URL/retrieve" -H "Content-Type: application/json" \
  -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null \
  || fail "retrieval server is not answering at $RETRIEVAL_URL"

# Segment targets: EVAL_EVERY, 2*EVAL_EVERY, ..., plus TOTAL_STEPS if it is not a multiple.
TARGETS=$(seq "$EVAL_EVERY" "$EVAL_EVERY" "$TOTAL_STEPS")
[ $((TOTAL_STEPS % EVAL_EVERY)) -ne 0 ] && TARGETS="$TARGETS $TOTAL_STEPS"
log "run=$NAME targets: $(echo $TARGETS) batch=${BATCH_SIZE}x${GROUP_SIZE} save_every=$SAVE_EVERY extra: $*"

PREV_T=0
UPLOAD_PIDS=""
for T in $TARGETS; do
  # 1. Train to step T.
  read -r BATCH STATE <<<"$(last_checkpoint)"
  if [ "$BATCH" -lt "$T" ]; then
    log "training: step $BATCH -> $T"
    /workspace/bin/restart_skyrl.sh "${NAME}_server_train_$T" >>"$DRIVER_LOG" 2>&1 \
      || fail "server restart failed before training to step $T; see $DRIVER_LOG"
    (cd "$PROJ" && uv run --frozen python -m search_r1.train \
      batch_size="$BATCH_SIZE" group_size="$GROUP_SIZE" max_steps="$T" save_every="$SAVE_EVERY" \
      log_path="$LOG_PATH" wandb_name="$NAME" behavior_if_log_dir_exists=resume "$@" \
      >"/workspace/logs/${NAME}_train_to_$T.log" 2>&1)
    read -r BATCH STATE <<<"$(last_checkpoint)"
    [ "$BATCH" -ge "$T" ] || fail "training to step $T stopped at checkpoint $BATCH; see /workspace/logs/${NAME}_train_to_$T.log"
    log "training done: checkpoint at step $BATCH ($STATE)"
  else
    log "training to step $T already done (last checkpoint at $BATCH)"
  fi

  # 2. Evaluate the checkpoint saved at step T on the dev set.
  STATE_T=$(state_at "$T")
  [ "$STATE_T" != "-" ] || fail "no checkpoint saved at step $T"
  EVAL_RUN=$(printf "%s-step%04d" "$NAME" "$T")
  SUMMARY="$PROJ/runs/eval/$EVAL_RUN/summary.json"
  if eval_summary_ok "$SUMMARY"; then
    log "eval at step $T already done ($EVAL_RUN)"
  else
    log "eval at step $T: $STATE_T -> runs/eval/$EVAL_RUN"
    /workspace/bin/restart_skyrl.sh "${NAME}_server_eval_$T" >>"$DRIVER_LOG" 2>&1 \
      || fail "server restart failed before eval at step $T; see $DRIVER_LOG"
    (cd "$PROJ" && uv run --frozen python -m search_r1.offline_eval backend=tinker \
      training_config_file="$LOG_PATH/config.json" \
      tinker_base_url="$TINKER_BASE_URL" eval_file=eval_sets/dev.jsonl \
      concurrency="$EVAL_CONCURRENCY" tinker_state_url="$STATE_T" run_name="$EVAL_RUN" \
      >"/workspace/logs/${EVAL_RUN}.log" 2>&1)
    eval_summary_ok "$SUMMARY" || fail "eval at step $T incomplete or with errors; see /workspace/logs/${EVAL_RUN}.log"
    log "eval at step $T done: $(python3 -c "import json; s=json.load(open('$SUMMARY'))['by_source']['OVERALL']; print('EM %.3f F1 %.3f answered %.3f' % (s['em'], s['f1'], s['answer_rate']))")"
  fi

  # 3. Free disk: drop full states no longer needed (eval steps and the latest are kept).
  if [ "$PRUNE_STATES" = "1" ]; then
    python3 /workspace/bin/ckpt_prune.py "$LOG_PATH/checkpoints.jsonl" "$CKPT_BASE" \
      "$(echo $TARGETS | tr ' ' ',')" >>"$DRIVER_LOG" 2>&1 || log "WARNING: prune failed; see $DRIVER_LOG"
  fi

  # 4. Upload this eval step's checkpoint to W&B in the background; training goes on.
  if [ "$UPLOAD" = "1" ] && [ ! -f "$UPLOAD_MARKERS/$(printf "step%04d" "$T").json" ]; then
    UPLOAD_LOG=$(mktemp "/workspace/logs/${NAME}_upload_step$(printf '%04d' "$T").log.XXXXXX") \
      || fail "cannot create upload log for step $T"
    setsid nohup "$PROJ/.venv/bin/python" /workspace/bin/upload_checkpoint.py \
      --run-name "$NAME" --log-path "$LOG_PATH" --step "$T" --prev-step "$PREV_T" \
      --eval-summary "$SUMMARY" --ckpt-base "$CKPT_BASE" --marker-dir "$UPLOAD_MARKERS" \
      >"$UPLOAD_LOG" 2>&1 </dev/null &
    UPLOAD_PIDS="$UPLOAD_PIDS $!"
    log "upload of step $T started in the background (pid $!, log $UPLOAD_LOG)"
  fi
  PREV_T=$T
done

tmux kill-session -t skyrl 2>/dev/null
(cd "$SKYRL" && uv run --frozen --extra tinker ray stop --force >/dev/null 2>&1)
log "training done: step $TOTAL_STEPS, dev evals at: $(echo $TARGETS); server stopped"

# Wait for the uploads started in this invocation, then check every eval step has a marker.
if [ "$UPLOAD" = "1" ]; then
  [ -n "$UPLOAD_PIDS" ] && log "waiting for uploads:$UPLOAD_PIDS" && wait $UPLOAD_PIDS
  MISSING=""
  for T in $TARGETS; do
    [ -f "$UPLOAD_MARKERS/$(printf "step%04d" "$T").json" ] || MISSING="$MISSING $T"
  done
  if [ -n "$MISSING" ]; then
    log "WARNING: W&B upload missing for steps:$MISSING (see /workspace/logs/${NAME}_upload_step*.log.*;"
    log "         rerun this script to retry them). Do not terminate the pod yet."
    exit 1
  fi
  log "all eval-step checkpoints are on W&B ($UPLOAD_MARKERS); the pod can be terminated"
fi
