#!/bin/bash
# Start the SkyRL Tinker API server for Qwen3.5-4B (FSDP, single GPU).
# Extra backend-config JSON entries can be passed in EXTRA_CFG, e.g. EXTRA_CFG='"trainer.micro_train_batch_size_per_gpu": 2'
source "$(dirname -- "${BASH_SOURCE[0]}")/env_common.sh" || exit 1
# Serialized through backend config to Ray workers; 0 restores always-on GC.
GC_SHORT_SEQ_THRESHOLD=${GC_SHORT_SEQ_THRESHOLD:-8000}
if ! [[ "$GC_SHORT_SEQ_THRESHOLD" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "GC_SHORT_SEQ_THRESHOLD must be a nonnegative integer" >&2
  exit 1
fi
set -e
mkdir -p "$POD_ROOT/state" "$CKPT_BASE" "$POD_ROOT/logs/skyrl-infra" "$POD_ROOT/ckpts/skyrl-train" "$POD_ROOT/ckpts/skyrl-export"
CFG=$(python3 - "$POD_ROOT" "$GC_SHORT_SEQ_THRESHOLD" "${EXTRA_CFG:-}" <<'PYCFG'
import json, sys
root, threshold, extra = sys.argv[1:]
cfg = {
    "trainer.policy.language_model_only": True,
    "trainer.ref.language_model_only": True,
    "generator.inference_engine.language_model_only": True,
    "trainer.remove_microbatch_padding": False,
    "trainer.log_path": root + "/logs/skyrl-infra",
    "trainer.ckpt_path": root + "/ckpts/skyrl-train",
    "trainer.export_path": root + "/ckpts/skyrl-export",
    "trainer.gradient_checkpointing_short_seq_threshold": int(threshold),
}
print(json.dumps(cfg | json.loads("{" + extra + "}")))
PYCFG
)
echo "backend-config: $CFG"
cd "$SKYRL"
exec uv run --frozen --extra tinker --extra fsdp -m skyrl.tinker.api --base-model Qwen/Qwen3.5-4B --backend fsdp \
  --checkpoints-base "$CKPT_BASE" \
  --database-url "sqlite:///$POD_ROOT/state/tinker.db" \
  --backend-config "$CFG"
