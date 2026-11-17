#!/bin/bash
# Start the SkyRL Tinker API server for Qwen3.5-4B (FSDP, single GPU).
# Extra backend-config JSON entries can be passed in EXTRA_CFG, e.g. EXTRA_CFG='"trainer.micro_train_batch_size_per_gpu": 2'
source /workspace/bin/env_common.sh
# Serialized through backend config to Ray workers; 0 restores always-on GC.
GC_SHORT_SEQ_THRESHOLD=${GC_SHORT_SEQ_THRESHOLD:-8000}
if ! [[ "$GC_SHORT_SEQ_THRESHOLD" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "GC_SHORT_SEQ_THRESHOLD must be a nonnegative integer" >&2
  exit 1
fi
mkdir -p /workspace/state /workspace/ckpts/skyrl /workspace/logs/skyrl-infra /workspace/ckpts/skyrl-train /workspace/ckpts/skyrl-export
CFG='{"trainer.policy.language_model_only": true, "trainer.ref.language_model_only": true, "generator.inference_engine.language_model_only": true, "trainer.remove_microbatch_padding": false, "trainer.log_path": "/workspace/logs/skyrl-infra", "trainer.ckpt_path": "/workspace/ckpts/skyrl-train", "trainer.export_path": "/workspace/ckpts/skyrl-export"'
CFG="$CFG, \"trainer.gradient_checkpointing_short_seq_threshold\": $GC_SHORT_SEQ_THRESHOLD"
if [ -n "${EXTRA_CFG:-}" ]; then CFG="$CFG, $EXTRA_CFG"; fi
CFG="$CFG}"
echo "backend-config: $CFG"
cd $SKYRL
exec uv run --frozen --extra tinker --extra fsdp -m skyrl.tinker.api --base-model Qwen/Qwen3.5-4B --backend fsdp \
  --checkpoints-base /workspace/ckpts/skyrl \
  --database-url sqlite:////workspace/state/tinker.db \
  --backend-config "$CFG"
