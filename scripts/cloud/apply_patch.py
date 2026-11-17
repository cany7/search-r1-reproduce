"""Apply padding trimming and length-based activation checkpointing to SkyRL 79d4c89.

Why: the Tinker backend left-pads every sequence of a forward_backward batch to the batch's
longest sequence. With trainer.remove_microbatch_padding=false (required for Qwen3.5 on FSDP),
each microbatch is then computed at that full padded length. Trimming the columns that are
padding for every row of the microbatch removes the wasted compute without changing the loss:
padded positions have loss_mask 0, and per-sample outputs are read from the right by valid length.
"""
import pathlib, sys
root = pathlib.Path(sys.argv[1])
wu = root / "skyrl/backends/skyrl_train/workers/worker_utils.py"
wk = root / "skyrl/backends/skyrl_train/workers/worker.py"
cfg = root / "skyrl/train/config/config.py"

HELPER = '''

# --- search-r1 local patch (not upstream): trim shared left padding per microbatch -------------
_TRIM_SEQ_KEYS = ("sequences", "attention_mask")
_TRIM_RESP_KEYS = (
    "action_log_probs", "base_action_log_probs", "values", "returns", "advantages",
    "loss_mask", "response_mask", "rollout_logprobs",
)
_TRIM_SKIP_KEYS = (
    "rollout_expert_indices", "router_padding_mask", SAMPLE_SUPPORT_FIELD,
    "pixel_values", "image_grid_thw", "sub_seq_lengths",
)


def trim_microbatch_left_padding(batch: TrainingInputBatch) -> TrainingInputBatch:
    """Drop the left-padding columns shared by every row of a left-padded microbatch.

    Returns the batch unchanged when it is not in the plain left-padded text layout, when it
    carries fields this trim does not understand, or when SKYRL_TRIM_MICROBATCH_PADDING=0.
    """
    import os

    if os.environ.get("SKYRL_TRIM_MICROBATCH_PADDING", "1") == "0":
        return batch
    am = batch.get("attention_mask")
    rm = batch.get("response_mask")
    if am is None or rm is None or am.dim() != 2 or rm.dim() != 2 or not batch.metadata:
        return batch
    if any(batch.get(k) is not None for k in _TRIM_SKIP_KEYS):
        return batch
    seq_len, resp_len = am.shape[1], rm.shape[1]
    if batch.metadata.get("response_length") != resp_len:
        return batch
    keep_seq = int(am.sum(dim=-1).max().item())
    keep_resp = int(rm.sum(dim=-1).max().item())
    if keep_seq >= seq_len and keep_resp >= resp_len:
        return batch
    # Only trim a strictly left-padded layout: every mask must be zero in the dropped columns.
    if am[:, : seq_len - keep_seq].any() or rm[:, : resp_len - keep_resp].any():
        return batch
    if keep_resp > keep_seq - 1:
        return batch
    new = {}
    for key, value in batch.items():
        if value is not None and key in _TRIM_SEQ_KEYS:
            value = value[:, seq_len - keep_seq :].contiguous()
        elif value is not None and key in _TRIM_RESP_KEYS:
            value = value[:, resp_len - keep_resp :].contiguous()
        new[key] = value
    trimmed = type(batch)(new)
    trimmed.metadata = {**batch.metadata, "response_length": keep_resp}
    return trimmed
# --- end search-r1 local patch ------------------------------------------------------------------
'''

s = wu.read_text()
if "trim_microbatch_left_padding" not in s:
    anchor = "class BaseBatchIterator:"
    assert anchor in s
    s = s.replace(anchor, HELPER.lstrip("\n") + "\n\n" + anchor, 1)
    wu.write_text(s)

w = wk.read_text()
old = '''        for microbatch in microbatch_iterator:
            experience = BaseBatchIterator.batch_to_experience(microbatch)
            microbatch_weight = len(microbatch) / len(data)
            metrics = self._forward_backward_micro(
                experience,
                microbatch_weight,'''
new = '''        for microbatch in microbatch_iterator:
            # search-r1 local patch: drop left padding shared by the whole microbatch.
            experience = BaseBatchIterator.batch_to_experience(trim_microbatch_left_padding(microbatch))
            microbatch_weight = len(microbatch) / len(data)
            metrics = self._forward_backward_micro(
                experience,
                microbatch_weight,'''
if "trim_microbatch_left_padding(microbatch)" not in w:
    assert w.count(old) == 1, w.count(old)
    w = w.replace(old, new, 1)
    imp = "from skyrl.backends.skyrl_train.workers.worker_utils import ("
    assert imp in w
    w = w.replace(imp, imp + "\n    trim_microbatch_left_padding,", 1)
    wk.write_text(w)

# Keep the baseline initialization (including checkpoint functions and input-grad
# hooks). Only change layer flags during a complete forward/backward microbatch.
GC_HELPER = '''

# --- search-r1 length-based activation checkpointing (FSDP, microbatch 1) ---
class LengthBasedGradientCheckpointing:
    def __init__(self, model, cfg, logger):
        self.threshold = cfg.gradient_checkpointing_short_seq_threshold
        self.logger = logger
        self.modules = []
        self.counts = {"on": 0, "off": 0}
        self.max_lengths = {"on": 0, "off": 0}
        if self.threshold < 0:
            raise ValueError("gradient_checkpointing_short_seq_threshold must be >= 0")
        logger.info(f"[search-r1 gc] short_seq_threshold={self.threshold}; 0 disables switching")
        if self.threshold == 0:
            return
        if cfg.strategy != "fsdp" or not cfg.gradient_checkpointing:
            raise ValueError("Length-based checkpointing requires FSDP and gradient_checkpointing=true")
        if cfg.gradient_checkpointing_use_reentrant:
            raise ValueError("Length-based checkpointing requires use_reentrant=false")
        self.modules = [
            module for module in model.modules()
            if getattr(module, "gradient_checkpointing", False)
        ]
        if not self.modules or any(
            not callable(getattr(module, "_gradient_checkpointing_func", None))
            for module in self.modules
        ):
            raise RuntimeError("No initialized gradient-checkpointing modules found")

    def for_microbatch(self, batch):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            if self.threshold == 0:
                yield
                return
            sequences = batch.get("sequences")
            mask = batch.get("attention_mask")
            # Unknown layouts and larger microbatches keep the initialized GC mode.
            plain = (
                sequences is not None and sequences.dim() == 2
                and sequences.shape[0] == 1 and mask is not None
                and mask.shape == sequences.shape
                and not any(batch.get(k) is not None for k in _TRIM_SKIP_KEYS)
            )
            length = sequences.shape[-1] if sequences is not None and sequences.dim() == 2 else 0
            enabled = not (plain and 0 < length <= self.threshold)
            mode = "on" if enabled else "off"
            if self.counts[mode] == 0:
                self.logger.info(f"[search-r1 gc] first {mode} microbatch: sequence_length={length}")
            try:
                for module in self.modules:
                    module.gradient_checkpointing = enabled
                yield
                self.counts[mode] += 1
                self.max_lengths[mode] = max(self.max_lengths[mode], length)
            finally:
                # Restore even on failure; never retry a partially accumulated backward.
                for module in self.modules:
                    module.gradient_checkpointing = True

        return scope()

    def log_summary(self):
        if self.threshold:
            self.logger.info(
                f"[search-r1 gc] completed microbatches={self.counts}; "
                f"max_sequence_lengths={self.max_lengths}; threshold={self.threshold}"
            )
# --- end search-r1 length-based activation checkpointing ---
'''

s = wu.read_text()
if "class LengthBasedGradientCheckpointing:" not in s:
    assert "class BaseBatchIterator:" in s
    s = s.replace("class BaseBatchIterator:", GC_HELPER.lstrip("\n") + "\n\nclass BaseBatchIterator:", 1)
    wu.write_text(s)

w = wk.read_text()
gc_old = '''        for microbatch in microbatch_iterator:
            # search-r1 local patch: drop left padding shared by the whole microbatch.
            experience = BaseBatchIterator.batch_to_experience(trim_microbatch_left_padding(microbatch))
            microbatch_weight = len(microbatch) / len(data)
            metrics = self._forward_backward_micro(
                experience,
                microbatch_weight,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                return_per_token_outputs=return_per_token_outputs,
            )'''
gc_new = '''        checkpointing = LengthBasedGradientCheckpointing(self.model, self.cfg, logger)
        for microbatch in microbatch_iterator:
            # Use the actual tensor width after trimming, including context and actions.
            microbatch = trim_microbatch_left_padding(microbatch)
            experience = BaseBatchIterator.batch_to_experience(microbatch)
            microbatch_weight = len(microbatch) / len(data)
            with checkpointing.for_microbatch(microbatch):
                metrics = self._forward_backward_micro(
                    experience,
                    microbatch_weight,
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                    return_per_token_outputs=return_per_token_outputs,
                )'''
if "checkpointing = LengthBasedGradientCheckpointing" not in w:
    assert w.count(gc_old) == 1, w.count(gc_old)
    w = w.replace(gc_old, gc_new, 1)
    imp = "from skyrl.backends.skyrl_train.workers.worker_utils import ("
    w = w.replace(imp, imp + "\n    LengthBasedGradientCheckpointing,", 1)
    anchor = "        # Token-based batching packs samples into microbatches out of input order"
    assert w.count(anchor) == 1, w.count(anchor)
    w = w.replace(anchor, "        checkpointing.log_summary()\n\n" + anchor, 1)
    wk.write_text(w)

s = cfg.read_text()
if "gradient_checkpointing_short_seq_threshold:" not in s:
    anchor = "    gradient_checkpointing_use_reentrant: bool = False"
    assert s.count(anchor) == 1, s.count(anchor)
    s = s.replace(anchor, anchor + '''
    gradient_checkpointing_short_seq_threshold: int = 0
    """Local patch: disable GC at or below this tensor width for plain text microbatch 1; 0 keeps GC unchanged."""''', 1)
    cfg.write_text(s)
print("patched padding trimming and length-based gradient checkpointing")
