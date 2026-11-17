"""Apply the search-r1 local patch to SkyRL 79d4c89: trim shared left padding per policy microbatch.

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
print("patched")
