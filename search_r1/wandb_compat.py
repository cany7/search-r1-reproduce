"""Persist W&B identity for the pinned cookbook's synchronous training loop."""

from collections.abc import Iterator
from contextlib import contextmanager

from tinker_cookbook.utils import ml_log


@contextmanager
def persistent_wandb_run() -> Iterator[None]:
    """Enable cookbook event steps as soon as its W&B logger is initialized.

    Cookbook 1991897 only enables this on its async path by default. Its existing
    implementation writes wandb_run.json for online runs, validates/restores that
    identity on restart, and logs the model step as training/step. Replayed steps
    remain separate events instead of being dropped by W&B's monotonic history.
    Scope the logging-only override to this CLI invocation; leave dependencies
    on disk, local metric rows, checkpoints and the training loop unchanged.
    """
    original = ml_log.WandbLogger

    class PersistentWandbLogger(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_event_steps()

    ml_log.WandbLogger = PersistentWandbLogger
    try:
        yield
    finally:
        ml_log.WandbLogger = original
