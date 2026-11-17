"""Compatibility with the self-hosted SkyRL Tinker server.

SkyRL 79d4c89 implements the tinker 0.27 protocol; the locked client is tinker 0.39. The
client's ``ServiceClient.create_training_client_from_state[_with_optimizer]`` sends one
``load_weights`` request without a ``model_id`` (a newer protocol), which SkyRL rejects with
422. The cookbook uses these calls to resume training, and ``offline_eval.py`` uses the first
to evaluate a saved state.

``patch_create_from_state`` replaces both with the older two-step form: create a LoRA training
client, then ``load_state`` / ``load_state_with_optimizer`` on it. Hosted Tinker accepts that
form too (the SDK marks it deprecated), so the patch is safe to leave on.
"""

import logging

import tinker

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_create_from_state(model_name: str, lora_rank: int) -> None:
    """Make ``create_training_client_from_state[_with_optimizer]`` work against SkyRL.

    Args:
        model_name: Base model of the saved state, used when the caller passes none.
        lora_rank: LoRA rank of the saved state; must match the run that saved it.
    """
    global _PATCHED
    if _PATCHED:
        return

    async def _create(
        self: tinker.ServiceClient,
        path: str,
        base_model: str | None,
        user_metadata: dict[str, str] | None,
        with_optimizer: bool,
    ):
        training_client = await self.create_lora_training_client(
            base_model or model_name, rank=lora_rank, user_metadata=user_metadata
        )
        if with_optimizer:
            await training_client.load_state_with_optimizer(path)
        else:
            await training_client.load_state(path)
        logger.info(
            "Loaded %s into a new LoRA training client (rank %d, optimizer state %s)",
            path,
            lora_rank,
            "restored" if with_optimizer else "not restored",
        )
        return training_client

    async def from_state(self, path, base_model=None, user_metadata=None, **_kwargs):
        return await _create(self, path, base_model, user_metadata, with_optimizer=False)

    async def from_state_with_optimizer(self, path, base_model=None, user_metadata=None, **_kwargs):
        return await _create(self, path, base_model, user_metadata, with_optimizer=True)

    tinker.ServiceClient.create_training_client_from_state = from_state  # type: ignore[method-assign]
    tinker.ServiceClient.create_training_client_from_state_with_optimizer = (  # type: ignore[method-assign]
        from_state_with_optimizer
    )
    _PATCHED = True
