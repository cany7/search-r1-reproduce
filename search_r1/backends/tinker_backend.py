"""Explicit SDK operations used by evaluation and training; no cookbook or global SDK patches."""

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import tinker

from search_r1.backends.completers import TinkerTokenCompleter

logger = logging.getLogger(__name__)
SessionStatus = Literal["success", "errored", "interrupted"]


@contextmanager
def training_session(service, *, skyrl_compat):
    """Older combined servers omit the finish endpoint. Only tolerate its 404 on success."""
    active = service.__enter__()
    try:
        yield active
    except BaseException as failure:
        try:
            service.__exit__(type(failure), failure, failure.__traceback__)
        except Exception:
            logger.warning("SDK cleanup also failed after the training error", exc_info=True)
        raise
    else:
        # The training loop has returned after publishing its final checkpoint.
        try:
            service.__exit__(None, None, None)
        except tinker.NotFoundError as error:
            if not (
                skyrl_compat
                and error.status_code == 404
                and error.request.method == "POST"
                and re.fullmatch(r"/api/v1/sessions/[^/]+/finish", error.request.url.path)
            ):
                raise
            logger.warning("SkyRL lacks session finish after successful checkpoint publication")


async def close_service(service_client, status: SessionStatus) -> None:
    """Finish the SDK session (including heartbeat/HTTP cleanup).

    Older SkyRL servers may reject finish-session. The SDK still cleans its local
    holders in a finally block; report the error without hiding evaluation output
    or replacing the exception that caused cleanup.
    """
    try:
        await service_client.close(status=status)
    except Exception:
        logger.warning("Could not finish evaluation SDK session", exc_info=True)


@dataclass(kw_only=True)
class TinkerEvaluationPolicy(TinkerTokenCompleter):
    """Own the session and loaded model for the entire evaluation lifetime."""

    service_client: tinker.ServiceClient = field(repr=False)
    training_client: tinker.TrainingClient | None = field(default=None, repr=False)

    async def aclose(self, status: SessionStatus = "success") -> None:
        await close_service(self.service_client, status)


async def load_training_client(
    service,
    state_path,
    model_name,
    lora_rank,
    *,
    with_optimizer=False,
    skyrl_compat=True,
    user_metadata=None,
):
    """SkyRL accepts create-client then load-state, including its older Tinker protocol."""
    metadata = {"user_metadata": user_metadata} if user_metadata is not None else {}
    if not skyrl_compat:
        create = (
            service.create_training_client_from_state_with_optimizer
            if with_optimizer
            else service.create_training_client_from_state
        )
        return await create(state_path, **metadata)
    client = await service.create_lora_training_client(model_name, rank=lora_rank, **metadata)
    if with_optimizer:
        await client.load_state_with_optimizer(state_path)
    else:
        await client.load_state(state_path)
    return client


async def get_renderer_name_from_checkpoint_async(service_client, checkpoint_path) -> str | None:
    """Read historical renderer metadata, falling back when a server lacks it.

    Adapted from tinker-cookbook/checkpoint_utils.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
    Apache-2.0; see third_party/tinker_cookbook/. Local logging/imports replace cookbook's.
    """
    try:
        run = await service_client.create_rest_client().get_training_run_by_tinker_path(
            checkpoint_path
        )
        return (run.user_metadata or {}).get("renderer_name")
    except (tinker.TinkerError, ValueError) as error:
        logger.warning(
            "Could not fetch renderer metadata for checkpoint %s: %s", checkpoint_path, error
        )
        return None
