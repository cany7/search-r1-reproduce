"""Local-only token completer for an OpenAI-compatible ``/v1/completions`` server (oMLX).

Used to debug the environment and to compare base models on a Mac, where the
SkyRL Tinker service cannot run. It is not used for training, and scores from it
do not go into the report: the cloud run re-evaluates the base model through
Tinker's own sampling client.

The prompt is still produced by the project renderer. This class only moves it
across a text API: prompt tokens are decoded to a string, the server tokenizes
that string again (verified to give the same ids), and the returned text is
encoded back to tokens so the project's parser sees what it would see from
Tinker. Log-probabilities are not available.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import aiohttp
import tinker

from search_r1.backends.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from search_r1.env.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class OpenAICompletionsTokenCompleter(TokenCompleter):
    """Sample from an OpenAI-compatible text-completions endpoint.

    Args:
        base_url: Server root including ``/v1`` (e.g. ``http://127.0.0.1:8015/v1``).
        api_key: Bearer token for the server.
        model: Model id as the server lists it (e.g. ``Qwen3.5-4B-bf16``).
        tokenizer: The tokenizer the renderer uses for this model.
        max_tokens: Maximum tokens to generate per call.
        temperature: Sampling temperature. 1.0 matches ``TinkerTokenCompleter``.
        max_concurrency: Maximum in-flight requests.
    """

    base_url: str
    api_key: str
    model: str
    tokenizer: Tokenizer
    max_tokens: int
    temperature: float = 1.0
    max_concurrency: int = 8
    request_timeout_seconds: float = 1800.0

    # Number of calls whose re-encoded length differed from the server's own
    # completion-token count. Non-zero means the text round trip lost something.
    token_count_mismatches: int = field(default=0, init=False)
    num_calls: int = field(default=0, init=False)
    _session: aiohttp.ClientSession | None = field(default=None, init=False, repr=False)
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
            )
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        *,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        max_tokens = self.max_tokens if max_tokens is None else min(self.max_tokens, max_tokens)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        stop_token_ids = [s for s in stop if isinstance(s, int)]
        stop_strings = [s for s in stop if isinstance(s, str)]
        body = {
            "model": self.model,
            "prompt": self.tokenizer.decode(model_input.to_ints()),
            "max_tokens": max_tokens,
            # No nucleus / top-k truncation, to match Tinker's plain temperature sampling.
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": 0,
        }
        if stop_strings:
            body["stop"] = stop_strings

        async with (
            self._semaphore,
            self._get_session().post(f"{self.base_url.rstrip('/')}/completions", json=body) as resp,
        ):
            if resp.status != 200:
                raise RuntimeError(f"completions request failed: {resp.status} {await resp.text()}")
            data = await resp.json()

        choice = data["choices"][0]
        tokens: list[int] = self.tokenizer.encode(choice["text"], add_special_tokens=False)
        finished = choice.get("finish_reason") != "length"

        # The server counts the end-of-turn token it consumed; the text does not contain it.
        reported = (data.get("usage") or {}).get("completion_tokens")
        self.num_calls += 1
        if reported is not None and reported - len(tokens) not in ((0, 1) if finished else (0,)):
            self.token_count_mismatches += 1
            logger.warning(
                "re-encoded completion has %d tokens but the server reported %d",
                len(tokens),
                reported,
            )

        if finished and stop_token_ids:
            # The end-of-turn token is not part of the returned text; put it back so the
            # renderer sees a cleanly terminated response.
            tokens.append(stop_token_ids[0])

        return TokensWithLogprobs(
            tokens=tokens,
            maybe_logprobs=None,
            stop_reason="stop" if finished else "length",
        )
