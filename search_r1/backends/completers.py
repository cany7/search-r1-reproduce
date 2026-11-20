# Derived from tinker-cookbook/completers.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local imports and a focused text-search API.
# SPDX-License-Identifier: Apache-2.0
# License and source mapping: third_party/tinker_cookbook/.

from dataclasses import dataclass

import tinker

StopCondition = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    """A sequence of token IDs with optional log-probabilities and a stop reason."""

    tokens: list[int]
    maybe_logprobs: list[float] | None
    stop_reason: tinker.StopReason = "stop"
    # Tinker's id for the sampled sequence, used for provenance; None if the completer has none.
    # https://tinker-docs.thinkingmachines.ai/tinker/datum/#provenance
    sequence_id: str | None = None
    # Prompt tokens billed as prefix-cache hits for this sample (SampleResponse.prompt_cache_hit_tokens).
    prompt_cache_hit_tokens: int = 0

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    """Abstract interface for generating tokens from a prompt."""

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        *,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        """Generate a token sequence from the given model input."""
        raise NotImplementedError


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """Token completer that uses a tinker.SamplingClient to sample actions."""

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    context_window: int | None = None

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        *,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        if max_tokens is not None:
            max_tokens = min(self.max_tokens, max_tokens)
        else:
            max_tokens = self.max_tokens
        if self.context_window is not None:
            max_tokens = min(max_tokens, self.context_window - model_input.length)
            if max_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({model_input.length}) exceeds context window "
                    f"({self.context_window}). No room for generation."
                )

        # Sample from the model
        sample_result = await self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=max_tokens,
                temperature=self.temperature,
            ),
        )

        # Extract tokens, logprobs, and stop_reason from the first (and only) sample
        sampled_seq = sample_result.sequences[0]
        assert sampled_seq.logprobs is not None

        return TokensWithLogprobs(
            tokens=sampled_seq.tokens,
            maybe_logprobs=sampled_seq.logprobs,
            stop_reason=sampled_seq.stop_reason,
            sequence_id=sampled_seq.sequence_id,
            prompt_cache_hit_tokens=sample_result.prompt_cache_hit_tokens,
        )
