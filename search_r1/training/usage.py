# Adapted from tinker-cookbook/utils/cost_estimation.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
from __future__ import annotations

from dataclasses import asdict, dataclass

TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class TokenUsage:
    """Billed token counts by kind, in the same four kinds as :class:`TokenCostRates`."""

    sampled: int = 0
    prefill_cached: int = 0
    prefill_uncached: int = 0
    trained: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            sampled=self.sampled + other.sampled,
            prefill_cached=self.prefill_cached + other.prefill_cached,
            prefill_uncached=self.prefill_uncached + other.prefill_uncached,
            trained=self.trained + other.trained,
        )

    def metrics(self, prefix: str) -> dict[str, int]:
        """The counts as ``<prefix>/<kind>`` metrics."""
        return {f"{prefix}/{kind}": count for kind, count in asdict(self).items()}


@dataclass(frozen=True)
class TokenCostRates:
    """USD per million tokens for each kind of token the trainer is billed for."""

    prefill_usd_per_million: float
    cached_prefill_usd_per_million: float
    sample_usd_per_million: float
    train_usd_per_million: float

    def estimate_usd(self, usage: TokenUsage) -> float:
        """Estimate the USD cost of *usage*."""
        return (
            usage.prefill_uncached * self.prefill_usd_per_million
            + usage.prefill_cached * self.cached_prefill_usd_per_million
            + usage.sampled * self.sample_usd_per_million
            + usage.trained * self.train_usd_per_million
        ) / TOKENS_PER_MILLION


class CostTracker:
    """Accumulate per-step token usage and turn it into ``tokens/`` and ``cost/`` metrics."""

    def __init__(
        self,
        rates: TokenCostRates | None = None,
        *,
        cumulative: TokenUsage = TokenUsage(),
        discarded: TokenUsage = TokenUsage(),
    ):
        self.rates = rates
        self.cumulative = cumulative
        "Running totals, discarded sampling included; persisted for resume."
        self.discarded = discarded
        "The part of :attr:`cumulative` that was sampled for groups dropped before training."
        self.step = TokenUsage()
        "The usage of the latest step recorded."

    def record_discarded_sampling(self, usage: TokenUsage) -> None:
        """Count the sampling of a group that was dropped before training."""
        self.cumulative += usage
        self.discarded += usage

    def cumulative_metrics(self) -> dict[str, float]:
        """The running totals only: ``tokens/cumulative/<kind>``, ``tokens/discarded/<kind>``"""
        metrics: dict[str, float] = {
            **self.cumulative.metrics("tokens/cumulative"),
            **self.discarded.metrics("tokens/discarded"),
        }
        if self.rates is not None:
            metrics["cost/cumulative_usd"] = self.rates.estimate_usd(self.cumulative)
            metrics["cost/discarded_usd"] = self.rates.estimate_usd(self.discarded)
        return metrics

    def record_step(self, usage: TokenUsage) -> dict[str, float]:
        """Add one step's usage and return the metrics to log for it: ``tokens/<kind>``,"""
        self.step = usage
        self.cumulative += usage
        return self._step_metrics()

    def extend_step(self, usage: TokenUsage) -> dict[str, float]:
        """Add usage that belongs to the latest step but was only known after it was"""
        self.step += usage
        self.cumulative += usage
        return self._step_metrics()

    def _step_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            **self.step.metrics("tokens"),
            **self.cumulative.metrics("tokens/cumulative"),
        }
        if self.rates is not None:
            metrics["cost/step_usd"] = self.rates.estimate_usd(self.step)
            metrics["cost/cumulative_usd"] = self.rates.estimate_usd(self.cumulative)
        return metrics
