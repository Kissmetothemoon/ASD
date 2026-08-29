"""Readable pure-Python reference rule for ASD prefix selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .asd_config import DSparkASDConfig


def _finite_non_negative(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True, slots=True)
class TokenScores:
    """Compact target scores for one draft block.

    All three fields cover draft positions only and must have equal length.
    Candidate/logit alignment is an engine-adapter responsibility tested in
    P03, not guessed by this engine-neutral module.
    """

    top_logits: tuple[float, ...]
    top_token_ids: tuple[int, ...]
    draft_logits: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_logits",
            tuple(float(value) for value in self.top_logits),
        )
        object.__setattr__(
            self,
            "top_token_ids",
            tuple(int(value) for value in self.top_token_ids),
        )
        object.__setattr__(
            self,
            "draft_logits",
            tuple(float(value) for value in self.draft_logits),
        )
        lengths = {
            len(self.top_logits),
            len(self.top_token_ids),
            len(self.draft_logits),
        }
        if len(lengths) != 1:
            raise ValueError("all compact score fields must have equal length")


@dataclass(slots=True)
class RequestRiskState:
    """Reference request state whose budget persists across draft blocks."""

    total_budget: float
    spent: float = 0.0

    def __post_init__(self) -> None:
        self.total_budget = _finite_non_negative(
            self.total_budget,
            name="total_budget",
        )
        self.spent = _finite_non_negative(self.spent, name="spent")
        if self.spent > self.total_budget:
            raise ValueError("spent must not exceed total_budget")

    @property
    def remaining(self) -> float:
        return self.total_budget - self.spent

    def reset(self, *, total_budget: float | None = None) -> None:
        """Initialize this state for a new request without leaking old spend."""

        if total_budget is not None:
            self.total_budget = _finite_non_negative(
                total_budget,
                name="total_budget",
            )
        self.spent = 0.0

    def _charge(self, regret: float) -> None:
        charge = _finite_non_negative(regret, name="regret")
        if charge > self.remaining:
            raise ValueError("regret charge exceeds the remaining request budget")
        self.spent += charge


@dataclass(frozen=True, slots=True)
class PrefixDecision:
    """Observable result of applying ASD to one block."""

    accepted_tokens: int
    exact_tokens: int
    relaxed_mismatches: int
    risk_spent: float
    stopped_on_position: int | None
    token_regrets: tuple[float, ...]
    token_values: tuple[float, ...]
    mismatched: tuple[bool, ...]

    @property
    def relaxed_tokens(self) -> int:
        """Backward-compatible alias for accepted relaxed mismatches."""

        return self.relaxed_mismatches


def normalized_suffix_values(length: int) -> tuple[float, ...]:
    """Return ``((L-i)/L)`` for every position in a positive-length block."""

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length <= 0:
        raise ValueError("length must be positive")
    return tuple((length - index) / length for index in range(length))


default_suffix_values = normalized_suffix_values


def choose_prefix(
    *,
    draft_token_ids: Sequence[int],
    scores: TokenScores,
    state: RequestRiskState,
    config: DSparkASDConfig,
) -> PrefixDecision:
    """Return the longest contiguous prefix satisfying all three ASD gates.

    Exact matches are free.  A mismatch is accepted only when its
    ``regret/value`` is at most ``g``, the cumulative charge fits within the
    request's remaining ``B``, and the block has accepted no more than ``m``
    mismatches.  Only accepted mismatches mutate ``state``.
    """

    draft = tuple(int(value) for value in draft_token_ids)
    block = len(draft)
    if block != len(scores.top_logits):
        raise ValueError("draft length must match target score length")
    config.validate_block_size(block)
    if state.total_budget != config.risk_budget:
        raise ValueError(
            f"state total_budget={state.total_budget} does not match "
            f"config B={config.risk_budget}"
        )

    regrets = tuple(
        max(0.0, top_logit - draft_logit)
        for top_logit, draft_logit in zip(
            scores.top_logits,
            scores.draft_logits,
        )
    )
    mismatched = tuple(
        draft_token_id != top_token_id
        for draft_token_id, top_token_id in zip(
            draft,
            scores.top_token_ids,
        )
    )
    values = normalized_suffix_values(block)

    remaining_before_block = state.remaining
    cumulative_risk = 0.0
    cumulative_relaxed = 0
    accepted = 0

    for index, (mismatch, regret, value) in enumerate(zip(mismatched, regrets, values)):
        if mismatch:
            risk_before_token = cumulative_risk
            cumulative_risk += regret
            cumulative_relaxed += 1
            worthwhile = regret / value <= config.g
            affordable = (
                risk_before_token < remaining_before_block
                and cumulative_risk <= remaining_before_block
            )
            within_cap = cumulative_relaxed <= config.m
            if not (worthwhile and affordable and within_cap):
                break
        accepted = index + 1

    spent_before = state.spent
    for index in range(accepted):
        if mismatched[index]:
            state._charge(regrets[index])

    exact_tokens = sum(not mismatch for mismatch in mismatched[:accepted])
    return PrefixDecision(
        accepted_tokens=accepted,
        exact_tokens=exact_tokens,
        relaxed_mismatches=accepted - exact_tokens,
        risk_spent=state.spent - spent_before,
        stopped_on_position=accepted if accepted < block else None,
        token_regrets=regrets,
        token_values=values,
        mismatched=mismatched,
    )
