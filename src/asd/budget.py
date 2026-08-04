"""Pure-Python reference implementation of ASD prefix selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .config import ASDConfig


@dataclass(frozen=True)
class TokenScores:
    """Compact target scores needed by ASD for one draft block."""

    top_logits: tuple[float, ...]
    top_token_ids: tuple[int, ...]
    draft_logits: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.top_logits),
            len(self.top_token_ids),
            len(self.draft_logits),
        }
        if len(lengths) != 1:
            raise ValueError("all compact score fields must have equal length")
        if not all(math.isfinite(value) for value in self.top_logits):
            raise ValueError("top_logits must be finite")
        if not all(math.isfinite(value) for value in self.draft_logits):
            raise ValueError("draft_logits must be finite")


@dataclass
class RequestRiskState:
    total_budget: float
    spent: float = 0.0
    relaxed_tokens: int = 0
    events: list[dict[str, float | int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_budget) or self.total_budget < 0:
            raise ValueError("total_budget must be non-negative")
        if (
            not math.isfinite(self.spent)
            or self.spent < 0
            or self.spent > self.total_budget + 1e-9
        ):
            raise ValueError("spent must lie within the total budget")

    @property
    def remaining(self) -> float:
        return max(0.0, float(self.total_budget) - float(self.spent))

    def charge(self, *, position: int, regret: float, value: float) -> None:
        if (
            not math.isfinite(regret)
            or regret < 0
            or regret > self.remaining + 1e-9
        ):
            raise ValueError("regret charge exceeds the remaining request budget")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("token value must be positive")
        self.spent += float(regret)
        self.relaxed_tokens += 1
        self.events.append(
            {
                "position": int(position),
                "regret": float(regret),
                "value": float(value),
                "cumulative_spent": float(self.spent),
            }
        )


@dataclass(frozen=True)
class PrefixDecision:
    accepted_tokens: int
    exact_tokens: int
    relaxed_tokens: int
    risk_spent: float
    stopped_on_position: int | None
    token_regrets: tuple[float, ...]
    token_values: tuple[float, ...]


def default_suffix_values(length: int) -> tuple[float, ...]:
    return tuple(float(length - index) for index in range(length))


def choose_prefix(
    *,
    draft_token_ids: Sequence[int],
    scores: TokenScores,
    state: RequestRiskState,
    config: ASDConfig,
    position_values: Sequence[float] | None = None,
) -> PrefixDecision:
    """Return the longest contiguous prefix satisfying all hard constraints."""

    draft = tuple(int(value) for value in draft_token_ids)
    if len(draft) != len(scores.top_logits):
        raise ValueError("draft length must match target score length")
    if abs(state.total_budget - config.risk_budget) > 1e-9:
        raise ValueError("state total_budget must match config risk_budget")
    values = (
        default_suffix_values(len(draft))
        if position_values is None
        else tuple(float(value) for value in position_values)
    )
    if len(values) != len(draft) or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError("position_values must be positive and match draft length")

    regrets = tuple(
        max(0.0, float(top_logit) - float(draft_logit))
        for top_logit, draft_logit in zip(scores.top_logits, scores.draft_logits)
    )
    mismatches = tuple(
        token_id != top_id
        for token_id, top_id in zip(draft, scores.top_token_ids)
    )

    strict_identity = (
        config.risk_budget == 0.0
        or config.max_regret_per_value == 0.0
        or config.max_relaxed_tokens_per_block == 0
    )
    cumulative_risk = 0.0
    cumulative_relaxed = 0
    accepted = 0
    for index, (mismatch, regret, value) in enumerate(
        zip(mismatches, regrets, values)
    ):
        if mismatch:
            if strict_identity:
                break
            cumulative_risk += regret
            cumulative_relaxed += 1
        worthwhile = (not mismatch) or (
            regret / value <= config.max_regret_per_value
        )
        affordable = cumulative_risk <= state.remaining + 1e-9
        within_cap = (
            cumulative_relaxed <= config.max_relaxed_tokens_per_block
        )
        if not (worthwhile and affordable and within_cap):
            break
        accepted = index + 1

    spent_before = state.spent
    for index in range(accepted):
        if mismatches[index]:
            state.charge(
                position=index,
                regret=regrets[index],
                value=values[index],
            )

    exact = sum(not mismatch for mismatch in mismatches[:accepted])
    return PrefixDecision(
        accepted_tokens=accepted,
        exact_tokens=exact,
        relaxed_tokens=accepted - exact,
        risk_spent=state.spent - spent_before,
        stopped_on_position=accepted if accepted < len(draft) else None,
        token_regrets=regrets,
        token_values=values,
    )
