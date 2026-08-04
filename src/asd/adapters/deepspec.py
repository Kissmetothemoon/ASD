"""Adapter for full target logits returned by DeepSpec/DSpark."""

from __future__ import annotations

from typing import Any, Sequence

from asd.adapters.base import TargetScoreProvider
from asd.budget import PrefixDecision, RequestRiskState, TokenScores, choose_prefix
from asd.config import ASDConfig


class FullLogitScoreProvider:
    """Extract compact scores from a `[K, V]` torch tensor."""

    def score_draft(
        self,
        draft_token_ids: Sequence[int],
        target_output: Any,
    ) -> TokenScores:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "install approximate-speculative-decoding[torch] for "
                "full-logit scoring"
            ) from exc

        logits = target_output
        if hasattr(target_output, "logits"):
            logits = target_output.logits
        if logits.ndim == 3:
            if logits.shape[0] != 1:
                raise ValueError("the reference adapter expects batch_size=1")
            logits = logits[0]
        if logits.ndim == 2 and logits.shape[0] == len(draft_token_ids) + 1:
            logits = logits[: len(draft_token_ids)]
        if logits.ndim != 2 or logits.shape[0] != len(draft_token_ids):
            raise ValueError("target logits must have shape [K, V]")

        draft_ids = torch.tensor(
            list(draft_token_ids),
            dtype=torch.long,
            device=logits.device,
        )
        top_logits, top_ids = logits.max(dim=-1)
        draft_logits = logits.gather(1, draft_ids.unsqueeze(1)).squeeze(1)
        return TokenScores(
            top_logits=tuple(
                float(value) for value in top_logits.detach().float().cpu()
            ),
            top_token_ids=tuple(int(value) for value in top_ids.detach().cpu()),
            draft_logits=tuple(
                float(value) for value in draft_logits.detach().float().cpu()
            ),
        )


class DeepSpecDSparkAdapter:
    """Minimal decision layer inserted after DSpark target verification."""

    def __init__(
        self,
        config: ASDConfig,
        score_provider: TargetScoreProvider | None = None,
    ) -> None:
        self.config = config
        self.score_provider = score_provider or FullLogitScoreProvider()

    def decide(
        self,
        *,
        draft_token_ids: Sequence[int],
        target_output: Any,
        state: RequestRiskState,
        position_values: Sequence[float] | None = None,
    ) -> PrefixDecision:
        scores = self.score_provider.score_draft(draft_token_ids, target_output)
        return choose_prefix(
            draft_token_ids=draft_token_ids,
            scores=scores,
            state=state,
            config=self.config,
            position_values=position_values,
        )
