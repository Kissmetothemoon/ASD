"""Engine-neutral score-provider protocol."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from asd.budget import TokenScores


class TargetScoreProvider(Protocol):
    """Compress target verification output into the scores ASD needs."""

    def score_draft(
        self,
        draft_token_ids: Sequence[int],
        target_output: Any,
    ) -> TokenScores:
        ...
