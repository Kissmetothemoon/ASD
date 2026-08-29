"""Contract tests for the readable ASD reference rule."""

from __future__ import annotations

import unittest

from asd.reproduce.dspark.asd_config import DSparkASDConfig
from asd.reproduce.dspark.rule import (
    RequestRiskState,
    TokenScores,
    choose_prefix,
    normalized_suffix_values,
)


def config(*, B: float, g: float, m: int) -> DSparkASDConfig:
    return DSparkASDConfig(
        risk_budget=B,
        max_regret_per_value=g,
        max_relaxed_mismatches_per_block=m,
        block_size=5,
    )


def scores(
    *,
    top_ids: tuple[int, ...],
    gaps: tuple[float, ...],
    top_logits: tuple[float, ...] | None = None,
) -> TokenScores:
    if top_logits is None:
        top_logits = (5.0,) * len(top_ids)
    return TokenScores(
        top_logits=top_logits,
        top_token_ids=top_ids,
        draft_logits=tuple(top_logit - gap for top_logit, gap in zip(top_logits, gaps)),
    )


class NormalizedSuffixTests(unittest.TestCase):
    def test_width_five_values_are_frozen(self) -> None:
        self.assertEqual(
            normalized_suffix_values(5),
            (1.0, 0.8, 0.6, 0.4, 0.2),
        )

    def test_invalid_width_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            normalized_suffix_values(0)


class ReferencePrefixTests(unittest.TestCase):
    def test_B_zero_is_the_strict_leading_argmax_prefix(self) -> None:
        decision = choose_prefix(
            draft_token_ids=(1, 2, 9, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(0.0, 0.0, 0.1, 0.0, 0.0),
            ),
            state=RequestRiskState(total_budget=0.0),
            config=config(B=0.0, g=1_000_000.0, m=5),
        )
        self.assertEqual(decision.accepted_tokens, 2)
        self.assertEqual(decision.exact_tokens, 2)
        self.assertEqual(decision.relaxed_mismatches, 0)
        self.assertEqual(decision.risk_spent, 0.0)
        self.assertEqual(decision.stopped_on_position, 2)

    def test_zero_regret_mismatch_is_rejected_at_B_zero(self) -> None:
        decision = choose_prefix(
            draft_token_ids=(9, 2, 3, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            state=RequestRiskState(total_budget=0.0),
            config=config(B=0.0, g=0.0, m=5),
        )
        self.assertEqual(decision.accepted_tokens, 0)
        self.assertEqual(decision.risk_spent, 0.0)

    def test_regret_is_non_negative_target_top_gap(self) -> None:
        decision = choose_prefix(
            draft_token_ids=(1, 2, 3, 4, 5),
            scores=TokenScores(
                top_logits=(1.0, 1.0, 1.0, 1.0, 1.0),
                top_token_ids=(1, 2, 3, 4, 5),
                # The compact-input seam can expose a slightly larger draft
                # logit after numeric reduction; regret remains clamped at 0.
                draft_logits=(1.25, 1.0, 0.75, 1.0, 1.0),
            ),
            state=RequestRiskState(total_budget=1.0),
            config=config(B=1.0, g=1.0, m=1),
        )
        self.assertEqual(decision.token_regrets, (0.0, 0.0, 0.25, 0.0, 0.0))
        self.assertEqual(decision.accepted_tokens, 5)
        self.assertEqual(decision.risk_spent, 0.0)

    def test_budget_persists_across_blocks_and_exhaustion_is_strict(self) -> None:
        cfg = config(B=0.25, g=1.0, m=1)
        state = RequestRiskState(total_budget=cfg.B)

        first = choose_prefix(
            draft_token_ids=(9, 2, 3, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(0.25, 0.0, 0.0, 0.0, 0.0),
            ),
            state=state,
            config=cfg,
        )
        second = choose_prefix(
            draft_token_ids=(1, 9, 3, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            state=state,
            config=cfg,
        )

        self.assertEqual(first.accepted_tokens, 5)
        self.assertEqual(first.risk_spent, 0.25)
        self.assertEqual(state.remaining, 0.0)
        self.assertEqual(second.accepted_tokens, 1)
        self.assertEqual(second.relaxed_mismatches, 0)

    def test_budget_exhaustion_inside_a_block_blocks_a_later_tie(self) -> None:
        cfg = config(B=0.25, g=10.0, m=2)
        decision = choose_prefix(
            draft_token_ids=(9, 2, 8, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(0.25, 0.0, 0.0, 0.0, 0.0),
            ),
            state=RequestRiskState(total_budget=cfg.B),
            config=cfg,
        )
        self.assertEqual(decision.accepted_tokens, 2)
        self.assertEqual(decision.relaxed_mismatches, 1)
        self.assertEqual(decision.stopped_on_position, 2)

    def test_reset_clears_spend_before_slot_reuse(self) -> None:
        cfg = config(B=0.5, g=1.0, m=1)
        state = RequestRiskState(total_budget=cfg.B)
        block_scores = scores(
            top_ids=(1, 2, 3, 4, 5),
            gaps=(0.2, 0.0, 0.0, 0.0, 0.0),
        )
        first = choose_prefix(
            draft_token_ids=(9, 2, 3, 4, 5),
            scores=block_scores,
            state=state,
            config=cfg,
        )
        self.assertAlmostEqual(first.risk_spent, 0.2)
        self.assertAlmostEqual(state.remaining, 0.3)

        state.reset(total_budget=cfg.B)
        self.assertEqual(state.spent, 0.0)
        self.assertEqual(state.remaining, cfg.B)
        reused = choose_prefix(
            draft_token_ids=(9, 2, 3, 4, 5),
            scores=block_scores,
            state=state,
            config=cfg,
        )
        self.assertEqual(reused.accepted_tokens, first.accepted_tokens)
        self.assertEqual(reused.risk_spent, first.risk_spent)

    def test_distinct_request_states_do_not_leak_budget(self) -> None:
        cfg = config(B=0.5, g=1.0, m=1)
        first_request = RequestRiskState(total_budget=cfg.B)
        second_request = RequestRiskState(total_budget=cfg.B)
        block_scores = scores(
            top_ids=(1, 2, 3, 4, 5),
            gaps=(0.2, 0.0, 0.0, 0.0, 0.0),
        )
        choose_prefix(
            draft_token_ids=(9, 2, 3, 4, 5),
            scores=block_scores,
            state=first_request,
            config=cfg,
        )
        self.assertAlmostEqual(first_request.remaining, 0.3)
        self.assertEqual(second_request.remaining, cfg.B)
        self.assertEqual(second_request.spent, 0.0)

    def test_failed_barrier_makes_later_exact_tokens_unreachable(self) -> None:
        decision = choose_prefix(
            draft_token_ids=(9, 2, 3, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(1.0, 0.0, 0.0, 0.0, 0.0),
            ),
            state=RequestRiskState(total_budget=10.0),
            config=config(B=10.0, g=0.01, m=5),
        )
        self.assertEqual(decision.accepted_tokens, 0)
        self.assertEqual(decision.risk_spent, 0.0)

    def test_m_one_caps_the_second_relaxed_mismatch(self) -> None:
        decision = choose_prefix(
            draft_token_ids=(9, 2, 8, 4, 5),
            scores=scores(
                top_ids=(1, 2, 3, 4, 5),
                gaps=(0.1, 0.0, 0.1, 0.0, 0.0),
            ),
            state=RequestRiskState(total_budget=10.0),
            config=config(B=10.0, g=10.0, m=1),
        )
        self.assertEqual(decision.accepted_tokens, 2)
        self.assertEqual(decision.relaxed_mismatches, 1)
        self.assertAlmostEqual(decision.risk_spent, 0.1)

    def test_block_size_mismatch_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size=5"):
            choose_prefix(
                draft_token_ids=(1, 2, 3, 4),
                scores=scores(
                    top_ids=(1, 2, 3, 4),
                    gaps=(0.0, 0.0, 0.0, 0.0),
                ),
                state=RequestRiskState(total_budget=0.0),
                config=config(B=0.0, g=0.0, m=0),
            )

    def test_score_field_length_mismatch_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            TokenScores(
                top_logits=(1.0, 1.0),
                top_token_ids=(1,),
                draft_logits=(1.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
