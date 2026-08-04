import unittest

from asd.budget import RequestRiskState, TokenScores, choose_prefix
from asd.config import ASDConfig


class BudgetTests(unittest.TestCase):
    def test_zero_budget_matches_strict_prefix(self):
        config = ASDConfig(0.0, 1.0, 2)
        state = RequestRiskState(0.0)
        decision = choose_prefix(
            draft_token_ids=[1, 9, 3],
            scores=TokenScores(
                top_logits=(5.0, 5.0, 5.0),
                top_token_ids=(1, 2, 3),
                draft_logits=(5.0, 4.9, 5.0),
            ),
            state=state,
            config=config,
        )
        self.assertEqual(decision.accepted_tokens, 1)
        self.assertEqual(decision.relaxed_tokens, 0)
        self.assertEqual(state.spent, 0.0)

    def test_cheap_mismatch_can_be_relaxed(self):
        config = ASDConfig(1.0, 0.25, 1)
        state = RequestRiskState(1.0)
        decision = choose_prefix(
            draft_token_ids=[9, 2, 8],
            scores=TokenScores(
                top_logits=(5.0, 5.0, 5.0),
                top_token_ids=(1, 2, 3),
                draft_logits=(4.8, 5.0, 4.9),
            ),
            state=state,
            config=config,
        )
        self.assertEqual(decision.accepted_tokens, 2)
        self.assertEqual(decision.exact_tokens, 1)
        self.assertEqual(decision.relaxed_tokens, 1)
        self.assertAlmostEqual(state.spent, 0.2)
        self.assertEqual(decision.stopped_on_position, 2)

    def test_request_budget_persists_across_blocks(self):
        config = ASDConfig(0.3, 1.0, 1)
        state = RequestRiskState(0.3)
        scores = TokenScores((5.0,), (1,), (4.8,))
        first = choose_prefix(
            draft_token_ids=[2], scores=scores, state=state, config=config
        )
        second = choose_prefix(
            draft_token_ids=[2], scores=scores, state=state, config=config
        )
        self.assertEqual(first.accepted_tokens, 1)
        self.assertEqual(second.accepted_tokens, 0)
        self.assertAlmostEqual(state.spent, 0.2)

    def test_prefix_reachability_blocks_later_exact_token(self):
        config = ASDConfig(1.0, 0.01, 2)
        state = RequestRiskState(1.0)
        decision = choose_prefix(
            draft_token_ids=[9, 2],
            scores=TokenScores((5.0, 5.0), (1, 2), (4.0, 5.0)),
            state=state,
            config=config,
        )
        self.assertEqual(decision.accepted_tokens, 0)

    def test_zero_budget_is_strict_even_when_mismatch_has_zero_regret(self):
        config = ASDConfig(0.0, 1.0, 2)
        decision = choose_prefix(
            draft_token_ids=[2],
            scores=TokenScores((5.0,), (1,), (5.0,)),
            state=RequestRiskState(0.0),
            config=config,
        )
        self.assertEqual(decision.accepted_tokens, 0)

    def test_state_budget_must_match_configuration(self):
        with self.assertRaises(ValueError):
            choose_prefix(
                draft_token_ids=[],
                scores=TokenScores((), (), ()),
                state=RequestRiskState(1.0),
                config=ASDConfig(2.0, 0.25, 1),
            )


if __name__ == "__main__":
    unittest.main()
