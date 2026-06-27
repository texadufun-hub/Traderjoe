"""Regression suite for the defect class: wrong-side stop/target levels.

The motivating failure (Traderjg run c4702ee0, NVIDIA SELL):
  entry=192.53, target=209.92 (above entry), stop=204.24 (above entry)
  — both levels in the BUY direction on a SELL decision.

This test module:
1. Confirms the v0.3.0 Pydantic schemas do NOT enforce side-correctness
   at parse time (no built-in cross-field validator) — this is the gap.
2. Tests the side_check utility that provides the post-hoc invariant.
3. Reproduces the original NVIDIA defect in v0.3.0 schema terms:
   - Trader SELL: stop_loss on the wrong side of entry (→ TraderProposal)
   - PM SELL rating: price_target above entry (→ PortfolioDecision)
4. Confirms valid orientations produce no violations.
"""
from __future__ import annotations

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.utils.side_check import (
    SideViolation,
    check_pm_decision,
    check_trader_proposal,
)


# ---------------------------------------------------------------------------
# Gap confirmation: schemas accept wrong-side values without raising
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchemaGap:
    """The Pydantic schemas are deliberately non-restrictive on price levels.
    These tests document that gap — wrong-side values parse without error,
    making post-hoc checks like side_check necessary."""

    def test_trader_sell_accepts_stop_below_entry(self):
        # BUG orientation: stop < entry for a SELL (should be stop > entry)
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="Guidance cut.",
            entry_price=192.53,
            stop_loss=180.00,  # WRONG for SELL — below entry
        )
        assert p.stop_loss == 180.00  # schema accepted it silently

    def test_trader_buy_accepts_stop_above_entry(self):
        # BUG orientation: stop > entry for a BUY (should be stop < entry)
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong momentum.",
            entry_price=192.53,
            stop_loss=205.00,  # WRONG for BUY — above entry
        )
        assert p.stop_loss == 205.00  # schema accepted it silently

    def test_pm_sell_accepts_target_above_entry(self):
        # NVIDIA defect repro in PM terms: SELL rating + target above entry
        d = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="Exit position.",
            investment_thesis="Guidance cut, multiple compression.",
            price_target=209.92,  # WRONG for SELL — above entry=192.53
        )
        assert d.price_target == 209.92  # schema accepted it silently


# ---------------------------------------------------------------------------
# TraderProposal side-correctness checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckTraderProposal:

    # -- BUY --

    def test_buy_clean_no_violations(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex intact.",
            entry_price=192.53,
            stop_loss=178.00,  # correctly below entry
        )
        assert check_trader_proposal(p) == []

    def test_buy_stop_above_entry_is_violation(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Technicals strong.",
            entry_price=192.53,
            stop_loss=205.00,  # wrong side for BUY
        )
        violations = check_trader_proposal(p)
        assert len(violations) == 1
        assert violations[0].field == "stop_loss"
        assert "< entry_price" in violations[0].expected

    def test_buy_stop_equal_to_entry_is_violation(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=192.53,
            stop_loss=192.53,  # equal is also invalid (no buffer)
        )
        assert len(check_trader_proposal(p)) == 1

    # -- SELL --

    def test_sell_clean_no_violations(self):
        # Correct SELL: stop ABOVE entry (where you cut the short loss)
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="Guidance cut.",
            entry_price=192.53,
            stop_loss=204.24,  # correctly above entry
        )
        assert check_trader_proposal(p) == []

    def test_sell_stop_below_entry_is_violation(self):
        # Stop below entry on a SELL = wrong side
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="Multiple compression.",
            entry_price=192.53,
            stop_loss=180.00,  # wrong side for SELL
        )
        violations = check_trader_proposal(p)
        assert len(violations) == 1
        assert violations[0].field == "stop_loss"
        assert "> entry_price" in violations[0].expected

    def test_sell_stop_equal_to_entry_is_violation(self):
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            entry_price=192.53,
            stop_loss=192.53,
        )
        assert len(check_trader_proposal(p)) == 1

    # -- HOLD / None fields --

    def test_hold_never_flagged(self):
        p = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="Wait for catalyst.",
            entry_price=192.53,
            stop_loss=180.00,
        )
        assert check_trader_proposal(p) == []

    def test_no_entry_price_skipped(self):
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            entry_price=None,
            stop_loss=180.00,
        )
        assert check_trader_proposal(p) == []

    def test_no_stop_loss_skipped(self):
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="r",
            entry_price=192.53,
            stop_loss=None,
        )
        assert check_trader_proposal(p) == []

    # -- Violation string repr --

    def test_violation_str_is_human_readable(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="r",
            entry_price=192.53,
            stop_loss=205.00,
        )
        v = check_trader_proposal(p)[0]
        s = str(v)
        assert "stop_loss" in s
        assert "192.53" in s
        assert "205.0" in s


# ---------------------------------------------------------------------------
# PortfolioDecision side-correctness checks (NVIDIA defect repro)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckPmDecision:
    ENTRY = 192.53  # NVIDIA entry price from the motivating bug

    # -- SELL / UNDERWEIGHT --

    def test_sell_target_below_entry_clean(self):
        d = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="Exit position.",
            investment_thesis="Guidance cut.",
            price_target=175.00,  # correctly below entry
        )
        assert check_pm_decision(d, self.ENTRY) == []

    def test_nvidia_defect_sell_target_above_entry(self):
        """Exact reproduction of the motivating bug: SELL, target=209.92 > entry=192.53."""
        d = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="Exit.",
            investment_thesis="Thesis.",
            price_target=209.92,  # THE BUG: above entry on a SELL
        )
        violations = check_pm_decision(d, self.ENTRY)
        assert len(violations) == 1
        assert violations[0].field == "price_target"
        assert "< entry_price" in violations[0].expected
        assert "209.92" in violations[0].actual

    def test_underweight_target_above_entry_is_violation(self):
        d = PortfolioDecision(
            rating=PortfolioRating.UNDERWEIGHT,
            executive_summary="Trim.",
            investment_thesis="Risk rising.",
            price_target=210.00,
        )
        assert len(check_pm_decision(d, self.ENTRY)) == 1

    def test_sell_target_equal_to_entry_is_violation(self):
        d = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="s",
            investment_thesis="t",
            price_target=self.ENTRY,  # equal = no edge
        )
        assert len(check_pm_decision(d, self.ENTRY)) == 1

    # -- BUY / OVERWEIGHT --

    def test_buy_target_above_entry_clean(self):
        d = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Build position.",
            investment_thesis="AI tailwind.",
            price_target=225.00,  # correctly above entry
        )
        assert check_pm_decision(d, self.ENTRY) == []

    def test_buy_target_below_entry_is_violation(self):
        d = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="s",
            investment_thesis="t",
            price_target=180.00,  # wrong side for BUY
        )
        violations = check_pm_decision(d, self.ENTRY)
        assert len(violations) == 1
        assert "> entry_price" in violations[0].expected

    def test_overweight_target_below_entry_is_violation(self):
        d = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="s",
            investment_thesis="t",
            price_target=180.00,
        )
        assert len(check_pm_decision(d, self.ENTRY)) == 1

    # -- HOLD / None target --

    def test_hold_never_flagged(self):
        d = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="s",
            investment_thesis="t",
            price_target=150.00,  # any target value — HOLD is not directional
        )
        assert check_pm_decision(d, self.ENTRY) == []

    def test_no_price_target_skipped(self):
        d = PortfolioDecision(
            rating=PortfolioRating.SELL,
            executive_summary="s",
            investment_thesis="t",
            price_target=None,
        )
        assert check_pm_decision(d, self.ENTRY) == []
