"""Side-correctness validation for structured trade decisions.

Checks that stop-loss and price-target levels sit on the correct side of
the entry price for the stated trade direction.  These invariants are not
enforced by the Pydantic schemas (cross-field validation would also need an
entry-price reference on PortfolioDecision, which that schema doesn't carry),
so this module provides them as post-hoc checks for smoke tests, batch
audits, and the defect-regression suite.

Standard conventions checked here:
  BUY  (long):   stop_loss  < entry_price          (stop below entry)
  BUY  (long):   price_target > entry_price         (target above entry)
  SELL (short):  stop_loss  > entry_price          (stop above entry)
  SELL (short):  price_target < entry_price         (target below entry)

Fields that are None are silently skipped (no level → no violation).
HOLD / OVERWEIGHT / UNDERWEIGHT ratings are not directionally constrained.
"""
from __future__ import annotations

from dataclasses import dataclass

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)


@dataclass
class SideViolation:
    field: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return f"{self.field}: expected {self.expected}, got {self.actual}"


def check_trader_proposal(proposal: TraderProposal) -> list[SideViolation]:
    """Return side-correctness violations for a TraderProposal, or [] if clean."""
    violations: list[SideViolation] = []
    action = proposal.action
    entry = proposal.entry_price
    stop = proposal.stop_loss

    if action == TraderAction.HOLD or entry is None or stop is None:
        return violations

    if action == TraderAction.BUY and stop >= entry:
        violations.append(SideViolation(
            field="stop_loss",
            expected=f"< entry_price ({entry})",
            actual=str(stop),
        ))
    elif action == TraderAction.SELL and stop <= entry:
        violations.append(SideViolation(
            field="stop_loss",
            expected=f"> entry_price ({entry})",
            actual=str(stop),
        ))

    return violations


def check_pm_decision(
    decision: PortfolioDecision,
    entry_price: float,
) -> list[SideViolation]:
    """Return side-correctness violations for a PortfolioDecision, or [] if clean.

    ``entry_price`` must be supplied by the caller because PortfolioDecision
    carries only the price target, not the reference entry.
    """
    violations: list[SideViolation] = []
    target = decision.price_target

    if target is None:
        return violations

    if decision.rating in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
        if target <= entry_price:
            violations.append(SideViolation(
                field="price_target",
                expected=f"> entry_price ({entry_price})",
                actual=str(target),
            ))
    elif decision.rating in (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT):
        if target >= entry_price:
            violations.append(SideViolation(
                field="price_target",
                expected=f"< entry_price ({entry_price})",
                actual=str(target),
            ))

    return violations
