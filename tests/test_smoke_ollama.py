"""Unit tests for the ollama smoke path — no live Ollama required.

These tests verify:
1. "ollama" is registered in PROVIDER_DEFAULTS with model qwen3:8b.
2. The smoke script's five structure checks (PASS/FAIL logic) work correctly
   with mocked LLM output.
3. The free-text fallback path (thinking-model None return) is detectable,
   and its raw output shape is observable.
4. The Ollama provider routes to the OpenAI-compatible registry with the
   expected default base URL (localhost:11434/v1).

Run without Ollama:  pytest tests/test_smoke_ollama.py -m unit
Run live smoke:      OLLAMA_BASE_URL=... python scripts/smoke_structured_output.py ollama
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_trader_proposal,
)


# ---------------------------------------------------------------------------
# PROVIDER_DEFAULTS registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaProviderRegistration:
    def _defaults(self):
        import scripts.smoke_structured_output as smoke
        return smoke.PROVIDER_DEFAULTS

    def test_ollama_key_present(self):
        assert "ollama" in self._defaults()

    def test_ollama_model_is_qwen3_8b(self):
        model, _ = self._defaults()["ollama"]
        assert model == "qwen3:8b"

    def test_ollama_no_api_key_sentinel(self):
        _, api_key = self._defaults()["ollama"]
        assert api_key is None, "Ollama needs no API key; sentinel should be None"

    def test_ollama_is_last_entry(self):
        keys = list(self._defaults().keys())
        assert keys[-1] == "ollama", (
            "ollama should be the last entry (added after upstream defaults)"
        )


# ---------------------------------------------------------------------------
# Smoke structure checks (PASS / FAIL logic)
# ---------------------------------------------------------------------------


def _minimal_rm_output():
    plan = ResearchPlan(
        recommendation=PortfolioRating.BUY,
        rationale="Bull case carried.",
        strategic_actions="Build gradually over two weeks.",
    )
    return render_research_plan(plan)


def _minimal_trader_output():
    proposal = TraderProposal(
        action=TraderAction.BUY,
        reasoning="Technicals strong, momentum intact.",
        entry_price=192.53,
        stop_loss=178.00,
    )
    return render_trader_proposal(proposal)


def _minimal_pm_output():
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Enter at 192-195, 6% portfolio cap.",
        investment_thesis="AI capex cycle intact; institutional flows constructive.",
        price_target=225.00,
        time_horizon="3-6 months",
    )
    return render_pm_decision(decision)


@pytest.mark.unit
class TestSmokeStructureChecks:
    """Mirrors the 5-item check list in scripts/smoke_structured_output.py."""

    def test_rm_recommendation_header_present(self):
        assert "**Recommendation**:" in _minimal_rm_output()

    def test_trader_action_header_present(self):
        assert "**Action**:" in _minimal_trader_output()

    def test_trader_final_proposal_header_present(self):
        assert "FINAL TRANSACTION PROPOSAL:" in _minimal_trader_output()

    def test_pm_rating_header_present(self):
        assert "**Rating**:" in _minimal_pm_output()

    def test_pm_executive_summary_header_present(self):
        assert "**Executive Summary**:" in _minimal_pm_output()

    def test_pm_investment_thesis_header_present(self):
        assert "**Investment Thesis**:" in _minimal_pm_output()

    def test_all_three_outputs_pass_smoke_checks(self):
        checks = [
            ("Research Manager", _minimal_rm_output(),   ["**Recommendation**:"]),
            ("Trader",           _minimal_trader_output(), ["**Action**:", "FINAL TRANSACTION PROPOSAL:"]),
            ("Portfolio Manager", _minimal_pm_output(),  ["**Rating**:", "**Executive Summary**:", "**Investment Thesis**:"]),
        ]
        failures = []
        for name, text, required in checks:
            for marker in required:
                if marker not in text:
                    failures.append(f"{name}: missing {marker!r}")
        assert failures == [], "\n".join(failures)


# ---------------------------------------------------------------------------
# Fallback detection: what happens when structured output returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFallbackBehavior:
    """Tests for invoke_structured_or_freetext fallback path.

    qwen3:8b is a thinking model; it can answer in plain text and leave the
    structured parser with None (v0.3.0 changelog: 'a thinking model that
    returns no parsed result still falls back to free text').  These tests
    verify: (a) the fallback fires and (b) the raw LLM content is returned
    verbatim so the smoke test can observe it.
    """

    def _invoke_with_none_structured(self, freetext_content: str) -> str:
        from tradingagents.agents.utils.structured import invoke_structured_or_freetext

        structured = MagicMock()
        structured.invoke.return_value = None  # thinking model returned nothing parsed

        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content=freetext_content)

        return invoke_structured_or_freetext(
            structured, plain, "prompt",
            render=lambda r: r.rating,
            agent_name="test-agent",
        )

    def test_none_structured_triggers_freetext_fallback(self):
        result = self._invoke_with_none_structured("Some freetext response.")
        assert result == "Some freetext response."

    def test_fallback_output_is_raw_llm_content(self):
        raw = "**Action**: Sell\n\nGuidance cut.\n\nFINAL TRANSACTION PROPOSAL: **SELL**"
        result = self._invoke_with_none_structured(raw)
        assert result == raw

    def test_well_formed_freetext_still_passes_structure_checks(self):
        """A fallback with correct headers passes the smoke checks."""
        freetext = (
            "**Action**: Sell\n\n"
            "Guidance cut, multiple compression risk.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        result = self._invoke_with_none_structured(freetext)
        assert "**Action**:" in result
        assert "FINAL TRANSACTION PROPOSAL:" in result

    def test_malformed_freetext_fails_structure_checks(self):
        """A degenerate fallback (missing headers) is detectable before downstream use."""
        freetext = "I think you should sell. Target 209.92, stop 204.24."
        result = self._invoke_with_none_structured(freetext)
        assert "**Action**:" not in result
        assert "FINAL TRANSACTION PROPOSAL:" not in result


# ---------------------------------------------------------------------------
# Ollama provider routes through OpenAI-compatible registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ollama_routes_to_openai_compatible_registry():
    """create_llm_client('ollama', ...) must resolve through is_openai_compatible."""
    from tradingagents.llm_clients.openai_client import is_openai_compatible
    assert is_openai_compatible("ollama"), (
        "'ollama' must be in OPENAI_COMPATIBLE_PROVIDERS — "
        "needed for tool_choice suppression fix in v0.3.0"
    )


@pytest.mark.unit
def test_ollama_default_base_url(monkeypatch):
    """Without OLLAMA_BASE_URL, the client defaults to localhost:11434/v1."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    mod = importlib.reload(
        importlib.import_module("tradingagents.llm_clients.openai_client")
    )
    client = mod.OpenAIClient(model="qwen3:8b", provider="ollama")
    assert "localhost:11434" in str(client.get_llm().openai_api_base)


@pytest.mark.unit
def test_ollama_base_url_override(monkeypatch):
    """OLLAMA_BASE_URL env var must be respected for remote swarm routing."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://swarm-host:11434/v1")
    mod = importlib.reload(
        importlib.import_module("tradingagents.llm_clients.openai_client")
    )
    client = mod.OpenAIClient(model="qwen3:8b", provider="ollama")
    assert "swarm-host" in str(client.get_llm().openai_api_base)
