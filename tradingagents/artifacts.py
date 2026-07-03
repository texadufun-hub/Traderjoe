"""Structured run-artifact writer and GCS uploader (Phase 1).

The framework's human-facing output is the markdown report tree written by
:mod:`tradingagents.reporting`. This module adds the *machine*-facing artifact:
a single JSON document capturing the Portfolio Manager's typed decision fields
(pulled straight from the retained ``PortfolioDecision`` instance, not scraped
from prose) plus every per-agent report section as clean text.

Layout written under ``results_dir`` for a run with
``base = {ticker}_{date}_{run_id}``:

    {base}.json                 structured artifact (the agent's food)
    {base}.md                   consolidated human report (= complete_report.md)
    {base}/                     full human report tree (numbered sections)
    {base}/sections/{key}.md    one clean-text file per section, for GCS parity

All of the above except the numbered human tree is mirrored to GCS:

    gs://{bucket}/{base}.json
    gs://{bucket}/{base}.md
    gs://{bucket}/{base}/sections/{key}.md

Design contract (locked): JSON is the agent's food, MD is the human artifact.
Local write and the side-correctness check always complete first; GCS upload is
best-effort — every upload is wrapped so a failure warns but never aborts the run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.agents.schemas import PortfolioDecision, signal_from_rating
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.reporting import write_report_tree

logger = logging.getLogger(__name__)

DEFAULT_GCS_BUCKET = "report-analyzis"

# GCP project billed for the upload (the `x-goog-user-project` header). The
# bucket lives here; pinning it means the upload does not depend on the caller's
# ADC "quota project" being set correctly — a mismatched/inaccessible quota
# project otherwise fails every upload with "User project ... is invalid".
# Override with the GCS_PROJECT env var when the bucket moves.
DEFAULT_GCS_PROJECT = "firestore-492119"

# Map a 5-tier rating *string* (free-text fallback path) to a BUY/HOLD/SELL
# signal, mirroring schemas.signal_from_rating for the typed path.
_RATING_STR_TO_SIGNAL = {
    "buy": "BUY",
    "overweight": "BUY",
    "hold": "HOLD",
    "underweight": "SELL",
    "sell": "SELL",
}

# section_key -> where to read the clean text from in final_state. Each callable
# takes final_state and returns a str (possibly empty). Kept declarative so the
# JSON section keys, the per-section .md filenames, and the source of truth stay
# in one place.
_SECTION_SOURCES: dict[str, Any] = {
    "market_report": lambda s: s.get("market_report"),
    "sentiment_report": lambda s: s.get("sentiment_report"),
    "news_report": lambda s: s.get("news_report"),
    "fundamentals_report": lambda s: s.get("fundamentals_report"),
    "bull_argument": lambda s: (s.get("investment_debate_state") or {}).get("bull_history"),
    "bear_argument": lambda s: (s.get("investment_debate_state") or {}).get("bear_history"),
    "research_manager_plan": lambda s: (s.get("investment_debate_state") or {}).get("judge_decision"),
    "trader_plan": lambda s: s.get("trader_investment_plan"),
    "risk_aggressive": lambda s: (s.get("risk_debate_state") or {}).get("aggressive_history"),
    "risk_conservative": lambda s: (s.get("risk_debate_state") or {}).get("conservative_history"),
    "risk_neutral": lambda s: (s.get("risk_debate_state") or {}).get("neutral_history"),
    "pm_rationale": lambda s: (s.get("risk_debate_state") or {}).get("judge_decision"),
}


def _clean(text: Any) -> str | None:
    """Return stripped text, or None when empty/absent — for JSON section values."""
    if not text:
        return None
    cleaned = str(text).strip()
    return cleaned or None


def _extract_sections(final_state: dict) -> dict[str, str | None]:
    """Pull every per-agent section as clean text, keyed by JSON section key."""
    return {key: _clean(source(final_state)) for key, source in _SECTION_SOURCES.items()}


def _apply_side_correctness(
    signal: str,
    entry: float | None,
    target: float | None,
    stop: float | None,
    warning: str | None,
) -> str | None:
    """Validate target/stop sit on the correct side of entry; flag (never mutate) on violation.

        BUY:  stop_loss < entry_reference_price < target_price
        SELL: target_price < entry_reference_price < stop_loss

    HOLD is skipped. The check only fires when all three levels are non-null so a
    partially-specified decision is never falsely flagged. On failure a note is
    appended to ``warning_message`` — the prices themselves are left untouched.
    """
    if signal not in ("BUY", "SELL"):
        return warning
    if entry is None or target is None or stop is None:
        return warning

    ok = (stop < entry < target) if signal == "BUY" else (target < entry < stop)
    if ok:
        return warning

    note = (
        f"[SIDE VIOLATION] target/stop do not satisfy {signal} side constraint — "
        "verify before trading."
    )
    return f"{warning}\n{note}".strip() if warning else note


def build_run_artifact(
    final_state: dict,
    pm_decision: PortfolioDecision | None,
    ticker: str,
    analysis_date: str,
    run_id: str,
    resolved_config: dict,
) -> dict:
    """Assemble the structured JSON artifact from the typed PM decision + report sections.

    When ``pm_decision`` is the typed instance (structured-output path) all
    decision fields come straight from it. When it is None (free-text fallback)
    only the signal is recoverable — parsed from the rendered decision prose —
    and the remaining typed fields are null.
    """
    if pm_decision is not None:
        signal = signal_from_rating(pm_decision.rating)
        size_fraction = pm_decision.size_fraction
        confidence = pm_decision.confidence
        entry_reference_price = pm_decision.entry_reference_price
        target_price = pm_decision.price_target
        stop_loss = pm_decision.stop_loss
        currency = pm_decision.currency or "USD"
        time_horizon_days = pm_decision.time_horizon_days
        rationale = pm_decision.investment_thesis
        warning_message = pm_decision.warning_message
    else:
        rating_str = parse_rating(final_state.get("final_trade_decision", ""))
        signal = _RATING_STR_TO_SIGNAL.get(rating_str.strip().lower(), "HOLD")
        size_fraction = confidence = entry_reference_price = None
        target_price = stop_loss = time_horizon_days = None
        rationale = warning_message = None
        currency = "USD"

    warning_message = _apply_side_correctness(
        signal, entry_reference_price, target_price, stop_loss, warning_message
    )

    return {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "size_fraction": size_fraction,
        "confidence": confidence,
        "entry_reference_price": entry_reference_price,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "currency": currency,
        "time_horizon_days": time_horizon_days,
        "rationale": rationale,
        "warning_message": warning_message,
        "resolved_config": resolved_config,
        "sections": _extract_sections(final_state),
    }


def _write_local_artifacts(
    final_state: dict,
    artifact: dict,
    ticker: str,
    base: str,
    results_dir: str | Path,
) -> tuple[Path, Path, list[Path]]:
    """Write the JSON artifact, consolidated .md, and per-section .md files locally.

    Returns ``(json_path, complete_md_path, section_paths)`` for the uploader.
    Local write always happens before any GCS attempt.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = results_dir / base

    # 1. JSON artifact (the agent's food).
    json_path = results_dir / f"{base}.json"
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Consolidated human report. Reuse the shared report-tree writer so the
    #    .md matches exactly what the CLI produces, then surface it as {base}.md.
    complete_report = write_report_tree(final_state, ticker, run_dir)
    complete_md_path = results_dir / f"{base}.md"
    complete_md_path.write_text(complete_report.read_text(encoding="utf-8"), encoding="utf-8")

    # 3. One clean-text .md per section, mirrored to what GCS expects.
    sections_dir = run_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    section_paths: list[Path] = []
    for key, text in artifact["sections"].items():
        if text is None:
            continue
        section_path = sections_dir / f"{key}.md"
        section_path.write_text(text, encoding="utf-8")
        section_paths.append(section_path)

    return json_path, complete_md_path, section_paths


def _upload_to_gcs(
    bucket_name: str,
    base: str,
    json_path: Path,
    complete_md_path: Path,
    section_paths: list[Path],
    gcs_project: str | None = None,
) -> str | None:
    """Best-effort upload of the full artifact set. Returns the JSON GCS URI or None.

    Auth is via Application Default Credentials. The billing/quota project is
    pinned (``gcs_project`` -> ``GCS_PROJECT`` env -> ``DEFAULT_GCS_PROJECT``) so
    a caller whose ADC quota project is unset or points at an inaccessible
    project does not fail every upload with "User project ... is invalid".
    Client construction and every individual upload are wrapped so a failure
    warns but never raises.
    """
    import os

    try:
        import google.auth
        from google.cloud import storage
    except ImportError:
        logger.warning(
            "google-cloud-storage not installed; skipping GCS upload of run artifacts"
        )
        return None

    project = gcs_project or os.environ.get("GCS_PROJECT") or DEFAULT_GCS_PROJECT
    try:
        credentials, _ = google.auth.default()
        # Pin the quota/billing project on the credentials so the request's
        # x-goog-user-project header is a project the caller can actually use.
        if hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(project)
        client = storage.Client(project=project, credentials=credentials)
        bucket = client.bucket(bucket_name)
    except Exception as exc:  # noqa: BLE001 — auth/config issues must not abort the run
        logger.warning("Could not initialise GCS client (%s); skipping upload", exc)
        return None

    def _put(blob_name: str, local_path: Path) -> bool:
        try:
            bucket.blob(blob_name).upload_from_filename(str(local_path))
            return True
        except Exception as exc:  # noqa: BLE001 — per-object failure is non-fatal
            logger.warning("GCS upload failed for gs://%s/%s (%s)", bucket_name, blob_name, exc)
            return False

    json_blob = f"{base}.json"
    json_ok = _put(json_blob, json_path)
    _put(f"{base}.md", complete_md_path)
    for section_path in section_paths:
        _put(f"{base}/sections/{section_path.name}", section_path)

    return f"gs://{bucket_name}/{json_blob}" if json_ok else None


def persist_run_artifacts(
    final_state: dict,
    pm_decision: PortfolioDecision | None,
    ticker: str,
    analysis_date: str,
    run_id: str,
    results_dir: str | Path,
    resolved_config: dict,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> dict:
    """Build + write the structured artifact locally, then fire-and-log the GCS upload.

    Returns the artifact dict (also written to disk) so callers can log or
    inspect it. Local write and the side-correctness check complete before any
    GCS work; the upload is best-effort and never raises.
    """
    base = f"{safe_ticker_component(ticker)}_{analysis_date}_{run_id}"

    artifact = build_run_artifact(
        final_state, pm_decision, ticker, analysis_date, run_id, resolved_config
    )

    json_path, complete_md_path, section_paths = _write_local_artifacts(
        final_state, artifact, ticker, base, results_dir
    )
    logger.info("Wrote structured run artifact: %s", json_path)

    gcs_uri = _upload_to_gcs(gcs_bucket, base, json_path, complete_md_path, section_paths)
    if gcs_uri:
        logger.info("Uploaded run artifact set to GCS: %s", gcs_uri)

    return artifact
