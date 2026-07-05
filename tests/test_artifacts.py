"""Structured run-artifact: JSON assembly, local write, and GCS quota-project pin.

No network: the GCS client is mocked. Verifies the JSON is written locally
regardless of upload outcome, and that the uploader pins the billing/quota
project so a wrong ADC quota project doesn't fail every upload.
"""
import json
import unittest
from pathlib import Path
from unittest import mock

import pytest

from tradingagents import artifacts

_FINAL_STATE = {
    "market_report": "Market strong.",
    "sentiment_report": "Positive.",
    "news_report": "Good news.",
    "fundamentals_report": "Solid.",
    "investment_debate_state": {
        "bull_history": "bull", "bear_history": "bear", "judge_decision": "long"
    },
    "risk_debate_state": {
        "aggressive_history": "a", "conservative_history": "c",
        "neutral_history": "n", "judge_decision": "pm rationale",
    },
    "trader_investment_plan": "buy plan",
    "final_trade_decision": "BUY",
    "pm_decision": None,  # free-text fallback path
}


@pytest.mark.unit
class BuildArtifactTests(unittest.TestCase):
    def test_free_text_fallback_recovers_signal(self):
        art = artifacts.build_run_artifact(
            _FINAL_STATE, None, "NVDA", "2024-05-10", "rid", {"k": "v"}
        )
        self.assertEqual(art["signal"], "BUY")
        self.assertIsNone(art["size_fraction"])  # only signal recoverable
        self.assertEqual(art["sections"]["market_report"], "Market strong.")
        self.assertEqual(art["sections"]["pm_rationale"], "pm rationale")


@pytest.mark.unit
class LocalWriteTests(unittest.TestCase):
    def test_json_written_even_when_upload_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(artifacts, "_upload_to_gcs", return_value=None):
            art = artifacts.persist_run_artifacts(
                final_state=_FINAL_STATE, pm_decision=None, ticker="NVDA",
                analysis_date="2024-05-10", run_id="rid",
                results_dir=d, resolved_config={"k": "v"},
            )
            json_path = Path(d) / "NVDA_2024-05-10_rid.json"
            self.assertTrue(json_path.exists())
            self.assertEqual(json.loads(json_path.read_text())["signal"], "BUY")
            self.assertEqual(art["signal"], "BUY")


@pytest.mark.unit
class UploadQuotaProjectTests(unittest.TestCase):
    def _run_upload(self, env, gcs_project=None):
        fake_creds = mock.Mock()
        fake_creds.with_quota_project.return_value = fake_creds
        fake_bucket = mock.Mock()
        fake_client = mock.Mock()
        fake_client.bucket.return_value = fake_bucket

        with mock.patch.dict("os.environ", env, clear=False), \
                mock.patch("google.auth.default", return_value=(fake_creds, "adc-proj")), \
                mock.patch("google.cloud.storage.Client", return_value=fake_client) as m_client:
            artifacts._upload_to_gcs(
                "bucket", "BASE", Path("j.json"), Path("c.md"), [],
                gcs_project=gcs_project,
            )
        return fake_creds, m_client

    def test_pins_default_project(self):
        creds, m_client = self._run_upload({})
        creds.with_quota_project.assert_called_once_with(artifacts.DEFAULT_GCS_PROJECT)
        self.assertEqual(
            m_client.call_args.kwargs["project"], artifacts.DEFAULT_GCS_PROJECT
        )

    def test_env_overrides_project(self):
        creds, m_client = self._run_upload({"GCS_PROJECT": "env-proj"})
        creds.with_quota_project.assert_called_once_with("env-proj")
        self.assertEqual(m_client.call_args.kwargs["project"], "env-proj")

    def test_explicit_arg_wins_over_env(self):
        creds, _ = self._run_upload({"GCS_PROJECT": "env-proj"}, gcs_project="arg-proj")
        creds.with_quota_project.assert_called_once_with("arg-proj")


if __name__ == "__main__":
    unittest.main()
