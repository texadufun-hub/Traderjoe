"""Secret Manager reference detection and best-effort env hydration.

All Secret Manager access is mocked, so these run with no network or credentials.
"""
import unittest
from unittest import mock

import pytest

from tradingagents import secrets

_REF = "projects/733349864865/secrets/FRED_API_KEY/versions/latest"


@pytest.mark.unit
class IsSecretRefTests(unittest.TestCase):
    def test_recognizes_latest_and_numeric_versions(self):
        self.assertTrue(secrets.is_secret_ref(_REF))
        self.assertTrue(
            secrets.is_secret_ref("projects/my-proj/secrets/KEY/versions/7")
        )

    def test_tolerates_surrounding_whitespace(self):
        self.assertTrue(secrets.is_secret_ref(f"  {_REF}\n"))

    def test_rejects_literals_and_empty(self):
        self.assertFalse(secrets.is_secret_ref("9455257fb4c5a30b8664b79a2e52ed45"))
        self.assertFalse(secrets.is_secret_ref("AIzaSyABC123"))
        self.assertFalse(secrets.is_secret_ref(""))
        self.assertFalse(secrets.is_secret_ref(None))

    def test_rejects_malformed_paths(self):
        # Missing /versions/ segment, or a non-numeric/alias version.
        self.assertFalse(secrets.is_secret_ref("projects/p/secrets/KEY"))
        self.assertFalse(
            secrets.is_secret_ref("projects/p/secrets/KEY/versions/newest")
        )


@pytest.mark.unit
class HydrateEnvTests(unittest.TestCase):
    def test_resolves_matching_var_in_place(self):
        env = {"FRED_API_KEY": _REF, "OTHER": "literal-value"}
        with mock.patch.dict("os.environ", env, clear=True), \
                mock.patch.object(secrets, "resolve_secret_ref",
                                  return_value="REALKEY") as m:
            resolved = secrets.hydrate_secret_env(["FRED_API_KEY", "OTHER"])
            import os
            self.assertEqual(resolved, ["FRED_API_KEY"])
            self.assertEqual(os.environ["FRED_API_KEY"], "REALKEY")
            self.assertEqual(os.environ["OTHER"], "literal-value")  # untouched
            m.assert_called_once_with(_REF)

    def test_scans_whole_environment_when_names_omitted(self):
        with mock.patch.dict("os.environ", {"K": _REF}, clear=True), \
                mock.patch.object(secrets, "resolve_secret_ref",
                                  return_value="REALKEY"):
            self.assertEqual(secrets.hydrate_secret_env(), ["K"])

    def test_failure_is_swallowed_and_reference_kept(self):
        # A resolution error must not crash startup, and must leave the path in
        # place so the downstream consumer reports a specific error.
        with mock.patch.dict("os.environ", {"FRED_API_KEY": _REF}, clear=True), \
                mock.patch.object(secrets, "resolve_secret_ref",
                                  side_effect=RuntimeError("403 denied")):
            import os
            self.assertEqual(secrets.hydrate_secret_env(["FRED_API_KEY"]), [])
            self.assertEqual(os.environ["FRED_API_KEY"], _REF)


if __name__ == "__main__":
    unittest.main()
