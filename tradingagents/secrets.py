"""Resolve Google Secret Manager references found in the environment.

An API-key environment variable may hold either a literal key *or* a Secret
Manager resource path, e.g.::

    FRED_API_KEY=projects/733349864865/secrets/FRED_API_KEY/versions/latest

Keeping the *path* in ``.env`` (and in version control) instead of the plaintext
key means the real secret lives only in Secret Manager and is fetched at
startup. This module detects the path form, resolves each one exactly once, and
rewrites ``os.environ`` in place so every downstream ``os.getenv`` consumer sees
the plaintext without knowing Secret Manager exists.

Resolution is best-effort: a missing client library, missing credentials, or an
access error is logged (by variable name and path — never the value) and the
reference is left untouched, so the vendor's own preflight/consumer surfaces a
clear, specific error rather than this layer crashing import for every entry
point.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Canonical Secret Manager version resource name:
#   projects/<project>/secrets/<name>/versions/<version|latest>
# <project> may be a numeric project number or a project ID; <version> is a
# positive integer or the alias "latest". Anything else is treated as a literal
# value and passed through untouched.
_SECRET_REF_RE = re.compile(
    r"^projects/[^/]+/secrets/[^/]+/versions/(?:\d+|latest)$"
)


def is_secret_ref(value: str | None) -> bool:
    """True when ``value`` is a Secret Manager version resource path."""
    return bool(value) and _SECRET_REF_RE.match(value.strip()) is not None


def resolve_secret_ref(ref: str) -> str:
    """Fetch a Secret Manager version path and return its plaintext payload.

    Raises if the client library is unavailable or the access fails; callers
    that want best-effort behaviour should use :func:`hydrate_secret_env`.
    """
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": ref.strip()})
    return response.payload.data.decode("utf-8")


def hydrate_secret_env(var_names: Iterable[str] | None = None) -> list[str]:
    """Resolve env vars whose value is a Secret Manager path, in place.

    Args:
        var_names: Which env vars to consider. ``None`` scans the whole
            environment, so any variable pointing at a secret is resolved with
            no per-variable configuration.

    Returns:
        The names of the variables that were successfully resolved.

    Never raises: a resolution failure is logged (name + path, not value) and
    the original reference is left in place.
    """
    names = list(var_names) if var_names is not None else list(os.environ.keys())
    resolved: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if not is_secret_ref(value):
            continue
        ref = value.strip()
        try:
            os.environ[name] = resolve_secret_ref(ref)
            resolved.append(name)
            logger.info("Resolved %s from Secret Manager (%s).", name, ref)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            logger.error(
                "Could not resolve %s from Secret Manager path %s: %s. "
                "Leaving the reference in place; the consumer will report a "
                "specific error.",
                name,
                ref,
                exc,
            )
    return resolved
