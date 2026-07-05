#!/usr/bin/env bash
# Build script for TradingAgents v0.3.0 + Ollama/qwen3:8b verification.
#
# Stages:
#   install   - install Python dependencies (editable + dev extras)
#   unit      - run the full unit test suite (no live LLM, no network)
#   smoke     - run the live Ollama smoke test (requires Ollama + qwen3:8b)
#
# Usage:
#   ./scripts/build.sh                     # install + unit only
#   ./scripts/build.sh smoke               # install + unit + live smoke
#   OLLAMA_BASE_URL=http://host:11434/v1 \
#     ./scripts/build.sh smoke             # remote Ollama instance
#
# Environment variables:
#   OLLAMA_BASE_URL  - Ollama server endpoint (default: http://localhost:11434/v1)
#   PYTEST_OPTS      - extra pytest flags, e.g. "-x -v"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_SMOKE=false
if [[ "${1:-}" == "smoke" ]]; then
  RUN_SMOKE=true
fi

# ── 1. Install ────────────────────────────────────────────────────────────────
echo "==> Installing dependencies (pip install -e '.[dev]')"
pip install -e ".[dev]" -q

# ── 2. Unit tests ─────────────────────────────────────────────────────────────
echo ""
echo "==> Running unit tests (no live LLM)"
pytest -m unit ${PYTEST_OPTS:-} \
  tests/test_side_correctness.py \
  tests/test_smoke_ollama.py \
  tests/test_structured_agents.py \
  tests/test_signal_processing.py

echo ""
echo "==> Running full test suite"
pytest ${PYTEST_OPTS:-}

# ── 3. Live smoke (optional) ──────────────────────────────────────────────────
if [[ "$RUN_SMOKE" == true ]]; then
  echo ""
  OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
  echo "==> Live smoke test: provider=ollama model=qwen3:8b endpoint=$OLLAMA_URL"

  # Confirm Ollama is reachable before spending time on it.
  if ! curl -sf "${OLLAMA_URL%/v1}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama not reachable at ${OLLAMA_URL%/v1}"
    echo "  Start Ollama: ollama serve"
    echo "  Pull model:   ollama pull qwen3:8b"
    echo "  Or set:       OLLAMA_BASE_URL=http://<host>:11434/v1"
    exit 1
  fi

  # Confirm qwen3:8b is pulled.
  if ! curl -sf "${OLLAMA_URL%/v1}/api/tags" | python3 -c \
      "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; \
       sys.exit(0 if any('qwen3:8b' in m for m in models) else 1)" 2>/dev/null; then
    echo "WARNING: qwen3:8b not found in Ollama model list."
    echo "  Pull it with: ollama pull qwen3:8b"
    echo "  Proceeding anyway — the smoke call will fail if the model is absent."
  fi

  echo ""
  echo "--- smoke output (verbatim) ---"
  OLLAMA_BASE_URL="$OLLAMA_URL" python scripts/smoke_structured_output.py ollama
  echo "--- end smoke output ---"

else
  echo ""
  echo "Skipping live smoke test. To run it:"
  echo "  ollama serve && ollama pull qwen3:8b"
  echo "  OLLAMA_BASE_URL=http://localhost:11434/v1 ./scripts/build.sh smoke"
fi
