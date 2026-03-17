#!/usr/bin/env bash
# Integration test runner for Alexandria.
#
# Starts isolated Qdrant and Ollama instances on non-default ports using
# temporary storage directories, runs pytest, then tears everything down.
#
# Intended to be run from within the Nix dev shell:
#   nix develop --command bash scripts/integration-test.sh
#
# Or if you are already inside the dev shell:
#   bash scripts/integration-test.sh
#
# Environment variables (all have safe defaults):
#   QDRANT_TEST_HTTP_PORT   (default: 16333)
#   QDRANT_TEST_GRPC_PORT   (default: 16334)
#   OLLAMA_TEST_PORT        (default: 14434)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Ports (non-default to avoid collisions) ---
QDRANT_HTTP_PORT="${QDRANT_TEST_HTTP_PORT:-16333}"
QDRANT_GRPC_PORT="${QDRANT_TEST_GRPC_PORT:-16334}"
OLLAMA_PORT="${OLLAMA_TEST_PORT:-14434}"

QDRANT_URL="http://localhost:${QDRANT_HTTP_PORT}"
OLLAMA_HOST="http://localhost:${OLLAMA_PORT}"

# --- Temp directories ---
TMPDIR_BASE="$(mktemp -d "${TMPDIR:-/tmp}/alexandria-test.XXXXXX")"
QDRANT_STORAGE="${TMPDIR_BASE}/qdrant-storage"
QDRANT_SNAPSHOTS="${TMPDIR_BASE}/qdrant-snapshots"
OLLAMA_MODELS="${TMPDIR_BASE}/ollama-models"
mkdir -p "${QDRANT_STORAGE}" "${QDRANT_SNAPSHOTS}" "${OLLAMA_MODELS}"

# PIDs to kill on exit
QDRANT_PID=""
OLLAMA_PID=""

cleanup() {
    echo ""
    echo "--- Cleaning up ---"
    [ -n "${QDRANT_PID}" ] && kill "${QDRANT_PID}" 2>/dev/null && echo "Stopped Qdrant (pid ${QDRANT_PID})"
    [ -n "${OLLAMA_PID}" ] && kill "${OLLAMA_PID}" 2>/dev/null && echo "Stopped Ollama (pid ${OLLAMA_PID})"
    # Give processes a moment to exit, then force-kill
    sleep 1
    [ -n "${QDRANT_PID}" ] && kill -9 "${QDRANT_PID}" 2>/dev/null || true
    [ -n "${OLLAMA_PID}" ] && kill -9 "${OLLAMA_PID}" 2>/dev/null || true
    rm -rf "${TMPDIR_BASE}"
    echo "Removed temp dir ${TMPDIR_BASE}"
}
trap cleanup EXIT

# --- Verify binaries are available ---
for cmd in qdrant ollama pytest; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "Error: '${cmd}' not found. Run this inside the Nix dev shell:"
        echo "  nix develop --command bash scripts/integration-test.sh"
        exit 1
    fi
done

echo "=== Alexandria Integration Tests ==="
echo "  Qdrant:  http://localhost:${QDRANT_HTTP_PORT}  (storage: ${QDRANT_STORAGE})"
echo "  Ollama:  http://localhost:${OLLAMA_PORT}  (models: ${OLLAMA_MODELS})"
echo ""

# --- Start Qdrant ---
echo "Starting Qdrant on port ${QDRANT_HTTP_PORT}..."
QDRANT__SERVICE__HTTP_PORT="${QDRANT_HTTP_PORT}" \
QDRANT__SERVICE__GRPC_PORT="${QDRANT_GRPC_PORT}" \
QDRANT__STORAGE__STORAGE_PATH="${QDRANT_STORAGE}" \
QDRANT__STORAGE__SNAPSHOTS_PATH="${QDRANT_SNAPSHOTS}" \
QDRANT__LOG_LEVEL="WARN" \
QDRANT__TELEMETRY_DISABLED="true" \
    qdrant &>"${TMPDIR_BASE}/qdrant.log" &
QDRANT_PID=$!

# --- Start Ollama ---
echo "Starting Ollama on port ${OLLAMA_PORT}..."
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" \
OLLAMA_MODELS="${OLLAMA_MODELS}" \
    ollama serve &>"${TMPDIR_BASE}/ollama.log" &
OLLAMA_PID=$!

# --- Wait for Qdrant ---
echo -n "Waiting for Qdrant..."
for i in $(seq 1 30); do
    if curl -sf "${QDRANT_URL}/healthz" &>/dev/null; then
        echo " ready (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo " FAILED"
        echo "Qdrant did not start. Log:"
        cat "${TMPDIR_BASE}/qdrant.log"
        exit 1
    fi
    sleep 1
    echo -n "."
done

# --- Wait for Ollama ---
echo -n "Waiting for Ollama..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" &>/dev/null; then
        echo " ready (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo " FAILED"
        echo "Ollama did not start. Log:"
        cat "${TMPDIR_BASE}/ollama.log"
        exit 1
    fi
    sleep 1
    echo -n "."
done

# --- Pull embedding model ---
echo "Pulling nomic-embed-text (this may take a minute on first run)..."
OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" \
    ollama pull nomic-embed-text

echo ""
echo "--- Running tests ---"
echo ""

# --- Run pytest ---
ALEXANDRIA_TEST_QDRANT_URL="${QDRANT_URL}" \
ALEXANDRIA_TEST_OLLAMA_HOST="${OLLAMA_HOST}" \
    pytest "${REPO_ROOT}/tests/" -v --tb=short "$@"

echo ""
echo "=== All tests passed ==="
