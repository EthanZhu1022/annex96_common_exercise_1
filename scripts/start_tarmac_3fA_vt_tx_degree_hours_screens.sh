#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v screen >/dev/null 2>&1; then
  echo "GNU screen is not installed." >&2
  exit 2
fi
if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is not installed." >&2
  exit 2
fi

if [[ -z "${PYTHON_EXE:-}" ]]; then
  if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON_EXE="$REPO_DIR/.venv/bin/python"
  else
    PYTHON_EXE="python3"
  fi
fi

AVAILABLE_CORES="$("$PYTHON_EXE" -c 'import os; print(len(os.sched_getaffinity(0)))')"
if ((AVAILABLE_CORES < 12)); then
  echo "Running VT and TX together requires 12 allowed CPU cores; found $AVAILABLE_CORES." >&2
  echo "Start one climate at a time if this machine has fewer than 12 allowed cores." >&2
  exit 2
fi

echo "Starting TX on the first 6 allowed CPU cores..."
CPU_OFFSET=0 SESSION_NAME=tx_3fA_comfort15_degree15 \
  bash "$SCRIPT_DIR/start_tarmac_3fA_tx_degree_hours_screen.sh"

echo "Starting VT on the next 6 allowed CPU cores..."
CPU_OFFSET=6 SESSION_NAME=vt_3fA_comfort15_degree15 \
  bash "$SCRIPT_DIR/start_tarmac_3fA_vt_degree_hours_screen.sh"

echo "VT and TX are running in separate detached screen sessions."
echo "Inspect them with: screen -ls"
