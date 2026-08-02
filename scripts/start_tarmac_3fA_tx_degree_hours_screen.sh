#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_NAME="${SESSION_NAME:-tx_3fA_degree_hours}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v screen >/dev/null 2>&1; then
  echo "GNU screen is not installed. Install it before starting this launcher." >&2
  exit 2
fi

if screen -ls | grep -q "[.]$SESSION_NAME"; then
  echo "screen session '$SESSION_NAME' already exists." >&2
  echo "Resume it with: screen -r $SESSION_NAME" >&2
  exit 2
fi

mkdir -p "$REPO_DIR/experiment_queue_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCREEN_LOG="$REPO_DIR/experiment_queue_logs/${SESSION_NAME}_${STAMP}.screen.log"
RUN_SCRIPT="$REPO_DIR/scripts/run_tarmac_3fA_tx_degree_hours_3seeds_parallel.sh"

screen -dmS "$SESSION_NAME" bash -lc \
  "set -o pipefail; cd '$REPO_DIR' && bash '$RUN_SCRIPT' 2>&1 | tee '$SCREEN_LOG'"

echo "Started detached screen session: $SESSION_NAME"
echo "All three seeds run concurrently; each seed is pinned to two CPU cores."
echo "Resume: screen -r $SESSION_NAME"
echo "Detach again: Ctrl-a then d"
echo "List sessions: screen -ls"
echo "Top-level log: $SCREEN_LOG"
