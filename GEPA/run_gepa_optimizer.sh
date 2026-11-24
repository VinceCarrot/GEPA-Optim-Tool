#!/usr/bin/env bash
# Runs the full GEPA optimization loop in the already-activated optim venv,
# rebuilds train/dev/test splits, then runs optimizer.py and renders the tree.
#
# Usage:
#   ./run_gepa_optimizer.sh
#
# Note: this makes real, potentially many, calls to the task and reflection
# LLMs (config.TASK_MODEL_ID / config.REFLECTION_MODEL_ID) per config.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${VIRTUAL_ENV:-}" != "$HOME/.venvs/optim" ]]; then
	echo "Activate the optim venv first: source ~/.venvs/optim/bin/activate" >&2
	exit 1
fi

PYTHON_BIN="$VIRTUAL_ENV/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python executable not found in active venv: $PYTHON_BIN" >&2
	exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
if [[ "$PYTHON_VERSION" != "3.13.14" ]]; then
	echo "Expected Python 3.13.14 in optim venv, got $PYTHON_VERSION" >&2
	exit 1
fi

mkdir -p logs
LOG_FILE="logs/optimizer_run_$(date +%Y%m%d_%H%M%S).log"

echo "== GEPA optimizer run started at $(date) ==" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_BIN ($PYTHON_VERSION)" | tee -a "$LOG_FILE"
echo "Log file:  $LOG_FILE" | tee -a "$LOG_FILE"

echo "-- Building/refreshing train/dev/test splits --" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -u dataset.py 2>&1 | tee -a "$LOG_FILE"

echo "-- Running optimizer.py --" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -u optimizer.py 2>&1 | tee -a "$LOG_FILE"

echo "-- Rendering candidate lineage tree --" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -u plot_candidate_tree.py 2>&1 | tee -a "$LOG_FILE"

echo "== GEPA optimizer run finished at $(date) ==" | tee -a "$LOG_FILE"