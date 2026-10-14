#!/bin/bash
# Sample the CUDA worker's device memory, bound to the PID the worker itself reports.
# Usage: watch-cuda.sh ENV_FILE OUTPUT.jsonl
# The PID comes from the worker's control API (QualificationState.process_id), not
# from matching command lines, which also match the guard that launched the worker.
ENV_FILE=${1:-}
OUTPUT=${2:-}
. "$(dirname "$0")/lib.sh"
require SOURCE_DIR CUDA_PORT GUARD_SECONDS
[ -n "$OUTPUT" ] || { echo "usage: $0 ENV_FILE OUTPUT.jsonl" >&2; exit 2; }

cd "$SOURCE_DIR"
PY="$SOURCE_DIR/.venv/bin/python"
pid=$("$PY" scripts/validation/worker_pid.py "127.0.0.1:$CUDA_PORT" \
  --expect-worker-id cuda --wait-seconds 30)
log "CUDA worker reports PID $pid"
exec "$PY" scripts/validation/memory_watch.py "$pid" "$OUTPUT" --cuda --seconds "$GUARD_SECONDS"
