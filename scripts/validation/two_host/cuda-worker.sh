#!/bin/bash
# Run the CUDA worker under resource_guard.py on the Linux host, in the foreground.
# Usage: cuda-worker.sh ENV_FILE baseline|concurrent
# The guard is exec'd, so this script's PID is the guard's PID: signal it to retire
# the worker and close the guard record with its exit event.
ENV_FILE=${1:-}
ROLE=${2:-}
. "$(dirname "$0")/lib.sh"
require RUN_DIR SOURCE_DIR CUDA_BUILD_DIR CUDA_MODEL_ROOT CUDA_LD_LIBRARY_PATH CUDA_PORT \
  CUDA_HOST_LIMIT_BYTES CUDA_DEVICE_LIMIT_BYTES MAX_ACTIVE_REQUESTS MAX_CACHED_TOKENS GUARD_SECONDS
case $ROLE in baseline | concurrent) ;; *) echo "role must be baseline or concurrent" >&2; exit 2 ;; esac

cd "$SOURCE_DIR"
export LD_LIBRARY_PATH=$CUDA_LD_LIBRARY_PATH
exec "$SOURCE_DIR/.venv/bin/python" scripts/validation/resource_guard.py \
  --seconds "$GUARD_SECONDS" --record "$RUN_DIR/$ROLE-cuda.guard.jsonl" -- \
  "$CUDA_BUILD_DIR/cpp/hllm-worker-cuda" --listen "127.0.0.1:$CUDA_PORT" --worker-id cuda \
  --model-root "$CUDA_MODEL_ROOT" \
  --memory-limit-bytes "$CUDA_HOST_LIMIT_BYTES" \
  --device-memory-limit-bytes "$CUDA_DEVICE_LIMIT_BYTES" \
  --max-active-requests "$MAX_ACTIVE_REQUESTS" --max-cached-tokens "$MAX_CACHED_TOKENS" \
  --request-observations on
