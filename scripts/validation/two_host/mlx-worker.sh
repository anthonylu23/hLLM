#!/bin/bash
# Run the MLX worker under resource_guard.py on the Mac, in the foreground.
# Usage (on the Mac): mlx-worker.sh MAC_RUN_DIR/two-host.env baseline|concurrent
# The guard is exec'd, so this script's PID is the guard's PID. `caffeinate -w` holds
# display, idle and system sleep assertions until that process exits: a sleeping Mac
# suspends the experiment and leaves gaps in the guard record.
ENV_FILE=${1:-}
ROLE=${2:-}
. "$(dirname "$0")/lib.sh"
require MAC_REPO MAC_RUN_DIR MLX_BUILD_DIR MLX_MODEL_ROOT FOOTPRINT_HELPER MLX_PORT \
  MLX_LIMIT_BYTES MAX_ACTIVE_REQUESTS MAX_CACHED_TOKENS GUARD_SECONDS
case $ROLE in baseline | concurrent) ;; *) echo "role must be baseline or concurrent" >&2; exit 2 ;; esac

cd "$MAC_REPO"
caffeinate -dis -w $$ &
exec "$MAC_REPO/.venv/bin/python" scripts/validation/resource_guard.py \
  --seconds "$GUARD_SECONDS" --record "$MAC_RUN_DIR/$ROLE-mlx.guard.jsonl" \
  --footprint-helper "$(under "$MAC_REPO" "$FOOTPRINT_HELPER")" -- \
  "$MLX_BUILD_DIR/cpp/hllm-worker-mlx" --listen "127.0.0.1:$MLX_PORT" --worker-id mlx \
  --model-root "$MLX_MODEL_ROOT" --memory-limit-bytes "$MLX_LIMIT_BYTES" \
  --max-active-requests "$MAX_ACTIVE_REQUESTS" --max-cached-tokens "$MAX_CACHED_TOKENS" \
  --request-observations on
