#!/bin/bash
# Fresh single-request CUDA memory profile under resource_guard.py, in the foreground.
# Usage: cuda-profile.sh ENV_FILE
# Writes RUN_DIR/cuda-memory.json (and .fit/.preflight/.spec beside it). Profiles
# expire after 24 hours; reprofile before any run on a later day.
ENV_FILE=${1:-}
. "$(dirname "$0")/lib.sh"
require RUN_DIR SOURCE_DIR CUDA_BUILD_DIR CUDA_MODEL_ROOT CUDA_LD_LIBRARY_PATH \
  CUDA_HOST_LIMIT_BYTES CUDA_DEVICE_LIMIT_BYTES MANIFEST PLAN WORKLOAD PROFILE_CYCLES \
  PROFILE_GUARD_SECONDS CONCURRENT_LOAD
LOG_FILE=$RUN_DIR/cuda-profile.driver.log

cd "$SOURCE_DIR"
PY="$SOURCE_DIR/.venv/bin/python"
[ ! -e "$RUN_DIR/cuda-memory.json" ] || { log "cuda-memory.json exists; move it aside"; exit 1; }
plan_path=$(under "$RUN_DIR" "${PLAN:?}")
stage=$("$PY" -c 'import json, sys
print(next(s["stage_index"] for s in json.load(open(sys.argv[1]))["stages"] if s["worker_id"] == "cuda"))' "$plan_path")
identity="$RUN_DIR/source-identity-cuda-profile-$(date -u +%Y%m%dT%H%M%SZ).json"
source_digest=$("$PY" scripts/validation/source_identity.py "$SOURCE_DIR" "$identity" \
  "$CUDA_BUILD_DIR/cpp/hllm-worker-cuda" "$CUDA_BUILD_DIR/cpp/hllm-profile-memory-cuda" | tail -n 1)
log "source digest $source_digest; CUDA stage $stage"
drop_page_cache
{
  echo "# prelaunch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  free -b | head -2
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv
  uptime
} > "$RUN_DIR/cuda-profile.prelaunch.txt"
export LD_LIBRARY_PATH=$CUDA_LD_LIBRARY_PATH
with_conda "$PY" scripts/validation/resource_guard.py \
  --seconds "$PROFILE_GUARD_SECONDS" --record "$RUN_DIR/cuda-profile.guard.jsonl" -- \
  "$SOURCE_DIR/.venv/bin/hllm" profile-memory \
  --manifest "$(under "$RUN_DIR" "$MANIFEST")" --plan "$plan_path" \
  --workload "$(under "$RUN_DIR" "$WORKLOAD")" --stage-index "$stage" \
  --model-root "$CUDA_MODEL_ROOT" \
  --binary "$CUDA_BUILD_DIR/cpp/hllm-profile-memory-cuda" --backend cuda \
  --host-bytes "$CUDA_HOST_LIMIT_BYTES" --device-bytes "$CUDA_DEVICE_LIMIT_BYTES" \
  --cycles "$PROFILE_CYCLES" --timeout $((PROFILE_GUARD_SECONDS - 100)) \
  --source-revision "$source_digest" --concurrent-load "$CONCURRENT_LOAD" \
  --output "$RUN_DIR/cuda-memory.json" 2>&1 | tee -a "$LOG_FILE"
