#!/bin/bash
# Fresh single-request MLX memory profile under resource_guard.py, on the Mac.
# Usage (on the Mac): mlx-profile.sh MAC_RUN_DIR/two-host.env
# profile.sh starts this over SSH and copies the results back. The profiler's own
# preflight refuses to load when macOS free+inactive memory is below the cap plus
# headroom; freeing memory (quitting apps) is the operator's decision.
# The exit status goes to MAC_RUN_DIR/mlx-profile.exit for profile.sh to propagate:
# the profiler writes its artifact before refusing an incomplete measurement (2) or
# an unsafe physical fit (3), so the artifact alone does not mean success.
ENV_FILE=${1:-}
. "$(dirname "$0")/lib.sh"
require MAC_REPO MAC_RUN_DIR MLX_BUILD_DIR MLX_MODEL_ROOT MLX_LIMIT_BYTES MANIFEST PLAN \
  WORKLOAD PROFILE_CYCLES PROFILE_GUARD_SECONDS CONCURRENT_LOAD
trap 'echo $? > "$MAC_RUN_DIR/mlx-profile.exit"' EXIT

cd "$MAC_REPO"
PY="$MAC_REPO/.venv/bin/python"
[ ! -e "$MAC_RUN_DIR/mlx-memory.json" ] || { echo "mlx-memory.json exists; move it aside" >&2; exit 1; }
plan_path=$(under "$MAC_RUN_DIR" "${PLAN:?}")
stage=$("$PY" -c 'import json, sys
print(next(s["stage_index"] for s in json.load(open(sys.argv[1]))["stages"] if s["worker_id"] == "mlx"))' "$plan_path")
source_digest=$("$PY" scripts/validation/source_identity.py "$MAC_REPO" \
  "$MAC_RUN_DIR/source-identity-mlx-profile-$(date -u +%Y%m%dT%H%M%SZ).json" \
  "$MLX_BUILD_DIR/cpp/hllm-worker-mlx" "$MLX_BUILD_DIR/cpp/hllm-profile-memory-mlx" | tail -n 1)
{
  echo "# prelaunch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  vm_stat
  sysctl vm.swapusage
} > "$MAC_RUN_DIR/mlx-profile.prelaunch.txt"
caffeinate -dis -w $$ &
"$PY" scripts/validation/resource_guard.py \
  --seconds "$PROFILE_GUARD_SECONDS" --record "$MAC_RUN_DIR/mlx-profile.guard.jsonl" -- \
  "$MAC_REPO/.venv/bin/hllm" profile-memory \
  --manifest "$(under "$MAC_RUN_DIR" "$MANIFEST")" --plan "$plan_path" \
  --workload "$(under "$MAC_RUN_DIR" "$WORKLOAD")" --stage-index "$stage" \
  --model-root "$MLX_MODEL_ROOT" \
  --binary "$MLX_BUILD_DIR/cpp/hllm-profile-memory-mlx" --backend mlx \
  --unified-bytes "$MLX_LIMIT_BYTES" --cycles "$PROFILE_CYCLES" \
  --timeout $((PROFILE_GUARD_SECONDS - 100)) \
  --source-revision "$source_digest" --concurrent-load "$CONCURRENT_LOAD" \
  --output "$MAC_RUN_DIR/mlx-memory.json"
