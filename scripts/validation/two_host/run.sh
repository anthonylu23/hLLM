#!/bin/bash
# Launch one guarded two-host experiment from the Linux host and clean up after it.
# Usage: run.sh ENV_FILE baseline|concurrent
#
# baseline:   single-request reload soak (reload_soak.py) on the current binaries; the
#             concurrency harness binds its report, so rebuilt binaries need a new one.
# concurrent: concurrency-two experiment (concurrent_soak.py) against that baseline.
#
# Sequence: refuse busy ports or running workers on either host; copy the settings to
# the Mac; drop the Linux page cache; start the tunnel, the guarded CUDA worker and
# (concurrent) the device watcher bound to the PID the worker reports; start the
# guarded MLX worker on the Mac and mirror its guard record; run the soak. Each
# launch fails at once, with the log tail, if its process dies before listening.
# Cleanup stops the watcher and both guards and waits for them to exit, so every
# guard record ends with its exit event, copies the final Mac records beside (not over)
# the mirror the soak read, closes the tunnel, and only then assembles the results.
# Inputs: RUN_DIR/inputs/ROLE-workers.json (worker specifications, see the README),
# today's profiles (profile.sh), and for concurrent RUN_DIR/baseline-reload-soak.json.
ENV_FILE=${1:-}
ROLE=${2:-}
. "$(dirname "$0")/lib.sh"
require RUN_DIR SOURCE_DIR MAC_SSH MAC_REPO MAC_RUN_DIR MLX_PORT CUDA_PORT MANIFEST PLAN \
  REFERENCE TIMEOUT IDLE_SECONDS
case $ROLE in
  baseline) require BASELINE_CYCLES BASELINE_REQUESTS; OUTPUT=$RUN_DIR/baseline-reload-soak.json ;;
  concurrent) require CYCLES PAIRS TRANSPORT GUARD_DESCRIPTION; OUTPUT=$RUN_DIR/concurrent.json ;;
  *) echo "role must be baseline or concurrent" >&2; exit 2 ;;
esac
LOG_FILE=$RUN_DIR/run-$ROLE.log
MAC_ENV=$MAC_RUN_DIR/two-host.env
WORKERS=$RUN_DIR/inputs/$ROLE-workers.json
PY=$SOURCE_DIR/.venv/bin/python
cd "$SOURCE_DIR"

# ------------------------------------------------------------------ preconditions
[ ! -e "$OUTPUT" ] || { log "$OUTPUT exists; use a new run directory or move it aside"; exit 1; }
[ -f "$WORKERS" ] || { log "missing $WORKERS"; exit 1; }
# The soak reads guard and sampler files at the paths this script writes.
"$PY" - "$WORKERS" "$RUN_DIR" "$ROLE" <<'EOF'
import json, sys
workers, run, role = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
expected = {"cuda": {}, "mlx": {}}
if role == "concurrent":
    expected = {
        "cuda": {
            "guard_record": f"{run}/concurrent-cuda.guard.jsonl",
            "device_samples": f"{run}/concurrent-cuda-device-samples.jsonl",
        },
        "mlx": {"guard_record": f"{run}/mac/concurrent-mlx.guard.jsonl"},
    }
problems = [
    f"{k}.{field} must be {path}"
    for k, fields in expected.items()
    for field, path in fields.items()
    if workers.get(k, {}).get(field) != path
]
if set(workers) != {"cuda", "mlx"}:
    problems.append("workers must be exactly cuda and mlx")
if problems:
    sys.exit("worker specifications: " + "; ".join(problems))
EOF
if [ "$ROLE" = concurrent ]; then
  [ -f "$RUN_DIR/baseline-reload-soak.json" ] || { log "run the baseline first"; exit 1; }
fi
for port in "$MLX_PORT" "$CUDA_PORT"; do
  if ss -ltnH "sport = :$port" | grep -q .; then log "port $port is in use locally"; exit 1; fi
done
if remote "lsof -nP -iTCP:$MLX_PORT -iTCP:$CUDA_PORT -sTCP:LISTEN" >/dev/null; then
  log "a worker port is in use on the Mac"
  exit 1
fi
# Any port: a worker or probe elsewhere would still contaminate the memory evidence.
if pgrep -f "$HLLM_PROCESS_PATTERN" >/dev/null; then
  log "an hLLM worker or probe is already running here; this run only uses workers it starts"
  exit 1
fi
mac_check=0
remote "pgrep -f $(quote "$HLLM_PROCESS_PATTERN")" >/dev/null || mac_check=$?
case $mac_check in
  0) log "an hLLM worker or probe is already running on the Mac"; exit 1 ;;
  1) ;;
  *) log "cannot check the Mac for running workers (status $mac_check)"; exit 1 ;;
esac
identity="$RUN_DIR/inputs/source-identity-$ROLE-$(date -u +%Y%m%dT%H%M%SZ).json"
source_digest=$("$PY" scripts/validation/source_identity.py "$SOURCE_DIR" "$identity" \
  "${CUDA_BUILD_DIR:-build/cuda}/cpp/hllm-worker-cuda" | tail -n 1)
log "source digest $source_digest ($identity)"

# ------------------------------------------------------------------------ cleanup
TUNNEL='' CUDA_GUARD='' WATCHER='' MLX_GUARD='' MIRROR=''
cleanup() {
  set +e
  log "cleanup: stopping owned processes"
  [ -n "$MIRROR" ] && kill "$MIRROR" 2>/dev/null && wait "$MIRROR" 2>/dev/null
  [ -n "$WATCHER" ] && stop_and_wait "$WATCHER" 15
  [ -n "$CUDA_GUARD" ] && stop_and_wait "$CUDA_GUARD" 60
  [ -n "$MLX_GUARD" ] && stop_and_wait_remote "$MLX_GUARD" 60
  if [ -n "$MLX_GUARD" ]; then
    mkdir -p "$RUN_DIR/mac/final"
    scp -q "$MAC_SSH:$MAC_RUN_DIR/$ROLE-mlx.guard.jsonl" "$RUN_DIR/mac/final/" \
      || log "could not copy the final Mac guard record"
    scp -q "$MAC_SSH:$MAC_RUN_DIR/$ROLE-mlx-worker.log" "$RUN_DIR/mac/final/" || true
  fi
  [ -n "$TUNNEL" ] && stop_and_wait "$TUNNEL" 10
  MIRROR='' WATCHER='' CUDA_GUARD='' MLX_GUARD='' TUNNEL=''
  log "GPU compute processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  pgrep -af hllm-worker | tee -a "$LOG_FILE" || log "no local hLLM workers"
  remote "pgrep -af hllm-worker || echo 'no Mac hLLM workers'" | tee -a "$LOG_FILE"
  set -e
}
trap cleanup EXIT

# ------------------------------------------------------------------------- launch
remote "mkdir -p $(quote "$MAC_RUN_DIR")"
scp -q "$ENV_FILE" "$MAC_SSH:$MAC_ENV"
drop_page_cache
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) prelaunch"
  free -b | head -3
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
} >> "$LOG_FILE"

log "tunnel"
nohup "$TWO_HOST_DIR/tunnel.sh" "$ENV_FILE" > "$RUN_DIR/$ROLE-tunnel.log" 2>&1 &
TUNNEL=$!
sleep 3
kill -0 "$TUNNEL" 2>/dev/null || { log "tunnel exited"; cat "$RUN_DIR/$ROLE-tunnel.log" >&2; exit 1; }

log "CUDA worker"
nohup "$TWO_HOST_DIR/cuda-worker.sh" "$ENV_FILE" "$ROLE" > "$RUN_DIR/$ROLE-cuda-worker.log" 2>&1 &
CUDA_GUARD=$!
wait_listen_local "$CUDA_PORT" "$CUDA_GUARD" "$RUN_DIR/$ROLE-cuda-worker.log" 120

if [ "$ROLE" = concurrent ]; then
  log "device watcher"
  nohup "$TWO_HOST_DIR/watch-cuda.sh" "$ENV_FILE" "$RUN_DIR/concurrent-cuda-device-samples.jsonl" \
    > "$RUN_DIR/watch-cuda.log" 2>&1 &
  WATCHER=$!
  sleep 3
  kill -0 "$WATCHER" 2>/dev/null || { log "device watcher exited"; cat "$RUN_DIR/watch-cuda.log" >&2; exit 1; }
fi

log "MLX worker"
mac_log=$MAC_RUN_DIR/$ROLE-mlx-worker.log
MLX_GUARD=$(remote_detach "$mac_log" scripts/validation/two_host/mlx-worker.sh \
  "$(quote "$MAC_ENV")" "$ROLE")
wait_listen_remote "$MLX_PORT" "$MLX_GUARD" "$mac_log" 180
log "MLX worker listening (guard PID $MLX_GUARD on the Mac)"

if [ "$ROLE" = concurrent ]; then
  # The soak reads the Mac guard through this mirror; it snapshots the exact bytes it
  # hashes, so later copies cannot invalidate the record.
  mkdir -p "$RUN_DIR/mac"
  (
    while :; do
      scp -q "$MAC_SSH:$MAC_RUN_DIR/concurrent-mlx.guard.jsonl" \
        "$RUN_DIR/mac/concurrent-mlx.guard.jsonl.part" 2>/dev/null \
        && mv "$RUN_DIR/mac/concurrent-mlx.guard.jsonl.part" "$RUN_DIR/mac/concurrent-mlx.guard.jsonl"
      sleep 4
    done
  ) &
  MIRROR=$!
fi

# --------------------------------------------------------------------------- soak
status=0
if [ "$ROLE" = baseline ]; then
  log "reload soak"
  "$PY" -m scripts.validation.reload_soak \
    --manifest "$(under "$RUN_DIR" "$MANIFEST")" --plan "$(under "$RUN_DIR" "$PLAN")" \
    --reference "$(under "$RUN_DIR" "$REFERENCE")" --workers "$WORKERS" \
    --cycles "$BASELINE_CYCLES" --requests "$BASELINE_REQUESTS" \
    --idle-seconds "$IDLE_SECONDS" --timeout "$TIMEOUT" \
    --output "$OUTPUT" 2>&1 | tee -a "$LOG_FILE" || status=$?
else
  log "concurrency soak"
  "$PY" -m scripts.validation.concurrent_soak \
    --manifest "$(under "$RUN_DIR" "$MANIFEST")" --plan "$(under "$RUN_DIR" "$PLAN")" \
    --reference "$(under "$RUN_DIR" "$REFERENCE")" --workers "$WORKERS" \
    --baseline-report "$RUN_DIR/baseline-reload-soak.json" \
    --source-digest "$source_digest" --transport "$TRANSPORT" \
    --guard-description "$GUARD_DESCRIPTION" \
    --cycles "$CYCLES" --pairs "$PAIRS" --timeout "$TIMEOUT" --idle-seconds "$IDLE_SECONDS" \
    --output "$OUTPUT" 2>&1 | tee -a "$LOG_FILE" || status=$?
fi
log "soak exited with status $status"

cleanup
trap - EXIT
if [ "$ROLE" = concurrent ]; then
  results=$RUN_DIR/results-$(date -u +%Y%m%dT%H%M%SZ).json
  "$PY" scripts/validation/assemble_two_host_results.py "$RUN_DIR" "$results" \
    | tee -a "$LOG_FILE" || log "results assembly failed"
fi
exit "$status"
