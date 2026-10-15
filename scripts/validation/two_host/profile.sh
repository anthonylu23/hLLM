#!/bin/bash
# Produce today's single-request memory profile for one worker.
# Usage: profile.sh ENV_FILE cuda|mlx
# cuda runs cuda-profile.sh here. mlx copies the settings and inputs to the Mac,
# starts mlx-profile.sh there detached (an SSH drop does not kill it), waits for it to
# finish, and copies the profile, its fit, preflight and guard record into RUN_DIR/mac/.
# It exits with the Mac profile's status (2: incomplete measurement, 3: physical fit
# not safe, 124: stopped by the guard), and with 1 unless the copied profile is
# complete with a safe fit.
ENV_FILE=${1:-}
WORKER=${2:-}
. "$(dirname "$0")/lib.sh"
require RUN_DIR SOURCE_DIR MAC_SSH MAC_REPO MAC_RUN_DIR MANIFEST PLAN WORKLOAD
LOG_FILE=$RUN_DIR/$WORKER-profile.driver.log

case $WORKER in
  cuda) exec "$TWO_HOST_DIR/cuda-profile.sh" "$ENV_FILE" ;;
  mlx) ;;
  *) echo "worker must be cuda or mlx" >&2; exit 2 ;;
esac

remote "mkdir -p $(quote "$MAC_RUN_DIR/inputs")"
scp -q "$ENV_FILE" "$MAC_SSH:$MAC_RUN_DIR/two-host.env"
for input in "$MANIFEST" "$PLAN" "$WORKLOAD"; do
  scp -q "$(under "$RUN_DIR" "$input")" "$MAC_SSH:$(under "$MAC_RUN_DIR" "$input")"
done
remote "rm -f $(quote "$MAC_RUN_DIR/mlx-profile.exit")"
pid=$(remote_detach "$MAC_RUN_DIR/mlx-profile.log" scripts/validation/two_host/mlx-profile.sh \
  "$(quote "$MAC_RUN_DIR/two-host.env")")
log "MLX profile running on the Mac as PID $pid"
while :; do
  status=0
  remote "kill -0 $pid 2>/dev/null" || status=$?
  case $status in
    0) sleep 15 ;;
    255) log "SSH to the Mac failed; the profile keeps running there; retrying"; sleep 15 ;;
    *) break ;;
  esac
done
mkdir -p "$RUN_DIR/mac"
for name in mlx-memory.json mlx-memory.fit.json mlx-memory.preflight.json mlx-memory.spec.json \
  mlx-memory.native.jsonl mlx-profile.guard.jsonl mlx-profile.prelaunch.txt mlx-profile.log \
  mlx-profile.exit; do
  scp -q "$MAC_SSH:$MAC_RUN_DIR/$name" "$RUN_DIR/mac/$name" || log "no $name on the Mac"
done
scp -q "$MAC_SSH:$MAC_RUN_DIR/source-identity-mlx-profile-*.json" "$RUN_DIR/mac/" || true
remote "tail -n 3 $(quote "$MAC_RUN_DIR/mlx-profile.log")" | tee -a "$LOG_FILE"
status=$(cat "$RUN_DIR/mac/mlx-profile.exit" 2>/dev/null || true)
case $status in
  0) ;;
  '' | *[!0-9]*) log "the MLX profile left no exit status; see mac/mlx-profile.log"; exit 1 ;;
  *) log "the MLX profile exited with status $status"; exit "$status" ;;
esac
# The exit status already covers this; check the copies too, since later steps read them.
"$SOURCE_DIR/.venv/bin/python" - "$RUN_DIR/mac/mlx-memory.json" "$RUN_DIR/mac/mlx-memory.fit.json" \
  <<'EOF' || { log "the copied MLX profile is not complete with a safe fit"; exit 1; }
import json, sys
artifact, fit = (json.load(open(path)) for path in sys.argv[1:])
if artifact["measurement"].get("completed") is not True:
    sys.exit("measurement incomplete")
if fit["assessment"]["status"] != "safe":
    sys.exit(f"physical fit {fit['assessment']['status']}")
EOF
log "MLX profile complete with a safe fit"
