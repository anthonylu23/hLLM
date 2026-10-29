# shellcheck shell=bash
# Shared helpers for the two-host launchers. Source after setting ENV_FILE:
#   ENV_FILE=$1; source "$(dirname "$0")/lib.sh"
# Compatible with the bash 3.2 that macOS ships, because the Mac-side scripts use it.

set -euo pipefail

[ -n "${ENV_FILE:-}" ] && [ -f "$ENV_FILE" ] || {
  echo "usage: $0 /path/to/two-host.env ..." >&2
  exit 2
}
# Absolute, because the scripts change directory before copying the file to the Mac or
# handing it to the scripts they start.
ENV_FILE=$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")
# shellcheck disable=SC1090
. "$ENV_FILE"

# shellcheck disable=SC2034 # Used by the scripts that source this file.
TWO_HOST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

require() {
  local name
  for name in "$@"; do
    [ -n "${!name:-}" ] || { echo "$ENV_FILE: $name is not set" >&2; exit 2; }
  done
}

# Absolute path for a value that may be relative to a base directory.
under() { case "$2" in /*) echo "$2" ;; *) echo "$1/$2" ;; esac; }

log() { echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG_FILE:-/dev/null}" >&2; }

remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$MAC_SSH" "$@"; }

# Single-quote a value for a remote shell command line.
quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# Start a script under MAC_REPO on the Mac, detached, and print its PID. The script
# is the only background job (no subshell wraps it), so the PID is the script's own,
# or that of the process it execs. Its redirected stdio releases the SSH session.
# Usage: remote_detach LOG SCRIPT [ARG...], with ARGs already quoted for the remote shell.
remote_detach() {
  local log_path=$1
  shift
  remote "cd $(quote "$MAC_REPO") || exit 1; nohup $* </dev/null >$(quote "$log_path") 2>&1 & echo \$!"
}

# Command-line pattern for hLLM workers and memory probes, and the guards running them.
# It matches the executable path because Linux truncates process names to 15
# characters, so `pgrep -x hllm-worker-cuda` never matches. Used with `pgrep -f`.
# shellcheck disable=SC2034 # Used by the scripts that source this file.
HLLM_PROCESS_PATTERN='(^|/)hllm-(worker|profile-memory)-[a-z]+( |$)'

# Wait for a local listener while the process that should open it stays alive.
# Fails at once, with the log tail, if that process exits first.
wait_listen_local() {
  local port=$1 pid=$2 log_path=$3 seconds=$4 i=0
  while [ "$i" -lt "$seconds" ]; do
    if ss -ltnH "sport = :$port" | grep -q .; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "process $pid exited before listening on port $port; last log lines:"
      tail -n 20 "$log_path" >&2 || true
      return 1
    fi
    sleep 1
    i=$((i + 1))
  done
  log "port $port not listening after ${seconds}s; last log lines:"
  tail -n 20 "$log_path" >&2 || true
  return 1
}

# The same on the Mac: one SSH round trip checks both the process and the port.
wait_listen_remote() {
  local port=$1 pid=$2 log_path=$3 seconds=$4 i=0 status
  while [ "$i" -lt "$seconds" ]; do
    status=0
    remote "kill -0 $pid 2>/dev/null || exit 3; lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null" \
      || status=$?
    case $status in
      0) return 0 ;;
      3)
        log "Mac process $pid exited before listening on port $port; last log lines:"
        remote "tail -n 20 $(quote "$log_path")" >&2 || true
        return 1
        ;;
    esac
    sleep 2
    i=$((i + 2))
  done
  log "Mac port $port not listening after ${seconds}s; last log lines:"
  remote "tail -n 20 $(quote "$log_path")" >&2 || true
  return 1
}

# Signal one process and wait until it is gone, so its final records are complete.
stop_and_wait() {
  local pid=$1 seconds=${2:-30} i=0
  kill -TERM "$pid" 2>/dev/null || return 0
  while kill -0 "$pid" 2>/dev/null && [ "$i" -lt "$seconds" ]; do sleep 1; i=$((i + 1)); done
  if kill -0 "$pid" 2>/dev/null; then
    log "process $pid ignored SIGTERM for ${seconds}s"
    return 1
  fi
}

stop_and_wait_remote() {
  local pid=$1 seconds=${2:-30}
  remote "kill -TERM $pid 2>/dev/null || exit 0; i=0;
    while kill -0 $pid 2>/dev/null && [ \$i -lt $seconds ]; do sleep 1; i=\$((i + 1)); done;
    ! kill -0 $pid 2>/dev/null"
}

drop_page_cache() {
  if [ "${DROP_PAGE_CACHE:-0}" = 1 ]; then
    log "dropping the page cache"
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  fi
}

# Prefix for commands that need the optional conda prefix (gRPC, host compiler).
with_conda() {
  if [ -n "${CONDA_EXE:-}" ]; then
    "$CONDA_EXE" run --no-capture-output -p "$CONDA_PREFIX_DIR" "$@"
  else
    "$@"
  fi
}
