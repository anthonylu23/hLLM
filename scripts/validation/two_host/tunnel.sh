#!/bin/bash
# Forward both worker ports over one SSH connection, in the foreground.
# Usage: tunnel.sh ENV_FILE
# Linux reaches the MLX worker at 127.0.0.1:MLX_PORT; the Mac reaches the CUDA worker
# at its own 127.0.0.1:CUDA_PORT.
ENV_FILE=${1:-}
. "$(dirname "$0")/lib.sh"
require MAC_SSH MLX_PORT CUDA_PORT
exec ssh -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -L "$MLX_PORT:127.0.0.1:$MLX_PORT" -R "$CUDA_PORT:127.0.0.1:$CUDA_PORT" -N "$MAC_SSH"
