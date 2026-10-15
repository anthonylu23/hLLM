#!/bin/bash
# Build the CUDA worker and probes in SOURCE_DIR (RelWithDebInfo) and print their hashes.
# Usage: build-cuda.sh ENV_FILE
ENV_FILE=${1:-}
. "$(dirname "$0")/lib.sh"
require SOURCE_DIR UV CUDA_BUILD_DIR CUDA_ROOT CUDA_ARCH CUDA_LD_LIBRARY_PATH TORCH_PREFIX \
  CUDA_HOST_COMPILER

cd "$SOURCE_DIR"
export LD_LIBRARY_PATH=$CUDA_LD_LIBRARY_PATH
echo "== uv sync =="
"$UV" sync --locked
echo "== configure =="
with_conda cmake -S . -B "$CUDA_BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DHLLM_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX_DIR:+$CONDA_PREFIX_DIR;}$TORCH_PREFIX" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER" \
  -DTORCH_CUDA_ARCH_LIST="$CUDA_ARCH" -DHLLM_UV_EXECUTABLE="$UV"
echo "== build =="
with_conda cmake --build "$CUDA_BUILD_DIR" -j 4
echo "== BUILD OK =="
sha256sum "$CUDA_BUILD_DIR"/cpp/hllm-worker-cuda "$CUDA_BUILD_DIR"/cpp/hllm-profile-memory-cuda \
  "$CUDA_BUILD_DIR"/cpp/hllm-profile-compute-cuda "$CUDA_BUILD_DIR"/cpp/hllm-profile-link
