#!/usr/bin/env bash
# Matching gRPC/Protobuf CMake packages for the Linux CPU CI job.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/../.." && pwd)"
task_deps="$task_root/build/ci-deps"
mkdir -p "$task_deps"
if [[ ! -d "$task_deps/grpc/.git" ]]; then
  git clone --branch v1.74.0 --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/grpc/grpc.git "$task_deps/grpc"
fi
test "$(git -C "$task_deps/grpc" rev-parse HEAD)" = 3e7a4d52d257990fa5b9d80f69f4a591178d9d7c
# The pinned c-ares minimum predates policies still supported by CMake 4.
uv run cmake -S "$task_deps/grpc" -B "$task_deps/compile" -G Ninja \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_INSTALL_PREFIX="$task_deps/install" \
  -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -Dprotobuf_BUILD_TESTS=OFF \
  -DABSL_PROPAGATE_CXX_STD=ON
uv run cmake --build "$task_deps/compile" --parallel 2
uv run cmake --install "$task_deps/compile"
