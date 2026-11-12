#!/usr/bin/env bash
# Builds every plugin of this repo with CMake + Ninja (GCC or Clang) and collects the libraries in one folder.
#
#   ./build_linux.sh [build dir]          default: build/linux-x64 next to this script; libraries in <build dir>/bin
#
# Environment (all optional):
#   CMAKE / NINJA         cmake / ninja binaries (default: from the PATH; pip install cmake ninja works without root)
#   NVCC                  nvcc for libmvtools_cuda.so: CUDA 12.8 .. 12.x only, 12.9 recommended (CUDA 13 has no Pascal /
#                         Volta targets). Default: the first CUDA 12.8+ one of $CUDA_PATH/bin, /usr/local/cuda-12.9,
#                         /usr/local/cuda-12.8, the PATH, /usr/local/cuda, ~/opt/cuda. NVCC=none skips it
#   CUDA_HOST_CXX         host compiler for nvcc when the default g++ is newer than CUDA 12 accepts (12.9: GCC 14 and
#                         older), e.g. CUDA_HOST_CXX=g++-14
#   CUDA_ARCH             e.g. "120-real" (local GPU only, much faster); default: the list of CUDA_ARCH_SET
#   CUDA_ARCH_SET         nightly (default: one machine code per GPU generation, ~30 % less build time) or full (releases)
#   BUILD_TYPE            Release (default), RelWithDebInfo, Debug
#   CMAKE_EXTRA           further CMake arguments, e.g. "-DMVTOOLS_VAPOURSYNTH=OFF"
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:-$SRC/build/linux-x64}"
CMAKE="${CMAKE:-cmake}"
NINJA="${NINJA:-$(command -v ninja || true)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

args=(-S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DMVTOOLS_DIST_DIR="$BUILD/bin")
if [ -n "$NINJA" ]; then
  args+=(-G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA")
fi

# CUDA
if [ "${NVCC:-}" = "none" ]; then
  args+=(-DMVTOOLS_CUDA=OFF)
else
  # CUDA 12.8 .. 12.x only
  cuda12() {
    local rel
    rel="$("$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
    [ -n "$rel" ] && set -- $rel && [ "$1" = 12 ] && [ "$2" -ge 8 ]
  }
  nvcc=""
  if [ -n "${NVCC:-}" ]; then
    cuda12 "$NVCC" || { echo "build_linux.sh: NVCC=$NVCC is not CUDA 12.8 .. 12.x (12.9 recommended)" >&2; exit 1; }
    nvcc="$NVCC"
  else
    for c in "${CUDA_PATH:-}/bin/nvcc" /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-12.8/bin/nvcc "$(command -v nvcc || true)" \
             /usr/local/cuda/bin/nvcc "$HOME/opt/cuda/bin/nvcc"; do
      [ -z "$nvcc" ] && [ -x "$c" ] && cuda12 "$c" && nvcc="$c"
    done
  fi
  if [ -n "$nvcc" ]; then
    args+=(-DCMAKE_CUDA_COMPILER="$nvcc")
    [ -n "${CUDA_HOST_CXX:-}" ] && args+=(-DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_CXX")
    [ -n "${CUDA_ARCH:-}" ] && args+=(-DMVTOOLS_CUDA_ARCH="$CUDA_ARCH")
    args+=(-DMVTOOLS_CUDA_ARCH_SET="${CUDA_ARCH_SET:-nightly}")
  else
    echo "build_linux.sh: no CUDA 12.8 .. 12.x nvcc found, libmvtools_cuda.so is not built (install CUDA 12.9 or set NVCC)" >&2
    args+=(-DMVTOOLS_CUDA=OFF)
  fi
fi

# shellcheck disable=SC2206
[ -n "${CMAKE_EXTRA:-}" ] && args+=(${CMAKE_EXTRA})

echo "$CMAKE ${args[*]}"
"$CMAKE" "${args[@]}"
start=$(date +%s)
"$CMAKE" --build "$BUILD"
echo "Built in $(( $(date +%s) - start )) s. Libraries in $BUILD/bin:"
ls -l "$BUILD/bin"
