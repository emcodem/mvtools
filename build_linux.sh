#!/usr/bin/env bash
# Builds every plugin of this repo with CMake + Ninja (GCC or Clang) and collects the libraries in one folder.
#
#   ./build_linux.sh [build dir]          default: build/linux-x64 next to this script; libraries in <build dir>/bin
#
# Environment (all optional):
#   CMAKE / NINJA         cmake / ninja binaries (default: from the PATH; pip install cmake ninja works without root)
#   NVCC                  nvcc for libmvtools_cuda.so (default: nvcc on the PATH, $CUDA_PATH/bin, /usr/local/cuda/bin);
#                         NVCC=none skips it
#   CUDA_ARCH             e.g. "120-real" (local GPU only, much faster); default: the release list of Sources/cuda
#   GLSLANG               glslangValidator for libmvtools_vulkan.so (default: from the PATH, $VULKAN_SDK/bin); without
#                         root: apt download glslang-tools && dpkg -x glslang-tools_*.deb ~/opt/glslang, then
#                         GLSLANG=~/opt/glslang/usr/bin/glslangValidator; GLSLANG=none skips mvtools_vulkan
#   SPV_PREBUILT          instead of GLSLANG: folder with the compiled shaders of another build (<build>/Sources/vulkan/spv)
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
  nvcc="${NVCC:-$(command -v nvcc || true)}"
  for c in "${CUDA_PATH:-}/bin/nvcc" /usr/local/cuda/bin/nvcc "$HOME/opt/cuda/bin/nvcc"; do
    [ -z "$nvcc" ] && [ -x "$c" ] && nvcc="$c"
  done
  if [ -n "$nvcc" ]; then
    args+=(-DCMAKE_CUDA_COMPILER="$nvcc")
    [ -n "${CUDA_ARCH:-}" ] && args+=(-DMVTOOLS_CUDA_ARCH="$CUDA_ARCH")
  else
    echo "build_linux.sh: nvcc not found, libmvtools_cuda.so is not built (set NVCC)" >&2
    args+=(-DMVTOOLS_CUDA=OFF)
  fi
fi

# Vulkan shaders
if [ "${GLSLANG:-}" = "none" ]; then
  args+=(-DMVTOOLS_VULKAN=OFF)
elif [ -n "${SPV_PREBUILT:-}" ]; then
  args+=(-DMVTOOLS_GLSLANG=MVTOOLS_GLSLANG-NOTFOUND -DMVTOOLS_SPV_PREBUILT="$SPV_PREBUILT")
else
  glslang="${GLSLANG:-$(command -v glslangValidator || command -v glslang || true)}"
  for c in "${VULKAN_SDK:-}/bin/glslangValidator" "$HOME/opt/glslang/usr/bin/glslangValidator"; do
    [ -z "$glslang" ] && [ -x "$c" ] && glslang="$c"
  done
  if [ -n "$glslang" ]; then
    args+=(-DMVTOOLS_GLSLANG="$glslang")
  else
    echo "build_linux.sh: glslangValidator not found, libmvtools_vulkan.so is not built (set GLSLANG or SPV_PREBUILT)" >&2
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
