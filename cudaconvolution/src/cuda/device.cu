// Runtime dependency check of the GPU engine. Kernels follow in phase 3.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cuda_runtime.h>

#include <map>
#include <mutex>

#include "cuda/device.h"

namespace cudaconvolution {
namespace {

// oldest architecture of the fatbin and its list (CMakeLists.txt, CUDACONVOLUTION_CUDA_ARCH)
#ifndef CUDACONVOLUTION_MIN_CC
#error "CUDACONVOLUTION_MIN_CC is set by CMakeLists.txt"
#endif
constexpr int kMinComputeCapability = CUDACONVOLUTION_MIN_CC;
constexpr const char* kBuildArchs = CUDACONVOLUTION_BUILD_ARCHS;
constexpr const char* kHint = " Use CudaConvolution(..., cpu=true) to run on the CPU instead (identical output, slower).";

// Oldest driver branch for the CUDA major release the DLL is built with (minor version compatibility).
constexpr const char* kMinDriver = CUDART_VERSION >= 13000 ? "R580" : "R525";

std::string cuda_ver(int v) { return std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10); }

// NVIDIA driver release ("610.88") from the file version of nvcuda.dll (32.0.16.1088 -> 16.1088 -> 61088).
std::string driver_release(HMODULE nvcuda) {
  wchar_t path[MAX_PATH];
  if (!GetModuleFileNameW(nvcuda, path, MAX_PATH)) return "unknown";
  DWORD dummy = 0, size = GetFileVersionInfoSizeW(path, &dummy);
  std::string buf(size, '\0');
  VS_FIXEDFILEINFO* fi = nullptr;
  UINT len = 0;
  if (!size || !GetFileVersionInfoW(path, 0, size, buf.data()) ||
      !VerQueryValueW(buf.data(), L"\\", reinterpret_cast<void**>(&fi), &len) || !fi)
    return "unknown";
  const unsigned build = (HIWORD(fi->dwFileVersionLS) % 10) * 10000 + LOWORD(fi->dwFileVersionLS);  // 61088
  return std::to_string(build / 100) + "." + (build % 100 < 10 ? "0" : "") + std::to_string(build % 100);
}

__global__ void probe_kernel(int* out) { *out = 0x5EED; }

CudaStatus check(int device) {
  CudaStatus s;
  const std::string runtime = cuda_ver(CUDART_VERSION);
  const int needed = CUDART_VERSION / 1000 * 1000;  // minor version compatibility: any driver of the same major release

  // 1. driver library: load from System32 only, never from the application or plugin folder
  HMODULE nvcuda = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!nvcuda) {
    s.error = "no NVIDIA driver found (nvcuda.dll missing in System32). Install an NVIDIA driver with CUDA " +
              cuda_ver(needed) + " support (" + kMinDriver + " or newer)." + kHint;
    return s;
  }
  const std::string release = driver_release(nvcuda);

  // 2. driver version (CUDA version the driver supports) vs. runtime linked into this DLL
  using DriverGetVersion = int(__stdcall*)(int*);
  int drv = 0;
  if (auto f = reinterpret_cast<DriverGetVersion>(GetProcAddress(nvcuda, "cuDriverGetVersion")); !f || f(&drv) != 0) {
    s.error = "NVIDIA driver " + release + " is installed but not working (cuDriverGetVersion failed). "
              "Reinstall the NVIDIA driver." + std::string(kHint);
    return s;
  }
  const std::string drvtxt = "driver " + release + " (CUDA " + cuda_ver(drv) + ")";
  if (drv < needed) {
    s.error = "NVIDIA " + drvtxt + " is too old: this plugin is built with CUDA " + runtime + " and needs a driver for CUDA " +
              cuda_ver(needed) + " or newer (" + kMinDriver + "+). Update the NVIDIA driver." + kHint;
    return s;
  }

  // 3. device
  int count = 0;
  if (cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess || count == 0) {
    s.error = "no usable CUDA GPU found (" + std::string(e == cudaSuccess ? "0 devices" : cudaGetErrorString(e)) + ", " +
              drvtxt + ")." + kHint;
    return s;
  }
  if (device < 0 || device >= count) {
    s.error = "device=" + std::to_string(device) + " does not exist: " + std::to_string(count) +
              " CUDA device(s) found, valid indices 0.." + std::to_string(count - 1) + ".";
    return s;
  }
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, device);
  const int cc = p.major * 10 + p.minor;
  s.info = std::string(p.name) + " (sm_" + std::to_string(cc) + "), " + drvtxt + ", runtime " + runtime;
  if (cc < kMinComputeCapability) {
    s.error = std::string(p.name) + " (sm_" + std::to_string(cc) + ") is too old: this build needs compute capability " +
              std::to_string(kMinComputeCapability / 10) + "." + std::to_string(kMinComputeCapability % 10) +
              " or newer (built for " + kBuildArchs + ")." + kHint;
    return s;
  }
  int mode = cudaComputeModeDefault;
  cudaDeviceGetAttribute(&mode, cudaDevAttrComputeMode, device);
  if (mode == cudaComputeModeProhibited) {
    s.error = std::string(p.name) + " is in compute mode 'prohibited' (nvidia-smi -c). Set it to default." + kHint;
    return s;
  }

  // 4. a kernel of this build runs on the device and device memory works
  int* d = nullptr;
  int h = 0;
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = cudaMalloc(&d, sizeof(int));
  if (e == cudaSuccess) {
    probe_kernel<<<1, 1>>>(d);
    e = cudaGetLastError();
  }
  if (e == cudaSuccess) e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  if (d) cudaFree(d);
  if (e == cudaErrorNoKernelImageForDevice || e == cudaErrorUnsupportedPtxVersion || e == cudaErrorInvalidPtx ||
      e == cudaErrorJitCompilerNotFound) {
    s.error = s.info + ": this build has no code for this GPU (built for " + kBuildArchs +
              ") and the driver cannot compile the PTX (" + cudaGetErrorString(e) + "). Update the NVIDIA driver." + kHint;
    return s;
  }
  if (e == cudaErrorMemoryAllocation) {
    s.error = s.info + ": cannot allocate GPU memory (GPU memory full?)." + kHint;
    return s;
  }
  if (e == cudaErrorDevicesUnavailable) {
    s.error = s.info + ": GPU is busy or in exclusive compute mode (used by another process)." + kHint;
    return s;
  }
  if (e != cudaSuccess || h != 0x5EED) {
    s.error = s.info + ": GPU self-test failed (" + std::string(e != cudaSuccess ? cudaGetErrorString(e) : "wrong result") +
              ")." + kHint;
    return s;
  }
  s.ok = true;
  return s;
}

}  // namespace

const CudaStatus& cuda_status(int device) {
  static std::mutex m;
  static std::map<int, CudaStatus> cache;
  std::lock_guard<std::mutex> lock(m);
  auto it = cache.find(device);
  if (it == cache.end()) it = cache.emplace(device, check(device)).first;
  return it->second;
}

}  // namespace cudaconvolution
