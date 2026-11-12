#pragma once
#include <string>

namespace cudaconvolution {

struct CudaStatus {
  bool ok = false;
  std::string error;  // speaking message when !ok: what is missing, what was found, what to do
  std::string info;   // "NVIDIA GeForce RTX 5050 (sm_120), driver 610.88 (CUDA 13.2), runtime 13.3"
};

// Full runtime dependency check for the GPU engine, cached per device (first call does the work):
// NVIDIA driver present (nvcuda.dll), driver new enough for the CUDA runtime linked into the DLL, device exists,
// compute capability supported, a kernel of this build actually runs on the device, device memory can be allocated.
const CudaStatus& cuda_status(int device);

}  // namespace cudaconvolution
