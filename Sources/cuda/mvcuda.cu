// Optional CUDA helper library for MVTools. See mvcuda.h.

#include "mvcuda.h"
#include "mvcuda_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(err_)); \
      return -2; \
    } \
  } while (0)

// SAD of every blk x blk block of src against the same block of ref displaced by (dx, dy).
// Ref is read with clamped coordinates. One CUDA block of blk x blk threads per image block.
template <typename pixel_t>
__global__ void sad_blocks_kernel(const pixel_t *src, const pixel_t *ref, int pitch, int width, int height,
  int blk, int dx, int dy, unsigned *out, int nb_x)
{
  const int x = blockIdx.x * blk + threadIdx.x;
  const int y = blockIdx.y * blk + threadIdx.y;
  const int rx = min(max(x + dx, 0), width - 1);
  const int ry = min(max(y + dy, 0), height - 1);

  unsigned v = abs((int)src[y * pitch + x] - (int)ref[ry * pitch + rx]);

  // warp reduction, then one value per warp through shared memory
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, o);

  __shared__ unsigned warp_sum[32];
  const int tid = threadIdx.y * blk + threadIdx.x;
  if ((tid & 31) == 0)
    warp_sum[tid >> 5] = v;
  __syncthreads();

  if (tid == 0)
  {
    unsigned s = 0;
    for (int i = 0; i < (blk * blk + 31) / 32; i++)
      s += warp_sum[i];
    out[blockIdx.y * nb_x + blockIdx.x] = s;
  }
}

extern "C" int mvcuda_api_version()
{
  return MVCUDA_API_VERSION;
}

// number of usable devices: compute capability 8.0 or newer (the kernels need cp.async); 0 = none
extern "C" int mvcuda_device_count()
{
  return mvc::usable_device_count();
}

extern "C" int mvcuda_device_info(int device, char *name, int name_len, int *cc_major, int *cc_minor,
  unsigned long long *total_mem)
{
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, device));
  if (name && name_len > 0)
  {
    strncpy(name, p.name, name_len - 1);
    name[name_len - 1] = 0;
  }
  if (cc_major) *cc_major = p.major;
  if (cc_minor) *cc_minor = p.minor;
  if (total_mem) *total_mem = p.totalGlobalMem;
  return 0;
}

extern "C" int mvcuda_selftest()
{
  if (mvcuda_device_count() <= 0)
    return -1;

  const int width = 256, height = 128, blk = 16, dx = 3, dy = -2;
  const int nb_x = width / blk, nb_y = height / blk;

  std::vector<uint16_t> src(width * height), ref(width * height);
  uint32_t seed = 12345;
  for (size_t i = 0; i < src.size(); i++)
  {
    seed = seed * 1664525u + 1013904223u;
    src[i] = (seed >> 16) & 1023;
    seed = seed * 1664525u + 1013904223u;
    ref[i] = (seed >> 16) & 1023;
  }

  std::vector<unsigned> expected(nb_x * nb_y, 0), got(nb_x * nb_y, 0);
  for (int by = 0; by < nb_y; by++)
    for (int bx = 0; bx < nb_x; bx++)
    {
      unsigned s = 0;
      for (int j = 0; j < blk; j++)
        for (int i = 0; i < blk; i++)
        {
          const int x = bx * blk + i, y = by * blk + j;
          const int rx = std::min(std::max(x + dx, 0), width - 1);
          const int ry = std::min(std::max(y + dy, 0), height - 1);
          s += abs((int)src[y * width + x] - (int)ref[ry * width + rx]);
        }
      expected[by * nb_x + bx] = s;
    }

  uint16_t *d_src = 0, *d_ref = 0;
  unsigned *d_out = 0;
  CUDA_CHECK(cudaMalloc(&d_src, src.size() * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_ref, ref.size() * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_out, got.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref, ref.data(), ref.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));

  sad_blocks_kernel<uint16_t> <<<dim3(nb_x, nb_y), dim3(blk, blk)>>>(d_src, d_ref, width, width, height, blk, dx, dy,
    d_out, nb_x);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_ref);
  cudaFree(d_out);

  return (got == expected) ? 0 : -3;
}

// ---- public entry points: dispatch on the sample type (8-bit -> *_8, 9..16 bit -> *_16) ------------------------------

extern "C" int mvcuda_super_host_8(const MvCudaSuperParams *, const void *const[3], const int[3], void *const[]);
extern "C" int mvcuda_super_host_16(const MvCudaSuperParams *, const void *const[3], const int[3], void *const[]);
extern "C" int mvcuda_recalc_host_8(const MvCudaSuperParams *, const MvCudaRecalcParams *, const void *const[3], const int[3],
  const void *const[3], const int[3], const int *, int *);
extern "C" int mvcuda_recalc_host_16(const MvCudaSuperParams *, const MvCudaRecalcParams *, const void *const[3], const int[3],
  const void *const[3], const int[3], const int *, int *);
extern "C" int mvcuda_analyse_host_8(const MvCudaSuperParams *, const MvCudaAnalyseParams *, const void *const[3], const int[3],
  const void *const[3], const int[3], int *, int);
extern "C" int mvcuda_analyse_host_16(const MvCudaSuperParams *, const MvCudaAnalyseParams *, const void *const[3], const int[3],
  const void *const[3], const int[3], int *, int);

extern "C" int mvcuda_super_plane_dims(const MvCudaSuperParams *p, int level, int plane, MvCudaPlaneDims *d)
{
  if (!p || !d || level < 0 || level >= p->levels || plane < 0 || plane > 2)
    return -4;
  mvc::plane_dims(*p, level, plane, *d);
  return 0;
}

extern "C" int mvcuda_super_host(const MvCudaSuperParams *p, const void *const src[3], const int src_pitch_bytes[3],
  void *const out[])
{
  if (!p)
    return -4;
  return p->bits_per_pixel <= 8 ? mvcuda_super_host_8(p, src, src_pitch_bytes, out)
                                : mvcuda_super_host_16(p, src, src_pitch_bytes, out);
}

extern "C" int mvcuda_recalc_host(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], const int *old_vectors,
  int *out_vectors)
{
  if (!sp)
    return -4;
  return sp->bits_per_pixel <= 8
    ? mvcuda_recalc_host_8(sp, rp, src, src_pitch_bytes, ref, ref_pitch_bytes, old_vectors, out_vectors)
    : mvcuda_recalc_host_16(sp, rp, src, src_pitch_bytes, ref, ref_pitch_bytes, old_vectors, out_vectors);
}

extern "C" int mvcuda_analyse_host(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words)
{
  if (!sp)
    return -4;
  return sp->bits_per_pixel <= 8
    ? mvcuda_analyse_host_8(sp, ap, src, src_pitch_bytes, ref, ref_pitch_bytes, payload, payload_words)
    : mvcuda_analyse_host_16(sp, ap, src, src_pitch_bytes, ref, ref_pitch_bytes, payload, payload_words);
}

extern "C" int mvcuda_flow_host_8(const MvCudaSuperParams *, const MvCudaFlowParams *, const void *const[3], const int[3],
  const void *const[3], const int[3], const int *, const int *, const int *, const int *, void *const[3], const int[3],
  void *const[]);
extern "C" int mvcuda_flow_host_16(const MvCudaSuperParams *, const MvCudaFlowParams *, const void *const[3], const int[3],
  const void *const[3], const int[3], const int *, const int *, const int *, const int *, void *const[3], const int[3],
  void *const[]);

extern "C" int mvcuda_flow_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const void *const src_raw[3],
  const int src_pitch_bytes[3], const void *const ref_raw[3], const int ref_pitch_bytes[3], const int *vec_b,
  const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3], const int dst_pitch_bytes[3],
  void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS])
{
  if (!sp)
    return -4;
  return sp->bits_per_pixel <= 8
    ? mvcuda_flow_host_8(sp, fp, src_raw, src_pitch_bytes, ref_raw, ref_pitch_bytes, vec_b, vec_f, vec_bb, vec_ff, dst, dst_pitch_bytes, dbg)
    : mvcuda_flow_host_16(sp, fp, src_raw, src_pitch_bytes, ref_raw, ref_pitch_bytes, vec_b, vec_f, vec_bb, vec_ff, dst, dst_pitch_bytes, dbg);
}
