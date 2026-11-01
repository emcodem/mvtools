// Shared part of the NNEDI3 shaders (nn_pre.pix.comp, nn_pred.pix.comp), port of Sources/cuda/mvcuda_nnedi3.cu: the
// argument struct (same layout as NnArgs in vk_nnedi3.cpp), the CPU's rcpps / rsqrtps from the host's tables and the
// mirrored field access. Include after common.glsl; x.pix.comp defines MVC_PIX8 for 8-bit samples.
//
// Floating point: the CUDA kernels use explicit __f*_rn intrinsics (no contraction) except where the asm fuses
// (__fmaf_rn). Here every float variable is `precise` (SPIR-V NoContraction on every operation feeding it) and the fused
// steps are fma(). No division, sqrt or transcendental function is used (rcpps / rsqrtps come from the tables): only
// add / mul (correctly rounded in Vulkan), fma and conversions. float(int) rounds to nearest even (__int2float_rn),
// int(float) truncates (__float2int_rz), __float2int_rn is int(roundEven()). Vulkan lets fma() be computed unfused; NVIDIA
// and Intel fuse it (random networks, where an unfused e0 would differ, match CUDA on both: test script
// C:/dev/mvtools/profile/vk_nn_test/nn_api.py).
// Denormals: CUDA keeps them (no ftz). Vulkan's default fp32 denormal mode is implementation defined, and the NVIDIA driver
// can request neither DenormPreserve nor FlushToZero for fp32, so the shaders declare no mode. A denormal can only appear
// in the predictor's fma accumulation (exp >= 2^-116 times a tiny elliott value); flushing it moves the weighted mean by
// less than FLT_MIN, which the following (vsum * sd) + mean and the rounding to integer samples absorb.

struct NnArgs
{
  uint64_t src[3], dst[3];
  uint64_t pitch_s[3], pitch_d[3]; // in samples
  uint64_t pre_w, pre_f;           // int16 [4][64], float [28]
  uint64_t pred_w[2];              // int16 [2 * nns][asize] per network
  uint64_t pred_scale[2], pred_bias[2]; // float [2 * nns]
  uint64_t rcp_tab, rsq_tab;       // uint [2048] each
  uint64_t queue, count;           // queued pixels, their number (uint)
  int width[3], height[3], interp[3], vmin[3], vmax[3];
  int off, xdia, ydia, nns, qual, asize;
  float r_asize;
  int pad_;
};
ARGS(NnArgs)

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer NnPV4 { uvec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NnPF { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NnPU { uint v[]; };

// rcpps for positive normal x: mantissa from the table, exponent mirrored (rcp_hw of the CUDA file)
float nn_rcp(uint64_t tab, float x)
{
  const uint u = floatBitsToUint(x);
  const int e = int(u >> 23) - 127;
  const uint r = NnPU(tab).v[(u >> 12) & 2047u];
  return uintBitsToFloat(r - (uint(e) << 23));
}

// rsqrtps for positive normal x: x = y * 4^k with y in [1, 4), result = rsqrt(y) * 2^-k
float nn_rsq(uint64_t tab, float x)
{
  const uint u = floatBitsToUint(x);
  const int e = int(u >> 23) - 127;
  const int par = e & 1;
  const int k = (e - par) >> 1;
  const uint r = NnPU(tab).v[uint(par) * 1024u + ((u >> 13) & 1023u)];
  return uintBitsToFloat(r - (uint(k) << 23));
}

int nn_reflect(int i, int n)
{
  if (i < 0)
    i = -i;
  if (i >= n)
    i = 2 * (n - 1) - i;
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

// sample of field line L (frame row 2 * L + off), column c, both reflected as the plugin's padding
int nn_fld(int p, int off, int L, int c)
{
  L = nn_reflect(L, A.height[p] >> 1);
  c = nn_reflect(c, A.width[p]);
  return ldpix(A.src[p], uint(uint64_t(2 * L + off) * A.pitch_s[p] + uint64_t(c)));
}

// the sample as the network's int16 dot products see it: 8-bit zero-extended, 16-bit as int16
int nn_in(int v)
{
#ifdef MVC_PIX8
  return v;
#else
  return bitfieldExtract(v, 0, 16);
#endif
}
