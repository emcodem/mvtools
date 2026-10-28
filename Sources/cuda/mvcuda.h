// C API of the optional CUDA helper library (mvtools_cuda.dll).
// Loaded at runtime, like avstp.dll: mvtools2.dll does not link against CUDA, and falls
// back to the CPU code when the library or a usable GPU is missing.
// Every function returns 0 on success unless stated otherwise.

#ifndef MVCUDA_H
#define MVCUDA_H

#ifdef MVCUDA_EXPORTS
#define MVCUDA_API __declspec(dllexport)
#else
#define MVCUDA_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define MVCUDA_API_VERSION 4

// version of this interface; the caller must refuse a library that reports another one
MVCUDA_API int mvcuda_api_version();

// number of usable CUDA devices (0 = no GPU or no driver), negative on error
MVCUDA_API int mvcuda_device_count();

MVCUDA_API int mvcuda_device_info(int device, char *name, int name_len, int *cc_major, int *cc_minor,
  unsigned long long *total_mem);

// runs the SAD kernel on device 0 and compares it with a CPU reference
// -1: no device, -2: CUDA error, -3: result differs from the CPU
MVCUDA_API int mvcuda_selftest();


// ---- GPU MSuper (M2), see mvcuda_super.cu ------------------------------------------------------------------
// All planes are packed (no pitch gap) padded planes of 16-bit samples: (w + 2*hpad) x (h + 2*vpad).

#define MVCUDA_MAX_LEVELS 12
#define MVCUDA_MAX_SUB 16

typedef struct
{
  int width, height;          // luma size of the source frame
  int x_ratio_uv, y_ratio_uv; // chroma subsampling (1 or 2); ignored without chroma
  int hpad, vpad;             // luma padding of the super frame (chroma: divided by the ratios)
  int pel;                    // 1 or 2 (4 later)
  int levels;                 // pyramid levels incl. level 0
  int bits_per_pixel;         // 9..16, samples are 16 bit
  int chroma;                 // 0: Y plane only
  int rfilter;                // reduction filter, only 4 (cubic) is implemented
  int sharp;                  // sub-pel filter, only 2 (Wiener) is implemented
} MvCudaSuperParams;

typedef struct
{
  int w, h, hpad, vpad, ew, eh; // ew = w + 2*hpad, eh = h + 2*vpad
} MvCudaPlaneDims;

// geometry of one plane (0 = Y, 1 = U, 2 = V) of one level; -4 for bad arguments
MVCUDA_API int mvcuda_super_plane_dims(const MvCudaSuperParams *p, int level, int plane, MvCudaPlaneDims *d);

// Builds the super frame planes on the GPU from the raw frame in host memory and copies them back.
// src[plane] / src_pitch_bytes[plane]: raw planes (w x h samples). out[(level * 3 + plane) * MVCUDA_MAX_SUB + sub]:
// host buffer of ew*eh samples for that plane and sub-pel index, or NULL to skip it (level > 0 only has sub 0).
// -1 no device, -2 CUDA error, -4 unsupported parameters or bad arguments (caller falls back to the CPU).
MVCUDA_API int mvcuda_super_host(const MvCudaSuperParams *p, const void *const src[3], const int src_pitch_bytes[3],
  void *const out[]);

// ---- GPU MRecalculate (M4), see mvcuda_recalc.cu ------------------------------------------------------------

typedef struct
{
  int blk_x, blk_y, overlap_x, overlap_y; // new block geometry (blocks of the finest level)
  int n_blk_x, n_blk_y;                   // number of new blocks
  int old_blk_x, old_blk_y;               // number of old (MAnalyse) blocks
  int old_size_x, old_size_y, old_overlap_x, old_overlap_y;
  int old_pel;                            // pel of the old vectors
  int lambda_level;                       // vector cost factor for all block rows but the first (CPU: 0 in row 0)
  int pnew;                               // penalty for a new vector (per 256)
  int th_sad;                             // normalised thSAD: blocks whose predictor SAD is <= this are not searched
  int smooth;                             // 1 interpolate the 4 nearest old vectors, 0 nearest
  int search_radius;                      // exhaustive search radius (searchparam)
  int chroma_sad_scale;                   // effective chroma SAD scale (>0 shift right, <0 shift left)
  int slice_threads;                      // 1: CPU single-thread result; N: CPU result with N threads (nLambda is 0 on the first row of every slice)
  unsigned long long src_key, ref_key;    // 0, or an identifier of the frame content (see mvcuda_analyse_host)
} MvCudaRecalcParams;

// Recalculates the vectors of one frame on the GPU. The super frames (level 0, chroma, pel 1 or 2) are built from the
// raw planes of the source and the reference frame (host memory, as for mvcuda_super_host). old_vectors and
// out_vectors are the (vx, vy, sad) int triples of the old and the new vector plane, block by block.
// Returns like mvcuda_super_host.
MVCUDA_API int mvcuda_recalc_host(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], const int *old_vectors,
  int *out_vectors);

// ---- GPU MAnalyse search (M3), see mvcuda_analyse.cu --------------------------------------------------------

// returned when the frame contains blocks that would need the CPU's "bad vector" wide search: use the CPU result
#define MVCUDA_NEEDS_CPU -5

typedef struct
{
  int blk_x, blk_y, overlap_x, overlap_y; // block geometry of level 0
  int n_blk_x, n_blk_y;                   // number of blocks of level 0
  int n_levels;                           // pyramid levels searched (== levels of the super frame)
  int lambda;                             // user lambda (e.g. 16000 for 32x32 truemotion)
  int lsad, pnew, pzero, pglobal, plevel; // as MAnalyse (lsad already scaled by block size and bit depth)
  int use_global;                         // MAnalyse global=true
  int meander;                            // MAnalyse meander=true
  int bad_sad;                            // badSAD, already scaled
  int search_param;                       // exhaustive search radius at the coarse levels (searchparam), 1..3
  int pel_search;                         // exhaustive search radius at level 0 (pelsearch), 1..3
  int chroma_sad_scale;                   // effective chroma SAD scale (>0 shift right, <0 shift left)
  int slice_threads;                      // 1: single-slice CPU result; N: CPU result with N threads (see file)
  unsigned long long src_key, ref_key;    // 0, or an identifier of the frame content: the library keeps built super frames
                                          // by (key, geometry) and reuses them; same key must always mean the same pixels
} MvCudaAnalyseParams;

// Runs the whole coarse-to-fine search of one frame on the GPU. The super frames (all levels, chroma, pel 1 or 2) are
// built from the raw planes as for mvcuda_super_host. payload receives the vector frame payload exactly as MAnalyse
// writes it: int32 [total words, 1], then per level, coarse to fine, [size, (vx, vy, sad) * blocks]; payload_words is
// its capacity. Returns like mvcuda_super_host, or MVCUDA_NEEDS_CPU.
MVCUDA_API int mvcuda_analyse_host(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words);

// ---- GPU MFlowFps (M7), see mvcuda_flow.cu -----------------------------------------------------------------------

#define MVCUDA_FLOW_DEBUG_SLOTS 40

typedef struct
{
  int width, height;                       // frame size (luma)
  int x_ratio_uv, y_ratio_uv;              // chroma subsampling (1 or 2)
  int chroma_distinct;                     // 1: chroma planes have their own vector/mask upsizing (4:2:2, 4:2:0), 0: 4:4:4
  int n_blk_x, n_blk_y;                    // vector blocks
  int n_blk_xp, n_blk_yp;                  // blocks padded to cover the frame
  int blk_step_x, blk_step_y;              // block size - overlap
  int width_p, height_p;                   // luma size of the padded block grid (n_blk_xp * step + overlap)
  int width_p_uv, height_p_uv;             // same for chroma
  int time256;                             // interpolation position, 1..255
  double occ_norm;                         // 10 / ml / pel as MFlowFps computes it
  int mode;                                // 2 (extra frames), the only one implemented
  unsigned long long src_key, ref_key;     // frame cache keys of the left / right frame (0 = uncached)
} MvCudaFlowParams;

// Renders one interpolated frame (MFlowFps mask=2 path) on the GPU. The super frames (level 0, chroma, pel 1 or 2) of the
// left and the right frame come from the frame cache or are built from the raw planes. vec_b / vec_f / vec_bb / vec_ff
// are (vx, vy, sad) int triples per block: backward vectors of the left frame, forward vectors of the right frame
// (MFlowFps names: mvClipB at nleft, mvClipF at nright), forward vectors of the left frame, backward vectors of the right
// frame. dst[plane] / dst_pitch_bytes[plane]: host frame planes to fill. dbg: NULL or MVCUDA_FLOW_DEBUG_SLOTS host
// pointers (entries may be NULL) that receive intermediate arrays for the tests (see mvcuda_flow.cu).
MVCUDA_API int mvcuda_flow_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const void *const src_raw[3],
  const int src_pitch_bytes[3], const void *const ref_raw[3], const int ref_pitch_bytes[3], const int *vec_b,
  const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3], const int dst_pitch_bytes[3],
  void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS]);

// drops all cached super frames and pooled device memory (call when the last user of the library goes away)
MVCUDA_API void mvcuda_cache_clear();

#ifdef __cplusplus
}
#endif

#endif
