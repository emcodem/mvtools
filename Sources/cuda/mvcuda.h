// C API of the optional CUDA helper library (mvtools_cuda.dll).
// Loaded at runtime, like avstp.dll: mvtools2.dll does not link against CUDA, and falls
// back to the CPU code when the library or a usable GPU is missing.
// Every function returns 0 on success unless stated otherwise.

#ifndef MVCUDA_H
#define MVCUDA_H

#if defined(MVCUDA_EXPORTS) && defined(_WIN32)
#define MVCUDA_API __declspec(dllexport)
#elif defined(MVCUDA_EXPORTS)
#define MVCUDA_API __attribute__((visibility("default")))
#else
#define MVCUDA_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define MVCUDA_API_VERSION 30

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
  int rfilter;                // reduction filter of the first level: 4 (cubic) or 2 (bilinear filtered, the MSuper default)
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
  unsigned long long src_key, ref_key;    // ignored since API 16 (pictures are device frames, see mvcuda_frame_upload_keyed)
  int search_type;                        // SearchType.h value (1, 2, 4, 8, 16, 32, 64, 128); 0 = exhaustive. search_radius is its parameter
} MvCudaRecalcParams;

// Recalculates the vectors of one frame on the GPU. The super frames (level 0, chroma, pel 1 or 2) are built from the
// raw planes of the source and the reference frame (host memory, as for mvcuda_super_host). old_vectors and
// out_vectors are the (vx, vy, sad) int triples of the old and the new vector plane, block by block.
// Returns like mvcuda_super_host. Test entry point (uploads both pictures); filters use mvcuda_recalc_token.
MVCUDA_API int mvcuda_recalc_host(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], const int *old_vectors,
  int *out_vectors);

// The same with the pictures as device frames (tokens): the super frames are kept with them (made on first use).
// MVCUDA_TOKEN_GONE when a token was evicted (API 16).
MVCUDA_API int mvcuda_recalc_token(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, unsigned long long src_token,
  unsigned long long ref_token, const int *old_vectors, int *out_vectors);

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
  int search_param;                       // exhaustive search radius at the coarse levels (searchparam), 1..64
  int pel_search;                         // search radius at level 0 (pelsearch): 1..64 exhaustive, 1..24 other types
  int chroma_sad_scale;                   // effective chroma SAD scale (>0 shift right, <0 shift left)
  int slice_threads;                      // 1: single-slice CPU result; N: CPU result with N threads (see file)
  unsigned long long src_key, ref_key;    // ignored since API 16 (pictures are device frames, see mvcuda_frame_upload_keyed)
  int search_type;                        // SearchType.h value used at level 0 (and at every level for 64 / 128); 0 = exhaustive.
                                          // Coarse levels use search_type_coarse (radius search_param) for every other type, as on the CPU
  int bad_range;                          // MAnalyse badrange (wide search for blocks above badSAD; used by the meander=false search)
  int try_many;                           // MAnalyse trymany: refine around every predictor at the coarse levels (API 18)
  int search_type_coarse;                 // MAnalyse search_coarse as a SearchType.h value; 0 = exhaustive (the default, API 19)
  int try_many_all;                       // MAnalyse trymanyall: with try_many, the finest level too (if there are 2+ levels, API 19)
} MvCudaAnalyseParams;

// Runs the whole coarse-to-fine search of one frame on the GPU. The super frames (all levels, chroma, pel 1 or 2) are
// built from the raw planes as for mvcuda_super_host. payload receives the vector frame payload exactly as MAnalyse
// writes it: int32 [total words, 1], then per level, coarse to fine, [size, (vx, vy, sad) * blocks]; payload_words is
// its capacity. Returns like mvcuda_super_host, or MVCUDA_NEEDS_CPU. Test entry point (uploads both pictures).
MVCUDA_API int mvcuda_analyse_host(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words);

// The same with the source / reference pictures as device frames (tokens, see "device frames and tokens" below): no
// upload; the super frames are kept with the device frames (made on first use, counted in the registry budget).
// MVCUDA_TOKEN_GONE when a token was evicted (make it again and retry).
MVCUDA_API int mvcuda_analyse_token(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, unsigned long long src_token,
  unsigned long long ref_token, int *payload, int payload_words);

// ---- vectors from NVIDIA's optical flow accelerator (cudafilters' CudaNVOFAnalyse, API 17), see mvcuda_nvof_vec.cu ------
// NOT MAnalyse's search and not bit-exact with any CPU code: the median of NVOFA's flow under each block, at 1/pel pixel,
// clamped to MAnalyse's search area, with its SAD.

typedef struct
{
  int blk_x, blk_y, overlap_x, overlap_y; // block geometry
  int n_blk_x, n_blk_y;                   // number of blocks (MAnalyse's formula)
  int chroma_sad_scale;                   // effective chroma SAD scale (>0 shift right, <0 shift left), as MAnalyse's
  int preset;                             // NVOFA performance level: 1 fast, 2 medium, 3 slow
} MvCudaNvofParams;

// One vector field from the source picture (src_token) to the reference picture (ref_token), device frames of the
// format sp describes. sp: levels 1, pel 1 or 2, the padding of the MSuper used downstream (the vectors are clamped to
// it; the SAD is taken against MSuper(pel, hpad, vpad, sharp=2, levels=1) of the reference, kept with the frame).
// payload receives a one-level MAnalyse payload: [3 + 3N, 1, 1 + 3N, (vx, vy, sad) * N]. MVCUDA_TOKEN_GONE when a token
// was evicted, MVCUDA_NEEDS_CPU when NVOFA is not available or fails, -4 for bad arguments.
MVCUDA_API int mvcuda_nvof_token(const MvCudaSuperParams *sp, const MvCudaNvofParams *np, unsigned long long src_token,
  unsigned long long ref_token, int *payload, int payload_words);

// 1 when NVOFA can be used (device present, nvofapi64.dll of the driver loaded, a session can be made), else 0.
MVCUDA_API int mvcuda_nvof_available(void);

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
  unsigned long long src_key, ref_key;     // ignored since API 16
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

// gives the pooled device memory back to the driver and forgets the upload keys (call when the last user of the library
// goes away)
MVCUDA_API void mvcuda_cache_clear();

// ---- device frames and tokens, see mvcuda_frames.cu ---------------------
// A token stands for one frame kept in device memory. GPU filters return ordinary AviSynth frames whose first bytes hold a
// MvCudaTokenHeader instead of the picture ("token frames"); a consumer finds the device frame by the token. The registry
// has a byte budget (MVCUDA_CACHE_MB fixes it; default 256 MB, grown while frames have to be made again, up to a third of
// the free device memory at first use, at most 4096 MB) and evicts the least recently used frames. A consumer whose token was evicted asks the producer to make the frame again
// (mvcuda_token_regenerate). All plugins of a process must load the same mvtools_cuda.dll (same folder): the registry
// is per library instance.

#define MVCUDA_TOKEN_MAGIC 0x4B5441445543564DULL // "MVCUDATK" read as a little-endian 64-bit integer
#define MVCUDA_TOKEN_LAYOUT 1
#define MVCUDA_TOKEN_GONE -6                     // the token is not (any longer) in the registry

typedef struct
{
  int width, height;          // luma size in samples
  int planes;                 // 1 (Y only) or 3 (Y, U, V)
  int bytes_per_sample;       // 1 or 2
  int bits_per_sample;        // 8..16
  int x_ratio_uv, y_ratio_uv; // chroma subsampling (1 or 2); ignored for planes == 1
} MvCudaFrameFormat;

typedef struct
{
  unsigned long long magic;    // MVCUDA_TOKEN_MAGIC
  int layout;                  // MVCUDA_TOKEN_LAYOUT
  int device;                  // CUDA device index the frame lives on
  unsigned long long token;    // registry key
  unsigned long long producer; // mvcuda_producer_register id of the filter that made it (0: none)
  int frame;                   // frame number at the producer
  int reserved;
  MvCudaFrameFormat format;
} MvCudaTokenHeader;

typedef struct
{
  unsigned long long bytes, budget;          // device bytes held by the registry, byte budget
  unsigned long long frames;                 // frames in the registry
  unsigned long long uploads, downloads, evictions, regenerations, lookups_gone;
  unsigned long long redirects;               // lookups of evicted tokens served by a newer token of the same frame (API 14)
  unsigned long long upload_hits;             // mvcuda_frame_upload_keyed calls served by a frame already on the device (API 16)
} MvCudaFrameStats;

// Copies a host frame (src[plane], pitch in bytes) to the device and registers it. out receives the header to write into
// the token frame. producer / frame: the regeneration source (0 / any if none). Returns 0, -1 no device, -2 CUDA error,
// -4 bad arguments.
MVCUDA_API int mvcuda_frame_upload(const MvCudaFrameFormat *f, const void *const src[3], const int src_pitch_bytes[3],
  unsigned long long producer, int frame, MvCudaTokenHeader *out);

// The same for a picture identified by a content key (key != 0; same key = same pixels, e.g. a clip and a frame number):
// while a frame uploaded under that key (and format) is in the registry, out receives its token and nothing is copied.
// Uploaded frames have no producer: a consumer that finds the token evicted uploads again (API 16).
MVCUDA_API int mvcuda_frame_upload_keyed(const MvCudaFrameFormat *f, const void *const src[3], const int src_pitch_bytes[3],
  unsigned long long key, MvCudaTokenHeader *out);

// Copies the device frame of a token to host memory (dst[plane], pitch in bytes). MVCUDA_TOKEN_GONE when evicted.
MVCUDA_API int mvcuda_frame_download(unsigned long long token, void *const dst[3], const int dst_pitch_bytes[3]);

// Producers that can make a frame again. fn(user, frame, env, out) must return a fresh token for that frame (0 on
// success); env is passed through from mvcuda_token_regenerate (the caller's IScriptEnvironment). Register in the filter's
// constructor, unregister in its destructor.
typedef int (*MvCudaRegenFn)(void *user, int frame, void *env, MvCudaTokenHeader *out);
MVCUDA_API unsigned long long mvcuda_producer_register(MvCudaRegenFn fn, void *user);
MVCUDA_API void mvcuda_producer_unregister(unsigned long long producer);
// -7 when the producer is not registered (any more)
MVCUDA_API int mvcuda_token_regenerate(unsigned long long producer, int frame, void *env, MvCudaTokenHeader *out);

// Keeps the device frame of a token reachable, also after the registry evicted it, until the matching unpin (counted).
// A consumer that regenerates several inputs pins each one right after making it (the frame a thread just made is still
// reachable for that thread) and unpins them after its operation; otherwise making the next input can evict the first
// one again under a small budget. MVCUDA_TOKEN_GONE when the token is already gone.
MVCUDA_API int mvcuda_token_pin(unsigned long long token);
MVCUDA_API void mvcuda_token_unpin(unsigned long long token);

MVCUDA_API void mvcuda_frame_stats(MvCudaFrameStats *s);

// ---- MFlowFps on device frames (see mvcuda_flow.cu and mvcuda_flow_host above) -----------------------------------------
// The same with the left / right pictures as device frames (tokens); the result is a new device frame (the pictures'
// format), out its token (producer / frame for regeneration). MVCUDA_TOKEN_GONE when a token was evicted (API 16).
MVCUDA_API int mvcuda_flow_token(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left,
  unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, unsigned long long producer,
  int frame, MvCudaTokenHeader *out);
// The same with the result copied into host planes (dst[plane], pitch in bytes) instead of a device frame (API 16).
MVCUDA_API int mvcuda_flow_token_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left,
  unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3],
  const int dst_pitch_bytes[3]);

// ---- resampling (AviSynth+ FilteredResizeV / H semantics), see mvcuda_resample.cu ------------------------------------
// A program is the AviSynth integer resampling program of one plane after resize_prepare_coeffs: for each of target_size
// output positions the first source position (offset) and taps coefficients (zero padded), fixed point with 14 bits
// (8-bit data) or 13 bits (10..16-bit data), as the host made it for that bit depth. Kept on the device; 0 on failure.
MVCUDA_API unsigned long long mvcuda_resample_program_create(int target_size, int taps, const int *offset, const short *coeff);
MVCUDA_API void mvcuda_resample_program_free(unsigned long long program);

// Vertical resampling of every plane: output plane p (out_fmt) is prog[p] applied to the source rows row_start[p] +
// k * row_step[p] (k = 0, 1, ...) of plane p of the source token (row_step 2 reads one field). Integer arithmetic and
// clamping as AviSynth's C resampler. out receives the new token (producer / frame for regeneration).
MVCUDA_API int mvcuda_resample_v(const unsigned long long prog[3], unsigned long long src_token, const int row_start[3],
  const int row_step[3], const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// Two-pass resize as FilteredResize::CreateResize builds it: a horizontal pass with prog_h (per plane, columns) and a
// vertical pass with prog_v (rows), horizontal first when h_first. A pass whose prog[0] is 0 is skipped (the CPU skips
// identity passes). The intermediate frame is rounded and clamped like the CPU's.
MVCUDA_API int mvcuda_resample_2d(const unsigned long long prog_h[3], const unsigned long long prog_v[3], int h_first,
  unsigned long long src_token, const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// ---- AviSynth+ TemporalSoften (focus.cpp), see mvcuda_temporal.cu ----------------------------------------------------
// tokens[0 .. 2 * radius]: the frames n - radius .. n + radius (already clamped to the clip), the centre is tokens[radius].
// luma_thresh / chroma_thresh 0..255 as the filter's arguments; scenechange already scaled as the filter's constructor does
// (0 = off). MVCUDA_TOKEN_GONE when one of the inputs was evicted.
MVCUDA_API int mvcuda_temporal_soften(const unsigned long long *tokens, int radius, int luma_thresh, int chroma_thresh,
  int scenechange, unsigned long long producer, int frame, MvCudaTokenHeader *out);
// QTGMC's search-clip binomial (TR0 = 2): MergeLuma(MergeLuma(TemporalSoften(b, 1, ...), TemporalSoften(b, 2, ...), w1), b, w2)
// with both TemporalSoften calls given the same thresholds (merge_chroma: Merge instead of MergeLuma, QTGMC's ChromaMotion).
// tokens[0 .. 4]: b at n - 2 .. n + 2 (clamped to the clip). w1 / inv1, w2 / inv2: the weights as mvcuda_merge takes them
// (weighted mode); tail_y / tail_c: the luma / chroma tail_start of mvcuda_merge.
MVCUDA_API int mvcuda_temporal_binomial(const unsigned long long tokens[5], int luma_thresh, int chroma_thresh, int scenechange,
  int w1, int inv1, int w2, int inv2, int tail_y, int merge_chroma, int tail_c, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// ---- AviSynth+ Merge / MergeLuma / MergeChroma (merge.cpp), see mvcuda_merge.cu ----------------------------------------
// Per plane: mode 0 keeps clip a's plane, 1 weighted merge (weight_i / invweight_i as the SIMD path computes them), 2
// average, 3 clip b's plane. tail_start[p]: first sample of a row that the SIMD code leaves to its scalar tail (16-bit data
// only: the vector body and the tail round differently).
MVCUDA_API int mvcuda_merge(unsigned long long a, unsigned long long b, const int mode[3], int weight_i, int invweight_i,
  const int tail_start[3], unsigned long long producer, int frame, MvCudaTokenHeader *out);

// ---- MaskTools / RgTools building blocks and QTGMC_KeepOnlyBobShimmerFixes, see mvcuda_kobs.cu ------------------------
// 8 bit, and 9..16 bit in 16-bit containers; functions that take a lookup table work up to 10 bit (the table's bit depth
// must be the frames').
// 256 x 256 lookup table of a two-clip function (index x * 256 + y, as mt_lutxy(x, y)); kept on the device, 0 on failure
MVCUDA_API unsigned long long mvcuda_lut2_create(const unsigned char *table);
// the same for bits 8..10 with one table for luma and one for chroma (nullptr: the luma table): (1 << bits) x (1 << bits)
// entries each, index x << bits | y, uint8_t entries for 8 bit, uint16_t above (API 15)
MVCUDA_API unsigned long long mvcuda_lut2_create_bits(const void *luma, const void *chroma, int bits);
MVCUDA_API void mvcuda_lut2_free(unsigned long long lut);

// QTGMC_KeepOnlyBobShimmerFixes(Input, Ref, Rep, Chroma) of QTGMC.avsi on device frames (up to 10 bit). lut1 / lut2: the function's
// two restore expressions as tables. planes_mask: bit p set = plane p is processed (Y always, U / V with Chroma); other
// planes are Input's. Supported Rep: those that need neither RemoveGrain 4 (erosion 2 or 5) nor the square mt_expand
// (over-dilation 3 or more); -4 otherwise.
MVCUDA_API int mvcuda_kobs(unsigned long long input, unsigned long long ref, int rep, int planes_mask, unsigned long long lut1,
  unsigned long long lut2, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// RgTools RemoveGrain on device frames (8..16 bit), per plane: mode -1 or 0 copy (RgTools leaves the plane undefined for -1),
// 12 or 20 as RgTools' SSE4.1 code computes them; -4 for other modes
MVCUDA_API int mvcuda_removegrain(unsigned long long src, const int mode[3], unsigned long long producer, int frame, MvCudaTokenHeader *out);

// One masktools-style operation per plane on device frames a, b, c (same format, 8..16 bit; MVCUDA_MT_LUT up to 10 bit):
// op[p] is one of
#define MVCUDA_MT_COPY_A 0  // a (masktools U / V = 2: copy the first clip)
#define MVCUDA_MT_MIN 1     // mt_logic "min"
#define MVCUDA_MT_MAX 2     // mt_logic "max"
#define MVCUDA_MT_CLAMP 3   // mt_clamp(a, bright b, dark c, overshoot p1, undershoot p2) (p1 / p2 in sample units)
#define MVCUDA_MT_VEXPAND 4 // mt_expand(mode="vertical") of a
#define MVCUDA_MT_VINPAND 5 // mt_inpand(mode="vertical") of a
#define MVCUDA_MT_LUT 6     // mt_lutxy(a, b) through the table lut (mvcuda_lut2_create)
#define MVCUDA_MT_COPY_B 7  // b
MVCUDA_API int mvcuda_mt(const int op[3], unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long lut,
  int p1, int p2, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// QTGMC's binomial smooth, resharp and temporal sharpness limit (QTGMC.avsi "Create basic output" .. sharpLimit1, for
// Rep1 0, no source match / lossless / SVThin) in one step, up to 10 bit (8..15 bit with smode 0), all planes (masktools
// U = V = 3); sovs in sample units (masktools scales mt_clamp's overshoot: 255 = max, else << (bits - 8)):
//   binomial = tr1 0: edi;  1: Merge(deg1, edi, w2);  2: Merge(Merge(deg1, deg2, w1), edi, w2)
//   smode 0: resharp = binomial
//   smode 1: resharp = mt_lutxy(binomial, RemoveGrain(binomial, 12), lut)
//   smode 2: vresharp = Merge(mt_expand(binomial, "vertical"), mt_inpand(binomial, "vertical")), with lut_precise != 0 then
//            mt_lutxy(vresharp, binomial, lut_precise) (Precise); resharp = mt_lutxy(binomial, RemoveGrain(vresharp, 12), lut)
//   nref > 0: result = mt_clamp(resharp, max(refs), min(refs), sovs, sovs) over refs[0 .. nref - 1] (QTGMC: edi, fComp1,
//            bComp1[, fComp3, bComp3]), else resharp
// Weights as mvcuda_merge takes them (weighted mode); RemoveGrain 11 is the same on SSE4.1 CPUs (RgTools 1.2 runs mode 12
// code for it). The frames must have one format.
typedef struct
{
  int tr1, w1, inv1, w2, inv2;
  int smode;
  unsigned long long lut, lut_precise;  // mvcuda_lut2_create tables
  int nref, sovs;
} MvCudaQtgmcSharp;
MVCUDA_API int mvcuda_qtgmc_sharp(const MvCudaQtgmcSharp *p, unsigned long long edi, unsigned long long deg1, unsigned long long deg2,
  const unsigned long long *refs, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// QTGMC's back blend (Sbb), 8..16 bit: thin.mt_makediff(mt_makediff(thin, lossed, U=1,V=1).RemoveGrain(12, -1).GaussResize(...),
// U=2,V=2): the luma of the result is thin - G + half (clamped), G the resize of RemoveGrain 12 of thin - lossed + half,
// through the luma resampling programs prog_h / prog_v (mvcuda_resample_program_create, both passes, horizontal first
// when h_first, as CudaGaussResize); chroma is thin's. With nref > 0 the result is then clamped like mvcuda_qtgmc_sharp
// (all planes).
MVCUDA_API int mvcuda_qtgmc_backblend(unsigned long long thin, unsigned long long lossed, unsigned long long prog_h,
  unsigned long long prog_v, int h_first, const unsigned long long *refs, int nref, int sovs, unsigned long long producer, int frame,
  MvCudaTokenHeader *out);

// ---- NNEDI3 field interpolation, see mvcuda_nnedi3.cu ------------------------------------------------------------------
// Bit for bit what plugins_JPSDR 4.1.0 nnedi3 computes for 8-bit data on CPUs it runs with opt = 6 (AVX2 + FMA3, no
// AVX-512): new prescreener (pscrn 2..4) and int16 predictor (fapprox 15). The weights come prepared by the host exactly
// as the plugin prepares them (see cudafilters\CudaNnedi3.cpp), in plain (unshuffled) order.
typedef struct
{
  int xdia, ydia;               // predictor window (nsize 0..6: 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, 32x4)
  int nns;                      // predictor neurons per half: 16, 32, 64, 128 or 256
  int qual;                     // 1 or 2 predictor networks, averaged
  // prescreener: int16 weights [4][64] (hidden neuron, input = 4 field lines x 16 columns, row-major) and 28 floats:
  // [0..3] scale, [4..7] bias, [8 + 4 * j + i] hidden j -> output i, [24..27] output bias
  const short *pre_w;
  const float *pre_f;
  // predictor network q: int16 weights [2 * nns][xdia * ydia] (softmax neurons first), per neuron scale and bias
  const short *pred_w[2];
  const float *pred_scale[2];
  const float *pred_bias[2];
  // the host CPU's own rcpps / rsqrtps results (they are implementation specific): rcp of 1.m indexed by the top 11
  // mantissa bits, rsqrt of 1.m (index 0..1023) and of 2.m (1024..2047) by the top 10 mantissa bits
  const unsigned *rcp_tab;
  const unsigned *rsq_tab;
} MvCudaNnediParams;

// Uploads the network; NULL on failure. Free with mvcuda_nnedi3_free.
MVCUDA_API void *mvcuda_nnedi3_create(const MvCudaNnediParams *p);
MVCUDA_API void mvcuda_nnedi3_free(void *net);
// One output frame of 8-bit src: the lines of parity 1 - field_n are copied, the lines of parity field_n are made.
// plane_mask bit p: plane p is interpolated (others copied unchanged); val_min / val_max: output range per plane.
MVCUDA_API int mvcuda_nnedi3(void *net, unsigned long long src, int field_n, int plane_mask, const int val_min[3],
  const int val_max[3], unsigned long long producer, int frame, MvCudaTokenHeader *out);

// ---- MCompensate / MDegrain1..6 on device frames (8 bit, pel 1 or 2, levels 1 super), see mvcuda_comp.cu ---------------
// The pictures are device frames (tokens). The pel-1 super frame is the picture with replicated edges, read through
// clamped coordinates; the pel-2 one (sharp = 2) adds the three Wiener sub-pel planes of the padded picture, made once per
// device frame and kept with it (mvc::frame_subpel). Per-block tables are made by the host with mvtools' own code
// (vectors, usable flags, weights).
#define MVCUDA_MAX_DEGRAIN 6
typedef struct
{
  int blk_x, blk_y, ov_x, ov_y;  // luma block size and overlap
  int n_blk_x, n_blk_y;
  // overlap windows (OverlapWindows::Overlap9Windows: 9 windows of blk_x * blk_y, and of the chroma block size), NULL
  // without overlap
  const short *win_luma;
  const short *win_chroma;
  int pel;                       // 1 or 2; vectors are in units of 1 / pel sample
  int hpad, vpad;                // luma padding of the super clip (the area the pel-2 planes cover)
} MvCudaBlockGeom;

// MCompensate: blocks = 3 ints per block: luma vector x, y (pel units) and 1 = take the reference block (SAD < thSAD), 0 =
// the source block. Areas not covered by blocks (right / bottom) come from fill (the source or the reference frame).
MVCUDA_API int mvcuda_compensate(const MvCudaBlockGeom *g, unsigned long long src, unsigned long long ref, unsigned long long fill,
  const int *blocks, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// MDegrain<level> (level 1..6): ref_b[k] / ref_f[k] the backward / forward reference of radius k + 1. blocks =
// 4 * level + 2 * (1 + 2 * level) ints per block: for each k the backward vector x, y and the forward vector x, y (luma, pel
// units), then the normalised weights WSrc, WRefB[0 .. level - 1], WRefF[0 .. level - 1] for luma and the same for chroma
// (MVDegrainX::use_block_y / use_block_uv + norm_weights). plane_mask bit p: plane p is degrained, else copied from src.
MVCUDA_API int mvcuda_degrain(const MvCudaBlockGeom *g, int level, unsigned long long src, const unsigned long long *ref_b,
  const unsigned long long *ref_f, const int *blocks, int plane_mask, unsigned long long producer, int frame, MvCudaTokenHeader *out);

// ---- SplatFps picture build (mvcuda_splat.cu) ---------------------------------------------------------------------
// Motion-compensated in-between picture from its two source pictures P and N (Laplacian-pyramid splatting; the
// description and the exact CPU path are in splatfps/core/splat_core.h). Samples are float32 in 8-bit units,
// value = (code - 2^(b-1)) / 2^(b-8). The motion behind it: block phase correlation gives candidate vectors, a
// Lucas-Kanade step refines them, and a per-pixel choice by minimum filtered displaced frame difference (the match
// errors) gives a forward and a backward field per source picture, with occlusion handling (the assignment); the
// local fallback and the pair statistic (scene cut, temporal fallback) guard against unreliable vectors. The exact
// path is the deterministic scalar path: float32 in a fixed evaluation order, no contraction, IEEE division and sqrt,
// exp through double; the fast CPU path and CUDA (exact=true) are checked bit for bit against it.

typedef struct
{
  int width, height;  // picture size
  int exact;          // 1: bit-identical with the exact CPU path (slower); 0: fast (FMA, atomic splats in any order)
  const void *config; // splat::Config (splatfps/core/splat_config.h), resolved and checked, or null: the defaults of
                      // the picture size (config_for_size). It sets the ME scale, the borders (internal grid
                      // (width + 2 pad_x) x (height + 2 pad_y), ME grid that / scale), the pyramid and every tuned
                      // scalar. Copied by the entries (a context keeps its own copy).
  int bits;           // source / output samples of a context: bits per sample (8: one byte, 9..16: two bytes)
  int sub_w, sub_h;   // chroma subsampling, log2 (0 0: 4:4:4, 1 0: 4:2:2, 1 1: 4:2:0)
  int chroma_centre;  // 1: subsampled chroma horizontally half way between two luma samples (centre siting);
                      // 0: co-sited with the even luma samples (left siting)
} MvCudaSplatParams;

// Host-memory entry (tests): P, N, out: 3 planes width x height (Y, Cb, Cr 4:4:4, packed); fwd_P / bwd_N: 4 planes
// (vx, vy, err, local-fallback weight) of the ME grid, the forward field of P and the backward field of N; d: phase of
// the output between P (0) and N (1) after the temporal fallback. -1 no device, -2 CUDA error, -4 bad arguments.
MVCUDA_API int mvcuda_splat_build_host(const MvCudaSplatParams *p, const float *P, const float *N, const float *fwd_P,
  const float *bwd_N, float d, float *out);

// Host-memory entries of the stages around the motion vectors (tests; return codes as mvcuda_splat_build_host).
// ME grid: mw = (width + 2 pad_x) / scale, mh = (height + 2 pad_y) / scale (1080p defaults: (width + 32) / 4,
// (height + 40) / 4).
// ME picture of a luma plane (width x height) -> mw x mh; fir 0 = A, 1 = B.
MVCUDA_API int mvcuda_splat_me_picture_host(const MvCudaSplatParams *p, const float *y, int fir, float *out);
// Errors of n candidate vectors plus the zero vector: vecs = n x (vx, vy) planes, errs = n + 1 planes (zero last).
MVCUDA_API int mvcuda_splat_errors_host(const MvCudaSplatParams *p, const float *cur, const float *other, const float *vecs,
  int n, float *errs);
// Local fallback: channel 3 of the fields fv (count fields, each vx, vy planes).
MVCUDA_API int mvcuda_splat_disorder_weight_host(const MvCudaSplatParams *p, const float *fv, int count, float *ch3);
// Temporal fallback: the adjusted phase from the forward field of P and the backward field of N (4 planes each);
// info (or null): m, p, f, g2.
MVCUDA_API int mvcuda_splat_phase_pull_host(const MvCudaSplatParams *p, const float *fP, const float *fN, double d, double strength,
  double *d_adj, double *info);
// Motion estimation (description and exact CPU path: splatfps/core/splat_pcorr.h, splat_motion.h). Vector sets are
// n x (vx, vy) planes of the ME grid; error sets n + 1 planes (zero vector last); fields 4 planes (vx, vy, err, ch3).
// Phase correlation of the pair (B_prev, B_cur) (ME pictures B): the candidate fields (cands, 2 planes each; 29 at the
// 1080p defaults, splat::pcorr_ncand); optional (null): hp of B_cur (the prefiltered picture, 1 plane),
// glob = dx, dy, height, vx, vy of the global candidate.
MVCUDA_API int mvcuda_splat_pcorr_host(const MvCudaSplatParams *p, const float *B_prev, const float *B_cur, float *cands,
  float *hp, float *glob);
// Refinement of n candidates of cur against other; neg: they point from other into cur (turned around).
MVCUDA_API int mvcuda_splat_refine_host(const MvCudaSplatParams *p, const float *cur, const float *other, const float *cands,
  int n, int neg, float *out);
// Assignment: rf / rb n refined candidates, ef / eb their errors, bias n + 1 values; rb = eb = null: forward side
// only (lowest cost, bwd = 0, 0, splat::NO_SIDE_ERROR). Channel 3 of fwd / bwd = 0.
MVCUDA_API int mvcuda_splat_assign_host(const MvCudaSplatParams *p, const float *rf, const float *ef, const float *rb,
  const float *eb, int n, const float *bias, float *fwd, float *bwd);
// Refinement (refine != 0), errors, assignment and local fallback of centre picture c from the A pictures of c-1 (null:
// single side), c, c+1 and the candidates as a source delivers them (toward_b: the backward ones point from c).
MVCUDA_API int mvcuda_splat_fields_host(const MvCudaSplatParams *p, const float *A_prev, const float *A_cur, const float *A_next,
  const float *cand_fwd, const float *cand_bwd, int n, int toward_b, int refine, const float *bias, float *fwd, float *bwd);
// The whole chain from luma (width x height) of c-1 (null: single side), c, c+1 with the phase-correlation source.
// ms (4 floats or null): GPU time of one picture's ingest (ME pictures, spectra), one pair's candidates, the fields of
// one centre picture, and their sum (the cost per source picture).
MVCUDA_API int mvcuda_splat_fields_luma_host(const MvCudaSplatParams *p, const float *y_prev, const float *y_cur,
  const float *y_next, float *fwd, float *bwd, float *ms);

// ---- SplatFps device pipeline: one context per filter instance (one caller at a time). Source pictures are planar
// YUV codes in the format of MvCudaSplatParams (bits, sub_w, sub_h) in host memory; they are converted and analysed
// once on the device and kept while frames near them are made. Output frames have the same format.
MVCUDA_API void *mvcuda_splat_ctx_create(const MvCudaSplatParams *p); // null: no device, out of memory
MVCUDA_API void mvcuda_splat_ctx_free(void *ctx);
MVCUDA_API int mvcuda_splat_ctx_has_source(void *ctx, int q);         // 1 when source picture q is on the device
MVCUDA_API int mvcuda_splat_ctx_source(void *ctx, int q, const void *const planes[3], const int pitch_bytes[3]); // returns once staged
// The in-between picture at phase d (0..1) between source pictures p and p + 1, as codes into host planes.
// Needs the sources p - 1 (when p > 0) .. p + 2 (-4 when one is missing). d_adj (or null): the phase after the temporal
// fallback (strength: Config::pull_strength, 0 = off by default; at a scene cut it is 0 or 1).
MVCUDA_API int mvcuda_splat_ctx_frame(void *ctx, int p, double d, double strength, void *const out[3],
  const int out_pitch_bytes[3], double *d_adj);
// The same in two halves, so the caller can stage the next source picture while the device builds this frame:
// frame_start queues the frame (returns after the temporal-fallback formula), frame_finish waits for the oldest started
// frame and writes it (out null: dropped). mvcuda_splat_ctx_source may be called in between, and a second frame may be
// started before the first is finished (the device builds it while the caller hands on the first; a third start drops
// the oldest); frame_finish without a started frame returns -4.
MVCUDA_API int mvcuda_splat_ctx_frame_start(void *ctx, int p, double d, double strength, double *d_adj);
MVCUDA_API int mvcuda_splat_ctx_frame_finish(void *ctx, void *const out[3], const int out_pitch_bytes[3]);
// External vector sources (candidate clips): after mvcuda_splat_ctx_external (before the first source) the context runs
// no phase correlation; the fields of picture c come from mvcuda_splat_ctx_candidates, which needs the sources c-1
// (when bwd is given) .. c+1: fwd = 2n planes of the ME grid (vx, vy per candidate, from c into c+1), bwd = 2n planes or
// null (single side; bwd_toward: from c into c-1, else from c-1 into c), refine: run the refinement on them, bias =
// n + 1 per-candidate costs (zero vector last); n = 1..31. frame_start returns -4 while a needed field is missing.
MVCUDA_API int mvcuda_splat_ctx_external(void *ctx);
MVCUDA_API int mvcuda_splat_ctx_has_fields(void *ctx, int c); // 1 when the fields of picture c are on the device
MVCUDA_API int mvcuda_splat_ctx_candidates(void *ctx, int c, const float *fwd, int fwd_pitch_bytes, const float *bwd,
  int bwd_pitch_bytes, int n, int bwd_toward, int refine, const float *bias);

#ifdef __cplusplus
}
#endif

#endif
