// CudaNVOFAnalyse(clip, isb, delta, blksize, blksizeV, overlap, overlapV, pel, hpad, vpad, chroma, preset): motion vectors
// from NVIDIA's optical flow accelerator (NVOFA) as a vector clip in MAnalyse's format, for MRecalculate, MFlowFps,
// MDegrain, MCompensate (NVOF_FILTER_PLAN.md). It takes the plain clip: no MSuper pyramid is searched. One level
// (nLvCount 1), vectors at 1/pel pixel, clamped to the area MAnalyse's search reaches with the given padding (hpad / vpad
// must be those of the MSuper used downstream). NOT MAnalyse's search, and not bit-exact with anything on the CPU.
// CudaNVOFAvailable() tells a script whether NVOFA can be used.
//
// Fallback: without NVOFA the filter is MAnalyse(MSuper(c, pel, hpad, vpad, chroma), isb, delta, blksize, blksizeV,
// overlap, overlapV, truemotion=true, chroma) (mvtools2.dll must be loaded); when NVOFA fails on a single frame, that
// frame's vectors are level 0 of that MAnalyse.

#include "CudaCommon.h"

#include "../Sources/MVAnalysisData.h"

#include <cstdint>
#include <mutex>

using namespace cf;

namespace
{
  // same key as mvtools2's GpuBackend::frame_key: one upload per picture of a clip while it stays in the registry
  unsigned long long frame_key(const void *clip, int n)
  {
    unsigned long long k = (unsigned long long)(uintptr_t)clip * 0x9E3779B97F4A7C15ull;
    k ^= ((unsigned long long)(n + 1) * 0xC2B2AE3D27D4EB4Full) + (k << 6) + (k >> 2);
    return k | 1ull;
  }

  PClip make_manalyse(PClip c, bool isb, int delta, int bx, int by, int ox, int oy, int pel, int hpad, int vpad, bool chroma,
    IScriptEnvironment *env)
  {
    if (!env->FunctionExists("MSuper") || !env->FunctionExists("MAnalyse"))
      env->ThrowError("CudaNVOFAnalyse: NVOFA is not available and its fallback needs mvtools2.dll (MSuper / MAnalyse): load it");
    AVSValue sargs[5] = { cpu_input(c, env), pel, hpad, vpad, chroma };
    const char *snames[5] = { nullptr, "pel", "hpad", "vpad", "chroma" };
    PClip sup = env->Invoke("MSuper", AVSValue(sargs, 5), snames).AsClip();
    AVSValue aargs[10] = { sup, isb, delta, bx, by, ox, oy, true, chroma, 0 };
    const char *anames[10] = { nullptr, "isb", "delta", "blksize", "blksizeV", "overlap", "overlapV", "truemotion", "chroma", "scaleCSAD" };
    return env->Invoke("MAnalyse", AVSValue(aargs, 10), anames).AsClip();
  }

  class CudaNVOFAnalyse : public GenericVideoFilter
  {
    MVAnalysisData _ad = {}; // vi's pointer hack points here (MVClip), valid for the filter's lifetime
    PClip _fallback;         // MAnalyse, built on first use
    bool _isb, _chroma;
    int _delta, _pel, _hpad, _vpad;
    int _words = 0;          // payload words (one level)
    int _big_sad = 0;        // SAD of frames without a reference (MAnalyse's verybigSAD)
    MvCudaFrameFormat _fmt = {};
    MvCudaSuperParams _sp = {};
    MvCudaNvofParams _np = {};
    VideoInfo _src_vi;
    std::mutex _fb_m;

    static const int HEADER = 256; // max(4 + sizeof(MVAnalysisData), 256), as MAnalyse

    PClip fallback(IScriptEnvironment *env)
    {
      std::lock_guard<std::mutex> lk(_fb_m);
      if (!_fallback)
        _fallback = make_manalyse(child, _isb, _delta, _ad.nBlkSizeX, _ad.nBlkSizeY, _ad.nOverlapX, _ad.nOverlapY, _pel, _hpad, _vpad,
          _chroma, env);
      return _fallback;
    }

    // the token of frame n of the child: its own token, or the picture uploaded under its key
    int token_of(const PVideoFrame &f, int n, unsigned long long &tok)
    {
      MvCudaTokenHeader h;
      if (read_token(f, _src_vi, h))
      {
        if (!same_format(h.format, _fmt))
          return -4;
        tok = h.token;
        return 0;
      }
      const void *planes[3] = {};
      int pitches[3] = {};
      for (int p = 0; p < _fmt.planes; p++)
      {
        planes[p] = f->GetReadPtr(g_planes[p]);
        pitches[p] = f->GetPitch(g_planes[p]);
      }
      const int rc = CudaLib::get().upload_keyed(&_fmt, planes, pitches, frame_key((const void *)child.operator->(), n), &h);
      if (rc == 0)
        tok = h.token;
      return rc;
    }

    int gpu(int n, int nref, int *payload, IScriptEnvironment *env)
    {
      CudaLib &lib = CudaLib::get();
      PVideoFrame frames[2] = { child->GetFrame(n, env), child->GetFrame(nref, env) };
      unsigned long long tok[2];
      int rc = token_of(frames[0], n, tok[0]);
      if (!rc)
        rc = token_of(frames[1], nref, tok[1]);
      if (!rc)
        rc = lib.nvof_token(&_sp, &_np, tok[0], tok[1], payload, _words);
      if (rc != MVCUDA_TOKEN_GONE)
        return rc;
      // evicted: make the inputs again (regenerated / uploaded and pinned) and run once more
      return with_inputs(frames, 2, _src_vi, _fmt, env, [&](const unsigned long long *t) {
        return lib.nvof_token(&_sp, &_np, t[0], t[1], payload, _words);
      });
    }

    // level 0 of MAnalyse's frame n (its finest level is the last in the payload) as this filter's payload
    void from_fallback(int n, int *payload, IScriptEnvironment *env)
    {
      PVideoFrame f = fallback(env)->GetFrame(n, env);
      const int *src = reinterpret_cast<const int *>(f->GetReadPtr());
      const int hs = src[0];
      const int *p = src + hs / 4;
      const int total = p[0];
      const int lvl0 = 3 * _ad.nBlkX * _ad.nBlkY + 1;
      payload[0] = _words;
      payload[1] = p[1];
      std::memcpy(payload + 2, p + total - lvl0, (size_t)lvl0 * sizeof(int));
    }

  public:
    CudaNVOFAnalyse(PClip c, bool isb, int delta, int bx, int by, int ox, int oy, int pel, int hpad, int vpad, bool chroma, int preset,
      IScriptEnvironment *env)
      : GenericVideoFilter(c), _isb(isb), _chroma(chroma), _delta(delta), _pel(pel), _hpad(hpad), _vpad(vpad), _src_vi(c->GetVideoInfo())
    {
      const VideoInfo &svi = _src_vi;
      if (!make_format(svi, _fmt))
        env->ThrowError("CudaNVOFAnalyse: planar YUV or Y, 8..16 bit, chroma subsampling 1 or 2");
      if (chroma && _fmt.planes != 3)
        chroma = _chroma = false;
      _ad.nMagicKey = MVAnalysisData::MOTION_MAGIC_KEY;
      _ad.nVersion = MVAnalysisData::VERSION;
      _ad.nBlkSizeX = bx;
      _ad.nBlkSizeY = by;
      _ad.nPel = pel;
      _ad.nLvCount = 1;
      _ad.nDeltaFrame = delta;
      _ad.isBackward = isb;
      _ad.nFlags = (isb ? MOTION_IS_BACKWARD : 0) | (chroma ? MOTION_USE_CHROMA_MOTION : 0);
      _ad.nWidth = svi.width;
      _ad.nHeight = svi.height;
      _ad.pixelType = svi.pixel_type;
      _ad.xRatioUV = _fmt.planes == 3 ? _fmt.x_ratio_uv : 1;
      _ad.yRatioUV = _fmt.planes == 3 ? _fmt.y_ratio_uv : 1;
      _ad.chromaSADScale = 0;
      _ad.pixelsize = svi.ComponentSize();
      _ad.bits_per_pixel = svi.BitsPerComponent();
      _ad.nHPadding = hpad;
      _ad.nVPadding = vpad;
      _ad.nOverlapX = ox;
      _ad.nOverlapY = oy;
      int nbx = (svi.width - ox) / (bx - ox), nby = (svi.height - oy) / (by - oy);
      if (nbx < 2 || nby < 2) // MAnalyse 2.7.36: no overlap when either count is below 2
      {
        _ad.nOverlapX = _ad.nOverlapY = 0;
        nbx = svi.width / bx;
        nby = svi.height / by;
      }
      if (nbx < 1 || nby < 1)
        env->ThrowError("CudaNVOFAnalyse: the clip is smaller than one block");
      _ad.nBlkX = nbx;
      _ad.nBlkY = nby;
      _words = 3 + 3 * nbx * nby;
      _big_sad = 3 * bx * by * (1 << _ad.bits_per_pixel);

      _sp.width = svi.width;
      _sp.height = svi.height;
      _sp.x_ratio_uv = _ad.xRatioUV;
      _sp.y_ratio_uv = _ad.yRatioUV;
      _sp.hpad = hpad;
      _sp.vpad = vpad;
      _sp.pel = pel;
      _sp.levels = 1;
      _sp.bits_per_pixel = _ad.bits_per_pixel;
      _sp.chroma = chroma ? 1 : 0;
      _sp.rfilter = 2;
      _sp.sharp = 2;
      _np.blk_x = bx;
      _np.blk_y = by;
      _np.overlap_x = _ad.nOverlapX;
      _np.overlap_y = _ad.nOverlapY;
      _np.n_blk_x = nbx;
      _np.n_blk_y = nby;
      // MAnalyse's effective chroma SAD scale (scaleCSAD 0): 4:2:0 as is, 4:2:2 halved, 4:4:4 quartered
      {
        int lx = 0, ly = 0;
        while ((1 << lx) < _ad.xRatioUV) lx++;
        while ((1 << ly) < _ad.yRatioUV) ly++;
        _np.chroma_sad_scale = 2 - (lx + ly);
      }
      _np.preset = preset;

      // the vector clip: BGR32, one row of header + payload (ClipFnc::format_vector_clip(vi, true, ..., "rgb32", ...))
      vi.pixel_type = VideoInfo::CS_BGR32;
      vi.width = (HEADER + _words * 4 + 3) / 4;
      vi.height = 1;
      vi.audio_samples_per_second = 0;
      const uintptr_t p = reinterpret_cast<uintptr_t>(&_ad);
      vi.nchannels = (int)(0x80000000u | (unsigned)(p >> 32));
      vi.sample_type = (int)(p & 0xffffffffu);
      if (log_enabled())
        std::fprintf(stderr, "CudaNVOFAnalyse: %dx%d blocks of %dx%d, pel %d, preset %d\n", nbx, nby, bx, by, pel, preset);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      PVideoFrame dst = env->NewVideoFrame(vi);
      uint8_t *d = dst->GetWritePtr();
      std::memset(d, 0, HEADER);
      const int hs = HEADER;
      std::memcpy(d, &hs, sizeof(int));
      std::memcpy(d + sizeof(int), &_ad, sizeof(_ad));
      int *payload = reinterpret_cast<int *>(d + HEADER);
      const int nref = _isb ? n + _delta : n - _delta;
      if (nref < 0 || nref >= _src_vi.num_frames)
      {
        // no reference: invalid vectors, as MAnalyse (GroupOfPlanes / PlaneOfBlocks::WriteDefaultToArray)
        payload[0] = _words;
        payload[1] = 0;
        payload[2] = _words - 2;
        for (int i = 3; i < _words; i += 3)
        {
          payload[i] = 0;
          payload[i + 1] = 0;
          payload[i + 2] = _big_sad;
        }
        return dst;
      }
      const int rc = gpu(n, nref, payload, env);
      if (rc != 0)
      {
        if (log_enabled())
          std::fprintf(stderr, "CudaNVOFAnalyse: frame %d: NVOFA failed (%d), MAnalyse's level 0 instead\n", n, rc);
        from_fallback(n, payload, env);
      }
      return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  const int g_blksizes[][2] = {
    { 64, 64 }, { 64, 48 }, { 64, 32 }, { 64, 16 }, { 48, 64 }, { 48, 48 }, { 48, 24 }, { 48, 12 }, { 32, 64 }, { 32, 32 },
    { 32, 24 }, { 32, 16 }, { 32, 8 }, { 24, 48 }, { 24, 32 }, { 24, 24 }, { 24, 12 }, { 24, 6 }, { 16, 64 }, { 16, 32 },
    { 16, 16 }, { 16, 12 }, { 16, 8 }, { 16, 4 }, { 16, 2 }, { 12, 48 }, { 12, 24 }, { 12, 16 }, { 12, 12 }, { 12, 6 },
    { 12, 3 }, { 8, 32 }, { 8, 16 }, { 8, 8 }, { 8, 4 }, { 8, 2 }, { 8, 1 }, { 6, 24 }, { 6, 12 }, { 6, 6 }, { 6, 3 },
    { 4, 8 }, { 4, 4 }, { 4, 2 }, { 3, 6 }, { 3, 3 }, { 2, 4 }, { 2, 2 } };

  bool nvof_usable()
  {
    CudaLib &lib = CudaLib::get();
    return lib.ok && lib.nvof_available() == 1;
  }
}

AVSValue __cdecl Create_CudaNVOFAnalyse(AVSValue args, void *, IScriptEnvironment *env)
{
  PClip c = args[0].AsClip();
  const bool isb = args[1].AsBool(false);
  const int delta = args[2].AsInt(1);
  const int bx = args[3].AsInt(16), by = args[4].AsInt(bx);
  const int ox = args[5].AsInt(0), oy = args[6].AsInt(ox);
  const int pel = args[7].AsInt(2), hpad = args[8].AsInt(16), vpad = args[9].AsInt(16);
  const bool chroma = args[10].AsBool(true);
  const int preset = args[11].AsInt(2);
  bool found = false;
  for (const auto &b : g_blksizes)
    found = found || (b[0] == bx && b[1] == by);
  if (!found)
    env->ThrowError("CudaNVOFAnalyse: invalid block size %d x %d (MAnalyse's sizes)", bx, by);
  if (ox < 0 || ox > bx / 2 || oy < 0 || oy > by / 2)
    env->ThrowError("CudaNVOFAnalyse: overlap must be less or equal than half block size");
  if (pel != 1 && pel != 2)
    env->ThrowError("CudaNVOFAnalyse: pel must be 1 or 2");
  if (delta < 1)
    env->ThrowError("CudaNVOFAnalyse: delta must be 1 or more");
  if (hpad < 0 || vpad < 0)
    env->ThrowError("CudaNVOFAnalyse: hpad and vpad must be 0 or more");
  if (preset < 1 || preset > 3)
    env->ThrowError("CudaNVOFAnalyse: preset must be 1 (fast), 2 (medium) or 3 (slow)");
  const VideoInfo &svi = c->GetVideoInfo();
  if (svi.IsPlanar() && !svi.IsRGB() && !svi.IsY())
  {
    const int xr = 1 << svi.GetPlaneWidthSubsampling(PLANAR_U), yr = 1 << svi.GetPlaneHeightSubsampling(PLANAR_U);
    if (ox % xr || oy % yr)
      env->ThrowError("CudaNVOFAnalyse: wrong overlap for the colorspace subsampling");
  }
  if (!nvof_usable())
  {
    if (log_enabled())
      std::fprintf(stderr, "CudaNVOFAnalyse: NVOFA not available, using MAnalyse\n");
    return make_manalyse(c, isb, delta, bx, by, ox, oy, pel, hpad, vpad, chroma, env);
  }
  return new CudaNVOFAnalyse(c, isb, delta, bx, by, ox, oy, pel, hpad, vpad, chroma, preset, env);
}

AVSValue __cdecl Create_CudaNVOFAvailable(AVSValue, void *, IScriptEnvironment *)
{
  return nvof_usable();
}
