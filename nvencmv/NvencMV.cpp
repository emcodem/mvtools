// nvencmv: motion vectors from NVIDIA's video encoder (NVENC, motion-estimation-only mode) as an MVTools vector clip.
//
// CudaNVENCAnalyse(clip, isb, delta, blksize, blksizeV, overlap, overlapV, pel, hpad, vpad, chroma, codec, preset, qp,
//   intra, p16, fresh, half, channels, swap, neighbours, lambda, fbpenalty, temporal, hintclip, stats)
// CudaNVENCAvailable(string codec)
//
// The output is a one-level vector clip in MAnalyse's format (as CudaNVOFAnalyse's): MRecalculate, MFlowFps, MDegrain,
// MCompensate of any MVTools2 build accept it. It takes the plain clip, not a super clip. Input frames may be ordinary
// frames (uploaded once per picture) or token frames of the Cuda* filters (used in place). The device work runs in
// mvtools_cuda (next to this plugin, the same file mvtools2 / cudafilters load); NVENC needs Pascal (GTX 10xx) or newer.
// NOT MAnalyse's search and not bit-exact with anything: an encoder's block matching plus our candidate selection, made
// for experiments with every engine option (see the parameter list in AvisynthPluginInit3).

#include "avisynth.h"
#include "dyn_lib.h"
#include "mvcuda.h"

#include "../Sources/MVAnalysisData.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#define _stricmp strcasecmp
#include <strings.h>
#endif

namespace
{
  // mvtools_cuda next to this plugin
  struct CudaLib
  {
    decltype(&mvcuda_frame_upload_keyed) upload_keyed = nullptr;
    decltype(&mvcuda_token_regenerate) regenerate = nullptr;
    decltype(&mvcuda_token_pin) pin = nullptr;
    decltype(&mvcuda_token_unpin) unpin = nullptr;
    decltype(&mvcuda_nvenc_token) nvenc_token = nullptr;
    decltype(&mvcuda_nvenc_grid) nvenc_grid = nullptr;
    decltype(&mvcuda_nvenc_available) nvenc_available = nullptr;
    std::string why;
    bool ok = false;
    static CudaLib &get()
    {
      static CudaLib lib;
      return lib;
    }

  private:
    CudaLib()
    {
      if (std::getenv("MVTOOLS_GPU_DISABLE"))
      {
        why = "disabled by the environment variable MVTOOLS_GPU_DISABLE";
        return;
      }
      const std::filesystem::path dir = dynlib::module_dir(reinterpret_cast<const void *>(&CudaLib::get));
      void *lib = dir.empty() ? nullptr : dynlib::open(dir / dynlib::cuda_lib_name);
      if (!lib)
      {
        why = std::string(dynlib::cuda_lib_name) + " was not found or could not be loaded from the plugin folder";
        return;
      }
      auto version = reinterpret_cast<int (*)()>(dynlib::symbol(lib, "mvcuda_api_version"));
      if (!version || version() != MVCUDA_API_VERSION)
      {
        why = std::string(dynlib::cuda_lib_name) + " is from another build (library API version differs)";
        return;
      }
      auto count = reinterpret_cast<int (*)()>(dynlib::symbol(lib, "mvcuda_device_count"));
#define NV_LOAD(member, name) member = reinterpret_cast<decltype(member)>(dynlib::symbol(lib, name))
      NV_LOAD(upload_keyed, "mvcuda_frame_upload_keyed");
      NV_LOAD(regenerate, "mvcuda_token_regenerate");
      NV_LOAD(pin, "mvcuda_token_pin");
      NV_LOAD(unpin, "mvcuda_token_unpin");
      NV_LOAD(nvenc_token, "mvcuda_nvenc_token");
      NV_LOAD(nvenc_grid, "mvcuda_nvenc_grid");
      NV_LOAD(nvenc_available, "mvcuda_nvenc_available");
#undef NV_LOAD
      if (!(upload_keyed && regenerate && pin && unpin && nvenc_token && nvenc_grid && nvenc_available && count))
      {
        why = std::string(dynlib::cuda_lib_name) + " lacks the NVENC entry points";
        return;
      }
      if (count() <= 0)
      {
        why = "no usable NVIDIA GPU";
        return;
      }
      ok = true;
    }
  };

  const int g_planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };

  bool make_format(const VideoInfo &vi, MvCudaFrameFormat &f)
  {
    std::memset(&f, 0, sizeof(f));
    if (!vi.IsPlanar() || vi.IsRGB() || vi.IsYUVA() || vi.width <= 0 || vi.height <= 0)
      return false;
    f.width = vi.width;
    f.height = vi.height;
    f.planes = vi.IsY() ? 1 : 3;
    f.bytes_per_sample = vi.ComponentSize();
    f.bits_per_sample = vi.BitsPerComponent();
    if (f.bytes_per_sample != 1 && f.bytes_per_sample != 2)
      return false;
    f.x_ratio_uv = f.y_ratio_uv = 1;
    if (f.planes == 3)
    {
      f.x_ratio_uv = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
      f.y_ratio_uv = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
      if (f.x_ratio_uv > 2 || f.y_ratio_uv > 2)
        return false;
    }
    return (size_t)vi.width * f.bytes_per_sample >= sizeof(MvCudaTokenHeader);
  }

  bool read_token(const PVideoFrame &f, const VideoInfo &vi, MvCudaTokenHeader &h)
  {
    if (!vi.IsPlanar() || vi.IsRGB() || f->GetRowSize(PLANAR_Y) < (int)sizeof(h))
      return false;
    std::memcpy(&h, f->GetReadPtr(PLANAR_Y), sizeof(h));
    return h.magic == MVCUDA_TOKEN_MAGIC && h.layout == MVCUDA_TOKEN_LAYOUT;
  }

  bool same_format(const MvCudaFrameFormat &a, const MvCudaFrameFormat &b)
  {
    return a.width == b.width && a.height == b.height && a.planes == b.planes && a.bytes_per_sample == b.bytes_per_sample
      && a.bits_per_sample == b.bits_per_sample && (a.planes == 1 || (a.x_ratio_uv == b.x_ratio_uv && a.y_ratio_uv == b.y_ratio_uv));
  }

  // same key as mvtools2's GpuBackend::frame_key: one upload per picture of a clip while it stays in the registry
  unsigned long long frame_key(const void *clip, int n)
  {
    unsigned long long k = (unsigned long long)(uintptr_t)clip * 0x9E3779B97F4A7C15ull;
    k ^= ((unsigned long long)(n + 1) * 0xC2B2AE3D27D4EB4Full) + (k << 6) + (k >> 2);
    return k | 1ull;
  }

  bool fallback_allowed(IScriptEnvironment *env)
  {
    const AVSValue g = env->GetVarDef("cuda_cpu_fallback", AVSValue());
    if (g.IsBool())
      return g.AsBool();
    const char *e = std::getenv("CUDA_CPU_FALLBACK");
    return e && *e && std::strcmp(e, "0") != 0;
  }

  PClip make_manalyse(PClip c, bool isb, int delta, int bx, int by, int ox, int oy, int pel, int hpad, int vpad, bool chroma,
    IScriptEnvironment *env)
  {
    if (!env->FunctionExists("MSuper") || !env->FunctionExists("MAnalyse"))
      env->ThrowError("CudaNVENCAnalyse: NVENC is not available and its fallback needs mvtools2 (MSuper / MAnalyse): load it");
    // token frames (Cuda* filters in front) become pixels first, as CudaNVOFAnalyse's fallback does
    if (env->FunctionExists("CudaDownload"))
      c = env->Invoke("CudaDownload", c).AsClip();
    AVSValue sargs[5] = { c, pel, hpad, vpad, chroma };
    const char *snames[5] = { nullptr, "pel", "hpad", "vpad", "chroma" };
    PClip sup = env->Invoke("MSuper", AVSValue(sargs, 5), snames).AsClip();
    AVSValue aargs[10] = { sup, isb, delta, bx, by, ox, oy, true, chroma, 0 };
    const char *anames[10] = { nullptr, "isb", "delta", "blksize", "blksizeV", "overlap", "overlapV", "truemotion", "chroma", "scaleCSAD" };
    return env->Invoke("MAnalyse", AVSValue(aargs, 10), anames).AsClip();
  }

  // lower median of v (reordered)
  int median_of(std::vector<int> &v)
  {
    const size_t k = (v.size() - 1) / 2;
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)k, v.end());
    return v[k];
  }

  inline int round_div(int num, int den)
  {
    return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
  }

  class CudaNVENCAnalyse : public GenericVideoFilter
  {
    MVAnalysisData _ad = {}; // vi's pointer hack points here (MVClip), valid for the filter's lifetime
    PClip _fallback, _hintclip;
    bool _isb, _chroma, _temporal;
    int _delta, _pel, _hpad, _vpad;
    int _words = 0, _big_sad = 0;
    int _gw = 0, _gh = 0, _mbw = 0, _mbh = 0, _scale = 1; // ME grid, macroblocks, source pixels per ME pixel
    MvCudaFrameFormat _fmt = {};
    MvCudaSuperParams _sp = {};
    MvCudaNvencParams _ep = {};
    VideoInfo _src_vi;
    std::mutex _m;
    std::string _stats_path;
    long long _stats[64] = {};
    static const int HEADER = 256;

    PClip fallback(IScriptEnvironment *env)
    {
      std::lock_guard<std::mutex> lk(_m);
      if (!_fallback)
        _fallback = make_manalyse(child, _isb, _delta, _ad.nBlkSizeX, _ad.nBlkSizeY, _ad.nOverlapX, _ad.nOverlapY, _pel, _hpad, _vpad,
          _chroma, env);
      return _fallback;
    }

    // the device token of frame n of the child: its own token, or the picture uploaded under its key
    int token_of(const PVideoFrame &f, int n, unsigned long long &tok, MvCudaTokenHeader *hdr = nullptr)
    {
      MvCudaTokenHeader h;
      if (read_token(f, _src_vi, h))
      {
        if (!same_format(h.format, _fmt))
          return -4;
        tok = h.token;
        if (hdr)
          *hdr = h;
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

    // one library call on frames (a, b); on an evicted input: make both again (regenerate token inputs, upload the
    // others), pin them, and run once more
    int call(const MvCudaNvencParams &ep, int a, int b, const short *hints, int *payload, short *grid, long long *stats,
      IScriptEnvironment *env)
    {
      CudaLib &lib = CudaLib::get();
      PVideoFrame fr[2] = { child->GetFrame(a, env), child->GetFrame(b, env) };
      const int nn[2] = { a, b };
      unsigned long long tok[2] = {};
      MvCudaTokenHeader hdr[2] = {};
      bool is_tok[2];
      for (int i = 0; i < 2; i++)
      {
        is_tok[i] = read_token(fr[i], _src_vi, hdr[i]);
        const int rc = token_of(fr[i], nn[i], tok[i]);
        if (rc)
          return rc;
      }
      auto op = [&]() {
        return lib.nvenc_token(&_sp, &ep, tok[0], tok[1], hints, payload, _words, grid, grid ? 2 * _gw * _gh : 0, stats);
      };
      int rc = op();
      if (rc != MVCUDA_TOKEN_GONE)
        return rc;
      bool pinned[2] = {};
      rc = 0;
      for (int i = 0; i < 2 && rc == 0; i++)
      {
        if (lib.pin(tok[i]) == 0)
        {
          pinned[i] = true;
          continue;
        }
        MvCudaTokenHeader h;
        if (is_tok[i])
        {
          if (!hdr[i].producer || lib.regenerate(hdr[i].producer, hdr[i].frame, env, &h) != 0)
            rc = MVCUDA_TOKEN_GONE;
          else
            tok[i] = h.token;
        }
        else
          rc = token_of(fr[i], nn[i], tok[i]);
        if (rc == 0)
        {
          rc = lib.pin(tok[i]) == 0 ? 0 : MVCUDA_TOKEN_GONE;
          pinned[i] = rc == 0;
        }
      }
      if (rc == 0)
        rc = op();
      for (int i = 0; i < 2; i++)
        if (pinned[i])
          lib.unpin(tok[i]);
      return rc;
    }

    // hints from a grid of the ME picture (quarter pel, 8x8 cells): per macroblock the cell under its centre and the
    // median of the 3x3 cells around it, integer pel
    void hints_from_grid(const std::vector<short> &g, short *out, int stride, int first)
    {
      std::vector<int> vx, vy;
      for (int my = 0; my < _mbh; my++)
        for (int mx = 0; mx < _mbw; mx++)
        {
          const int cx = std::min((mx * 16 + 8) / 8, _gw - 1), cy = std::min((my * 16 + 8) / 8, _gh - 1);
          vx.clear();
          vy.clear();
          for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
              const int x = std::min(std::max(cx + dx, 0), _gw - 1), y = std::min(std::max(cy + dy, 0), _gh - 1);
              vx.push_back(g[2 * ((size_t)y * _gw + x)]);
              vy.push_back(g[2 * ((size_t)y * _gw + x) + 1]);
            }
          short *h = out + ((size_t)my * _mbw + mx) * stride * 2 + first * 2;
          h[0] = (short)round_div(g[2 * ((size_t)cy * _gw + cx)], 4);
          h[1] = (short)round_div(g[2 * ((size_t)cy * _gw + cx) + 1], 4);
          h[2] = (short)round_div(median_of(vx), 4);
          h[3] = (short)round_div(median_of(vy), 4);
        }
    }

    // hints from frame n of the hint clip (any one-or-more-level MVTools vector clip of the same picture size): its
    // finest level; per macroblock the block under the centre and the median of the 3x3 blocks around it, integer pel
    // of the ME picture
    void hints_from_clip(int n, short *out, int stride, int first, IScriptEnvironment *env)
    {
      PVideoFrame f = _hintclip->GetFrame(n, env);
      const uint8_t *base = f->GetReadPtr();
      int hs;
      std::memcpy(&hs, base, sizeof(int));
      MVAnalysisData ad;
      std::memcpy(&ad, base + sizeof(int), sizeof(ad));
      const int *p = reinterpret_cast<const int *>(base + hs);
      const int total = p[0], nb = ad.nBlkX * ad.nBlkY;
      const int *v = p + total - 3 * nb; // finest level: the last one
      const bool valid = p[1] != 0;
      const int sx = ad.nBlkSizeX - ad.nOverlapX, sy = ad.nBlkSizeY - ad.nOverlapY;
      std::vector<int> vx, vy;
      for (int my = 0; my < _mbh; my++)
        for (int mx = 0; mx < _mbw; mx++)
        {
          short *h = out + ((size_t)my * _mbw + mx) * stride * 2 + first * 2;
          if (!valid)
          {
            h[0] = h[1] = h[2] = h[3] = 0;
            continue;
          }
          const int px = (mx * 16 + 8) * _scale, py = (my * 16 + 8) * _scale; // source pixels
          const int bx = std::min(std::max((px - ad.nBlkSizeX / 2 + sx / 2) / sx, 0), ad.nBlkX - 1);
          const int by = std::min(std::max((py - ad.nBlkSizeY / 2 + sy / 2) / sy, 0), ad.nBlkY - 1);
          vx.clear();
          vy.clear();
          for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
              const int x = std::min(std::max(bx + dx, 0), ad.nBlkX - 1), y = std::min(std::max(by + dy, 0), ad.nBlkY - 1);
              vx.push_back(v[3 * (y * ad.nBlkX + x)]);
              vy.push_back(v[3 * (y * ad.nBlkX + x) + 1]);
            }
          const int den = ad.nPel * _scale;
          h[0] = (short)round_div(v[3 * (by * ad.nBlkX + bx)], den);
          h[1] = (short)round_div(v[3 * (by * ad.nBlkX + bx) + 1], den);
          h[2] = (short)round_div(median_of(vx), den);
          h[3] = (short)round_div(median_of(vy), den);
        }
      (void)env;
    }

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
    CudaNVENCAnalyse(PClip c, bool isb, int delta, int bx, int by, int ox, int oy, int pel, int hpad, int vpad, bool chroma,
      const MvCudaNvencParams &ep, bool temporal, PClip hintclip, const char *stats, IScriptEnvironment *env)
      : GenericVideoFilter(c), _hintclip(hintclip), _isb(isb), _chroma(chroma), _temporal(temporal), _delta(delta), _pel(pel), _hpad(hpad),
        _vpad(vpad), _ep(ep), _src_vi(c->GetVideoInfo()), _stats_path(stats ? stats : "")
    {
      const VideoInfo &svi = _src_vi;
      if (!make_format(svi, _fmt))
        env->ThrowError("CudaNVENCAnalyse: planar YUV or Y, 8..16 bit, chroma subsampling 1 or 2");
      if (_fmt.planes != 3)
      {
        if (_ep.channels)
          env->ThrowError("CudaNVENCAnalyse: channels=\"yuv\" / \"rgb\" need a YUV clip");
        chroma = _chroma = false;
      }
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
        env->ThrowError("CudaNVENCAnalyse: the clip is smaller than one block");
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
      // the chroma planes of the super frame: for the SADs (chroma) or for the colour runs (channels)
      _sp.chroma = (chroma || _ep.channels) && _fmt.planes == 3 ? 1 : 0;
      _ep.sad_chroma = chroma ? 1 : 0;
      _sp.rfilter = 2;
      _sp.sharp = 2;
      _ep.blk_x = bx;
      _ep.blk_y = by;
      _ep.overlap_x = _ad.nOverlapX;
      _ep.overlap_y = _ad.nOverlapY;
      _ep.n_blk_x = nbx;
      _ep.n_blk_y = nby;
      {
        int lx = 0, ly = 0;
        while ((1 << lx) < _ad.xRatioUV) lx++;
        while ((1 << ly) < _ad.yRatioUV) ly++;
        _ep.chroma_sad_scale = 2 - (lx + ly); // MAnalyse's effective scale with scaleCSAD 0
      }
      _scale = _ep.half ? 2 : 1;
      CudaLib::get().nvenc_grid(svi.width, svi.height, _ep.codec, _ep.half, &_gw, &_gh);
      {
        const int W = ((((svi.width + _scale - 1) / _scale) + 1) & ~1), H = ((((svi.height + _scale - 1) / _scale) + 1) & ~1);
        _mbw = (W + 15) / 16;
        _mbh = (H + 15) / 16;
      }
      _ep.hints = (_temporal ? 2 : 0) + (_hintclip ? 2 : 0);
      if (_ep.hints && _ep.codec)
        env->ThrowError("CudaNVENCAnalyse: hints (temporal, hintclip) work with codec=\"h264\" only");
      if (_hintclip)
      {
        const VideoInfo &hv = _hintclip->GetVideoInfo();
        if (hv.num_frames < svi.num_frames)
          env->ThrowError("CudaNVENCAnalyse: hintclip is shorter than the clip");
        PVideoFrame f = _hintclip->GetFrame(0, env);
        int hs = 0;
        MVAnalysisData ad;
        std::memcpy(&hs, f->GetReadPtr(), sizeof(int));
        std::memcpy(&ad, f->GetReadPtr() + sizeof(int), sizeof(ad));
        if (hv.IsRGB() == false || hs < (int)(sizeof(int) + sizeof(ad)) || ad.nMagicKey != MVAnalysisData::MOTION_MAGIC_KEY)
          env->ThrowError("CudaNVENCAnalyse: hintclip is not an MVTools vector clip");
        if (ad.nWidth != svi.width || ad.nHeight != svi.height)
          env->ThrowError("CudaNVENCAnalyse: hintclip was made for another picture size");
        if ((ad.isBackward != 0) != isb || ad.nDeltaFrame != delta)
          env->ThrowError("CudaNVENCAnalyse: hintclip must have the same direction (isb) and delta");
      }

      vi.pixel_type = VideoInfo::CS_BGR32;
      vi.width = (HEADER + _words * 4 + 3) / 4;
      vi.height = 1;
      vi.audio_samples_per_second = 0;
      const uintptr_t p = reinterpret_cast<uintptr_t>(&_ad);
      vi.nchannels = (int)(0x80000000u | (unsigned)(p >> 32));
      vi.sample_type = (int)(p & 0xffffffffu);
    }

    ~CudaNVENCAnalyse() override
    {
      if (_stats_path.empty())
        return;
      if (FILE *o = std::fopen(_stats_path.c_str(), "a"))
      {
        std::fprintf(o, "isb=%d delta=%d channels=%d swap=%d neighbours=%d", (int)_isb, _delta, _ep.channels, _ep.swap, _ep.neighbours);
        for (long long v : _stats)
          std::fprintf(o, " %lld", v);
        std::fprintf(o, "\n");
        std::fclose(o);
      }
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
      const int nf = _src_vi.num_frames;
      if (nref < 0 || nref >= nf)
      {
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
      std::vector<short> hints;
      int rc = 0;
      if (_ep.hints)
      {
        hints.assign((size_t)_mbw * _mbh * _ep.hints * 2, 0);
        int first = 0;
        if (_temporal)
        {
          // the previous pair of the same direction (the next one at the start), computed without hints
          int pn = n - 1, pr = nref - 1;
          if (pn < 0 || pr < 0)
          {
            pn = n + 1;
            pr = nref + 1;
          }
          if (pn >= 0 && pr >= 0 && pn < nf && pr < nf)
          {
            MvCudaNvencParams e2 = _ep;
            e2.hints = 0;
            std::vector<int> tmp(_words);
            std::vector<short> g((size_t)2 * _gw * _gh);
            rc = call(e2, pn, pr, nullptr, tmp.data(), g.data(), nullptr, env);
            if (rc == 0)
              hints_from_grid(g, hints.data(), _ep.hints, first);
          }
          first += 2;
        }
        if (rc == 0 && _hintclip)
          hints_from_clip(n, hints.data(), _ep.hints, first, env);
      }
      long long st[64] = {};
      if (rc == 0)
        rc = call(_ep, n, nref, hints.empty() ? nullptr : hints.data(), payload, nullptr, _stats_path.empty() ? nullptr : st, env);
      if (rc != 0)
      {
        static bool told = false;
        if (!told)
          std::fprintf(stderr, "CudaNVENCAnalyse: frame %d: NVENC failed (%d), MAnalyse's level 0 instead\n", n, rc);
        told = true;
        from_fallback(n, payload, env);
      }
      else if (!_stats_path.empty())
      {
        std::lock_guard<std::mutex> lk(_m);
        for (int i = 0; i < 64; i++)
          _stats[i] += st[i];
      }
      return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int) override
    {
      // fresh=false: each instance's encoders keep state, so calls must not interleave across threads differently per
      // run; MT_SERIALIZED would be safest, MT_NICE_FILTER keeps it parallel (deterministic with fresh=true)
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  const int g_blksizes[][2] = {
    { 64, 64 }, { 64, 48 }, { 64, 32 }, { 64, 16 }, { 48, 64 }, { 48, 48 }, { 48, 24 }, { 48, 12 }, { 32, 64 }, { 32, 32 },
    { 32, 24 }, { 32, 16 }, { 32, 8 }, { 24, 48 }, { 24, 32 }, { 24, 24 }, { 24, 12 }, { 24, 6 }, { 16, 64 }, { 16, 32 },
    { 16, 16 }, { 16, 12 }, { 16, 8 }, { 16, 4 }, { 16, 2 }, { 12, 48 }, { 12, 24 }, { 12, 16 }, { 12, 12 }, { 12, 6 },
    { 12, 3 }, { 8, 32 }, { 8, 16 }, { 8, 8 }, { 8, 4 }, { 8, 2 }, { 8, 1 }, { 6, 24 }, { 6, 12 }, { 6, 6 }, { 6, 3 },
    { 4, 8 }, { 4, 4 }, { 4, 2 }, { 3, 6 }, { 3, 3 }, { 2, 4 }, { 2, 2 } };

  int codec_of(const char *s, IScriptEnvironment *env)
  {
    if (!s || !_stricmp(s, "h264"))
      return 0;
    if (!_stricmp(s, "hevc"))
      return 1;
    env->ThrowError("CudaNVENCAnalyse: codec must be \"h264\" or \"hevc\"");
    return 0;
  }

  bool usable(int codec, std::string &why)
  {
    CudaLib &lib = CudaLib::get();
    if (!lib.ok)
    {
      why = lib.why;
      return false;
    }
    char b[512] = {};
    const int r = lib.nvenc_available(codec, b, (int)sizeof b);
    why = b;
    return r == 1;
  }

  AVSValue __cdecl Create(AVSValue args, void *, IScriptEnvironment *env)
  {
    PClip c = args[0].AsClip();
    const bool isb = args[1].AsBool(false);
    const int delta = args[2].AsInt(1);
    const int bx = args[3].AsInt(16), by = args[4].AsInt(bx);
    const int ox = args[5].AsInt(0), oy = args[6].AsInt(ox);
    const int pel = args[7].AsInt(2), hpad = args[8].AsInt(8), vpad = args[9].AsInt(8); // MSuper's defaults
    const bool chroma = args[10].AsBool(true);
    MvCudaNvencParams ep = {};
    ep.codec = codec_of(args[11].AsString("h264"), env);
    ep.preset = args[12].AsInt(4);
    ep.qp = args[13].AsInt(10);
    ep.no_intra = args[14].AsBool(true) ? 0 : 1;
    ep.p16 = args[15].AsBool(false) ? 1 : 0;
    const char *se = args[16].AsString("reset");
    ep.fresh = !_stricmp(se, "fresh") ? 1 : !_stricmp(se, "reset") ? 2 : !_stricmp(se, "kept") ? 0 : -1;
    ep.half = args[17].AsBool(false) ? 1 : 0;
    const char *ch = args[18].AsString("y");
    ep.channels = !_stricmp(ch, "y") ? 0 : !_stricmp(ch, "yuv") ? 1 : !_stricmp(ch, "rgb") ? 2 : -1;
    ep.swap = args[19].AsBool(true) ? 1 : 0;
    ep.neighbours = args[20].AsBool(false) ? 1 : 0;
    ep.lambda = args[21].AsInt(0);
    ep.fb_penalty = args[22].AsInt(0);
    const bool temporal = args[23].AsBool(false);
    PClip hintclip = args[24].Defined() ? args[24].AsClip() : nullptr;
    const char *stats = args[25].AsString(nullptr);

    bool found = false;
    for (const auto &b : g_blksizes)
      found = found || (b[0] == bx && b[1] == by);
    if (!found)
      env->ThrowError("CudaNVENCAnalyse: invalid block size %d x %d (MAnalyse's sizes)", bx, by);
    if (ox < 0 || ox > bx / 2 || oy < 0 || oy > by / 2)
      env->ThrowError("CudaNVENCAnalyse: overlap must be less or equal than half block size");
    if (pel != 1 && pel != 2)
      env->ThrowError("CudaNVENCAnalyse: pel must be 1 or 2");
    if (delta < 1)
      env->ThrowError("CudaNVENCAnalyse: delta must be 1 or more");
    if (hpad < 0 || vpad < 0)
      env->ThrowError("CudaNVENCAnalyse: hpad and vpad must be 0 or more");
    if (ep.preset < 1 || ep.preset > 7)
      env->ThrowError("CudaNVENCAnalyse: preset must be 1..7 (NVENC P1..P7)");
    if (ep.qp < -1 || ep.qp > 51)
      env->ThrowError("CudaNVENCAnalyse: qp must be -1 (the preset's rate control) or 0..51");
    if (ep.fresh < 0)
      env->ThrowError("CudaNVENCAnalyse: session must be \"reset\", \"fresh\" or \"kept\"");
    if (ep.channels < 0)
      env->ThrowError("CudaNVENCAnalyse: channels must be \"y\", \"yuv\" or \"rgb\"");
    if (ep.lambda < 0 || ep.fb_penalty < 0)
      env->ThrowError("CudaNVENCAnalyse: lambda and fbpenalty must be 0 or more");
    if (ep.fb_penalty && !ep.swap)
      env->ThrowError("CudaNVENCAnalyse: fbpenalty needs swap=true");
    const VideoInfo &svi = c->GetVideoInfo();
    if (svi.IsPlanar() && !svi.IsRGB() && !svi.IsY())
    {
      const int xr = 1 << svi.GetPlaneWidthSubsampling(PLANAR_U), yr = 1 << svi.GetPlaneHeightSubsampling(PLANAR_U);
      if (ox % xr || oy % yr)
        env->ThrowError("CudaNVENCAnalyse: wrong overlap for the colorspace subsampling");
    }
    std::string why;
    if (!usable(ep.codec, why))
    {
      if (!fallback_allowed(env))
        env->ThrowError("CudaNVENCAnalyse: NVENC cannot be used: %s. Set global cuda_cpu_fallback = true to run MAnalyse instead",
          why.c_str());
      return make_manalyse(c, isb, delta, bx, by, ox, oy, pel, hpad, vpad, chroma, env);
    }
    return new CudaNVENCAnalyse(c, isb, delta, bx, by, ox, oy, pel, hpad, vpad, chroma, ep, temporal, hintclip, stats, env);
  }

  AVSValue __cdecl CreateAvailable(AVSValue args, void *, IScriptEnvironment *env)
  {
    std::string why;
    return usable(codec_of(args[0].AsString("h264"), env), why);
  }
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
  AVS_linkage = vectors;
  env->AddFunction("CudaNVENCAnalyse",
    "c[isb]b[delta]i[blksize]i[blksizeV]i[overlap]i[overlapV]i[pel]i[hpad]i[vpad]i[chroma]b"
    "[codec]s[preset]i[qp]i[intra]b[p16]b[session]s[half]b[channels]s[swap]b[neighbours]b[lambda]i[fbpenalty]i"
    "[temporal]b[hintclip]c[stats]s",
    Create, nullptr);
  env->AddFunction("CudaNVENCAvailable", "[codec]s", CreateAvailable, nullptr);
  return "nvencmv: motion vectors from NVIDIA's video encoder (NVENC ME-only)";
}
