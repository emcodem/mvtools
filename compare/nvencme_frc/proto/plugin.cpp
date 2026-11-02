// NVENCMEProto(clip, isb, delta, blksize, blksizeV, overlap, overlapV, pel, hpad, vpad, chroma, preset, codec, half):
// throwaway prototype of NVENCMEAnalyse for the visual comparison. A one-level vector clip in
// MAnalyse's format (the layout of cudafilters' CudaNVOFAnalyse) from NVENC's ME-only mode, standalone (CPU frames,
// own CUDA driver context). SADs are written as 0: use MRecalculate downstream, it recomputes them
// (PlaneOfBlocks.cpp:3358-3362). half=true: ME on the luma downscaled 2x2 (twice the search range, half the precision).
#include "nvme_core.h"
#include "dx12me.h"

#include <map>
#include <mutex>
#include <thread>
#include <tuple>

#include "avisynth.h"
#include "MVAnalysisData.h"

#include <cmath>
#include <cstring>

const AVS_Linkage *AVS_linkage = nullptr;

namespace
{
  class NVENCMEProto : public GenericVideoFilter
  {
    MVAnalysisData _ad = {};
    bool _isb, _half, _chroma_me = nvme::env_on("NVME_CHROMA");
    int _levels = 1;
    int _dx12 = 0; // 0: NVENC SDK, 8 / 16: D3D12 video motion estimator with that block size (DTL's DX12_ME path)
    double _pseudo = std::getenv("NVME_PSEUDO") ? std::atof(std::getenv("NVME_PSEUDO")) : 0.0;
    int _delta, _pel, _words = 0, _big_sad = 0;
    nvme::Params _np;
    VideoInfo _src_vi;
    static const int HEADER = 256;

    // NV12 chroma of frame n for the ME picture (8 bit, averaged over the source chroma samples under each 2x2 (half:
    // 4x4) luma area); empty unless _chroma_me
    void chroma(const PVideoFrame &f, std::vector<uint8_t> &uv)
    {
      uv.clear();
      if (!_chroma_me || _src_vi.IsY())
        return;
      const int xr = 1 << _src_vi.GetPlaneWidthSubsampling(PLANAR_U), yr = 1 << _src_vi.GetPlaneHeightSubsampling(PLANAR_U);
      const int cw = _src_vi.width / xr, ch = _src_vi.height / yr, fx = _half ? 4 : 2;
      const int bits = _src_vi.BitsPerComponent(), shift = bits - 8;
      const int cols = (_np.w + 1) / 2, rows = (_np.h + 1) / 2;
      uv.assign((size_t)_np.w * rows, 128);
      for (int pl = 0; pl < 2; pl++)
      {
        const int plane = pl ? PLANAR_V : PLANAR_U;
        const uint8_t *p = f->GetReadPtr(plane);
        const int pitch = f->GetPitch(plane);
        for (int y = 0; y < rows; y++)
          for (int x = 0; x < cols; x++)
          {
            const int x0 = x * fx / xr, x1 = std::max(x0 + 1, std::min(cw, (x + 1) * fx / xr));
            const int y0 = y * fx / yr, y1 = std::max(y0 + 1, std::min(ch, (y + 1) * fx / yr));
            int sum = 0, cnt = 0;
            for (int yy = std::min(y0, ch - 1); yy < y1 && yy < ch; yy++)
              for (int xx = std::min(x0, cw - 1); xx < x1 && xx < cw; xx++)
              {
                sum += bits == 8 ? p[(size_t)yy * pitch + xx] : reinterpret_cast<const uint16_t *>(p + (size_t)yy * pitch)[xx];
                cnt++;
              }
            int v = cnt ? (sum + cnt / 2) / cnt : 128 << shift;
            if (shift > 0)
              v = std::min((v + (1 << (shift - 1))) >> shift, 255);
            if (2 * x + pl < _np.w)
              uv[(size_t)y * _np.w + 2 * x + pl] = (uint8_t)v;
          }
      }
    }

    // 8-bit luma of frame n (2x2 box-downscaled when _half), and its NV12 chroma (see chroma())
    void gray(int n, std::vector<uint8_t> &g, std::vector<uint8_t> &uv, IScriptEnvironment *env)
    {
      PVideoFrame f = child->GetFrame(n, env);
      chroma(f, uv);
      const uint8_t *p = f->GetReadPtr(PLANAR_Y);
      const int pitch = f->GetPitch(PLANAR_Y), w = _src_vi.width, h = _src_vi.height;
      const int bits = _src_vi.BitsPerComponent(), shift = bits - 8;
      auto rd = [&](const uint8_t *b, int pt, int x, int y) -> int {
        if (bits == 8)
          return b[(size_t)y * pt + x];
        const int v = reinterpret_cast<const uint16_t *>(b + (size_t)y * pt)[x];
        return std::min((v + (1 << (shift - 1))) >> shift, 255);
      };
      // pseudo-luma (NVME_PSEUDO=k): Y + k (U - 128) - k (V - 128), so that colour edges reach NVENC's luma-only ME
      const bool pseudo = _pseudo != 0.0 && !_src_vi.IsY();
      const uint8_t *pu = pseudo ? f->GetReadPtr(PLANAR_U) : nullptr, *pv = pseudo ? f->GetReadPtr(PLANAR_V) : nullptr;
      const int cpitch = pseudo ? f->GetPitch(PLANAR_U) : 0;
      const int lx = pseudo ? _src_vi.GetPlaneWidthSubsampling(PLANAR_U) : 0, ly = pseudo ? _src_vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;
      auto px = [&](int x, int y) -> int {
        const int yv = rd(p, pitch, x, y);
        if (!pseudo)
          return yv;
        const int u = rd(pu, cpitch, x >> lx, y >> ly) - 128, v = rd(pv, cpitch, x >> lx, y >> ly) - 128;
        return std::min(std::max((int)std::lround(yv + _pseudo * (u - v)), 0), 255);
      };
      g.resize((size_t)_np.w * _np.h);
      for (int y = 0; y < _np.h; y++)
        for (int x = 0; x < _np.w; x++)
        {
          int v;
          if (_half)
          {
            const int x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1), y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
            v = (px(x0, y0) + px(x1, y0) + px(x0, y1) + px(x1, y1) + 2) >> 2;
          }
          else
            v = px(x, y);
          g[(size_t)y * _np.w + x] = (uint8_t)v;
        }
    }

  public:
    NVENCMEProto(PClip c, bool isb, int delta, int bx, int by, int ox, int oy, int pel, int hpad, int vpad, bool chroma, int preset,
      bool hevc, bool half, int levels, int dx12, IScriptEnvironment *env)
      : GenericVideoFilter(c), _isb(isb), _half(half), _levels(levels), _dx12(dx12), _delta(delta), _pel(pel), _src_vi(c->GetVideoInfo())
    {
      const VideoInfo &svi = _src_vi;
      if (!svi.IsPlanar() || svi.IsRGB() || svi.BitsPerComponent() > 16 || svi.ComponentSize() == 4)
        env->ThrowError("NVENCMEProto: planar YUV or Y, 8..16 bit");
      if (svi.IsY())
        chroma = false;
      if (!nvme::Driver::get().ok)
        env->ThrowError("NVENCMEProto: NVENC not available: %s", nvme::Driver::get().err.c_str());
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
      _ad.xRatioUV = svi.IsY() ? 1 : 1 << svi.GetPlaneWidthSubsampling(PLANAR_U);
      _ad.yRatioUV = svi.IsY() ? 1 : 1 << svi.GetPlaneHeightSubsampling(PLANAR_U);
      _ad.chromaSADScale = 0;
      _ad.pixelsize = svi.ComponentSize();
      _ad.bits_per_pixel = svi.BitsPerComponent();
      _ad.nHPadding = hpad;
      _ad.nVPadding = vpad;
      _ad.nOverlapX = ox;
      _ad.nOverlapY = oy;
      int nbx = (svi.width - ox) / (bx - ox), nby = (svi.height - oy) / (by - oy);
      if (nbx < 2 || nby < 2)
      {
        _ad.nOverlapX = _ad.nOverlapY = 0;
        nbx = svi.width / bx;
        nby = svi.height / by;
      }
      if (nbx < 1 || nby < 1)
        env->ThrowError("NVENCMEProto: the clip is smaller than one block");
      _ad.nBlkX = nbx;
      _ad.nBlkY = nby;
      _words = 3 + 3 * nbx * nby;
      _big_sad = 3 * bx * by * (1 << _ad.bits_per_pixel);

      _np.w = half ? (svi.width + 1) / 2 : svi.width;
      _np.h = half ? (svi.height + 1) / 2 : svi.height;
      _np.hevc = hevc;
      _np.preset = preset;
      std::string err; // open one session / estimator now: a configuration problem shows at script load
      if (_dx12)
      {
        auto *e = new dx12me::Estimator(); // check only; leaked on purpose (see GetFrame)
        if (!e->open(_np.w, _np.h, _dx12))
          env->ThrowError("NVENCMEProto: D3D12 ME: %s", e->err.c_str());
      }
      else
      {
        auto s = nvme::Pool::get().acquire(_np, err);
        if (!s)
          env->ThrowError("NVENCMEProto: %s", err.c_str());
        nvme::Pool::get().release(std::move(s));
      }

      vi.pixel_type = VideoInfo::CS_BGR32;
      vi.width = (HEADER + _words * 4 + 3) / 4;
      vi.height = 1;
      vi.audio_samples_per_second = 0;
      const uintptr_t p = reinterpret_cast<uintptr_t>(&_ad);
      vi.nchannels = (int)(0x80000000u | (unsigned)(p >> 32));
      vi.sample_type = (int)(p & 0xffffffffu);
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
      payload[0] = _words;
      payload[2] = _words - 2;
      const int nref = _isb ? n + _delta : n - _delta;
      if (nref < 0 || nref >= _src_vi.num_frames)
      {
        payload[1] = 0;
        for (int i = 3; i < _words; i += 3)
        {
          payload[i] = payload[i + 1] = 0;
          payload[i + 2] = _big_sad;
        }
        return dst;
      }
      std::vector<uint8_t> gc, gr, uc, ur;
      gray(n, gc, uc, env);
      gray(nref, gr, ur, env);
      std::string err;
      std::vector<nvme::Mv> grid;
      int gw = 0, gh = 0;
      bool ok;
      if (_dx12)
      {
        // one estimator per (thread, size, block size), stateless per call. Never destroyed: releasing D3D12 objects from
        // thread_local / static destructors runs under the loader lock at process exit and hung the process there.
        static std::mutex reg_m;
        static auto *reg = new std::map<std::tuple<std::thread::id, int, int, int>, dx12me::Estimator *>();
        dx12me::Estimator *e;
        {
          std::lock_guard<std::mutex> lk(reg_m);
          dx12me::Estimator *&slot = (*reg)[{ std::this_thread::get_id(), _np.w, _np.h, _dx12 }];
          if (!slot)
          {
            auto *ne = new dx12me::Estimator();
            if (!ne->open(_np.w, _np.h, _dx12))
            {
              err = ne->err; // leaked as well (partly created D3D12 objects)
              reg->erase({ std::this_thread::get_id(), _np.w, _np.h, _dx12 });
              env->ThrowError("NVENCMEProto: frame %d: D3D12 ME: %s", n, err.c_str());
            }
            slot = ne;
          }
          e = slot;
        }
        std::vector<dx12me::Mv> g;
        ok = e->run(gc.data(), gr.data(), uc.empty() ? nullptr : uc.data(), ur.empty() ? nullptr : ur.data(), g);
        if (!ok)
          err = e->err;
        grid.resize(g.size());
        for (size_t i = 0; i < g.size(); i++)
          grid[i] = { g[i].x, g[i].y };
        gw = e->gw;
        gh = e->gh;
      }
      else if (_levels > 1 || uc.empty())
        ok = nvme::pyramid_me(gc, gr, _np.w, _np.h, _np, _levels, grid, gw, gh, err); // chroma: ignored by the ME anyway
      else
      {
        auto s = nvme::Pool::get().acquire(_np, err);
        if (!s)
          env->ThrowError("NVENCMEProto: frame %d: %s", n, err.c_str());
        ok = s->run(gc.data(), gr.data(), _np.w, grid, nullptr, uc.data(), ur.data());
        gw = s->gw();
        gh = s->gh();
        if (!ok)
          err = s->err;
        nvme::Pool::get().release(std::move(s));
      }
      if (!ok)
        env->ThrowError("NVENCMEProto: frame %d: %s", n, err.c_str());

      // per block: lower median (x, y separately) of the grid cells under it, quarter pel (of the ME picture) -> 1/pel
      // of the source, clamped to the area MAnalyse's search reaches (as k_of_blocks)
      const int cell = (_dx12 ? _dx12 : 8) * (_half ? 2 : 1), scale = _half ? 2 : 1;
      const int bx = _ad.nBlkSizeX, by = _ad.nBlkSizeY, sx = bx - _ad.nOverlapX, sy = by - _ad.nOverlapY;
      const int ext_w = _src_vi.width + 2 * _ad.nHPadding, ext_h = _src_vi.height + 2 * _ad.nVPadding;
      std::vector<int> vxs, vys;
      auto to_pel = [&](int q) {
        const int num = q * scale * _pel, den = 4;
        return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
      };
      payload[1] = 1;
      for (int b = 0; b < _ad.nBlkX * _ad.nBlkY; b++)
      {
        const int xi = (b % _ad.nBlkX) * sx, yi = (b / _ad.nBlkX) * sy;
        const int gx0 = std::min(xi / cell, gw - 1), gx1 = std::min((xi + bx - 1) / cell, gw - 1);
        const int gy0 = std::min(yi / cell, gh - 1), gy1 = std::min((yi + by - 1) / cell, gh - 1);
        vxs.clear();
        vys.clear();
        for (int gy = gy0; gy <= gy1; gy++)
          for (int gx = gx0; gx <= gx1; gx++)
          {
            vxs.push_back(grid[(size_t)gy * gw + gx].x);
            vys.push_back(grid[(size_t)gy * gw + gx].y);
          }
        const size_t k = (vxs.size() - 1) / 2;
        std::nth_element(vxs.begin(), vxs.begin() + k, vxs.end());
        std::nth_element(vys.begin(), vys.begin() + k, vys.end());
        const int x0 = _ad.nHPadding + xi, y0 = _ad.nVPadding + yi;
        const int dx_min = -_pel * x0, dy_min = -_pel * y0;
        const int dx_max = _pel * (ext_w - x0 - bx), dy_max = _pel * (ext_h - y0 - by);
        payload[3 + 3 * b] = std::max(dx_min, std::min(dx_max - 1, to_pel(vxs[k])));
        payload[4 + 3 * b] = std::max(dy_min, std::min(dy_max - 1, to_pel(vys[k])));
        payload[5 + 3 * b] = 0;
      }
      return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int) override { return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0; }
  };

  AVSValue __cdecl Create(AVSValue args, void *, IScriptEnvironment *env)
  {
    PClip c = args[0].AsClip();
    const int bx = args[3].AsInt(16), by = args[4].AsInt(bx);
    const int ox = args[5].AsInt(0), oy = args[6].AsInt(ox);
    const int pel = args[7].AsInt(2);
    const int preset = args[11].AsInt(4);
    const char *codec = args[12].AsString("h264");
    if (pel != 1 && pel != 2)
      env->ThrowError("NVENCMEProto: pel must be 1 or 2");
    if (preset < 1 || preset > 7)
      env->ThrowError("NVENCMEProto: preset is NVENC's P1..P7");
    if (ox < 0 || ox > bx / 2 || oy < 0 || oy > by / 2)
      env->ThrowError("NVENCMEProto: overlap must be less or equal than half block size");
    const bool hevc = _stricmp(codec, "hevc") == 0;
    const int levels = args[14].AsInt(1);
    const int dx12 = args[15].AsInt(0);
    if (dx12 != 0 && dx12 != 8 && dx12 != 16)
      env->ThrowError("NVENCMEProto: dx12 is 0 (NVENC SDK), 8 or 16 (D3D12 ME block size)");
    if (dx12 && levels > 1)
      env->ThrowError("NVENCMEProto: levels > 1 needs the NVENC SDK (dx12=0)");
    if (levels < 1 || levels > 4 || (hevc && levels > 1))
      env->ThrowError("NVENCMEProto: levels is 1..4 (H.264 only above 1)");
    if (!hevc && _stricmp(codec, "h264") != 0)
      env->ThrowError("NVENCMEProto: codec is \"h264\" or \"hevc\"");
    return new NVENCMEProto(c, args[1].AsBool(false), args[2].AsInt(1), bx, by, ox, oy, pel, args[8].AsInt(16), args[9].AsInt(16),
      args[10].AsBool(true), preset, hevc, args[13].AsBool(false), levels, dx12, env);
  }
}

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
  AVS_linkage = vectors;
  env->AddFunction("NVENCMEProto", "c[isb]b[delta]i[blksize]i[blksizeV]i[overlap]i[overlapV]i[pel]i[hpad]i[vpad]i[chroma]b[preset]i[codec]s[half]b[levels]i[dx12]i",
    Create, nullptr);
  return "NVENC ME-only prototype";
}
