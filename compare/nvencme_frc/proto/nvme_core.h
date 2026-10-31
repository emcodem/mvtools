// NVENC ME-only prototype core (NVENCMEAnalyse.md, throwaway): CUDA driver API context + NVENC sessions in ME-only
// mode, 8-bit luma pairs in, vectors on an 8x8 grid in quarter pel out. Windows only. Every library is loaded at run
// time; a missing / broken one only makes available() false, it never crashes or hangs.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "nvEncodeAPI.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nvme
{
  typedef int CUresult;
  typedef int CUdevice;
  typedef void *CUcontext;

  struct Mv
  {
    int16_t x, y; // quarter pel, current -> reference
  };

  class Driver
  {
  public:
    bool ok = false;
    std::string err;
    NV_ENCODE_API_FUNCTION_LIST api = {};
    CUcontext ctx = nullptr;
    CUresult(__stdcall *cuCtxPushCurrent)(CUcontext) = nullptr;
    CUresult(__stdcall *cuCtxPopCurrent)(CUcontext *) = nullptr;

    static Driver &get()
    {
      static Driver d; // C++11: thread-safe, first use only (never from a global constructor)
      return d;
    }

  private:
    Driver()
    {
      DWORD old = 0;
      SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &old);
      HMODULE cu = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      HMODULE ne = LoadLibraryExW(L"nvEncodeAPI64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      SetThreadErrorMode(old, nullptr);
      if (!cu || !ne)
      {
        err = !cu ? "nvcuda.dll not found" : "nvEncodeAPI64.dll not found";
        return;
      }
      auto cuInit = (CUresult(__stdcall *)(unsigned))GetProcAddress(cu, "cuInit");
      auto cuDeviceGet = (CUresult(__stdcall *)(CUdevice *, int))GetProcAddress(cu, "cuDeviceGet");
      auto cuRetain = (CUresult(__stdcall *)(CUcontext *, CUdevice))GetProcAddress(cu, "cuDevicePrimaryCtxRetain");
      cuCtxPushCurrent = (decltype(cuCtxPushCurrent))GetProcAddress(cu, "cuCtxPushCurrent_v2");
      cuCtxPopCurrent = (decltype(cuCtxPopCurrent))GetProcAddress(cu, "cuCtxPopCurrent_v2");
      auto maxver = (NVENCSTATUS(NVENCAPI *)(uint32_t *))GetProcAddress(ne, "NvEncodeAPIGetMaxSupportedVersion");
      auto create = (NVENCSTATUS(NVENCAPI *)(NV_ENCODE_API_FUNCTION_LIST *))GetProcAddress(ne, "NvEncodeAPICreateInstance");
      if (!cuInit || !cuDeviceGet || !cuRetain || !cuCtxPushCurrent || !cuCtxPopCurrent || !maxver || !create)
      {
        err = "a CUDA driver or NVENC entry point is missing";
        return;
      }
      uint32_t v = 0;
      if (maxver(&v) != NV_ENC_SUCCESS || v < ((NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION))
      {
        err = "the driver's NVENC API is older than 13.0";
        return;
      }
      api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
      CUdevice dev = 0;
      if (create(&api) != NV_ENC_SUCCESS || cuInit(0) != 0 || cuDeviceGet(&dev, 0) != 0 || cuRetain(&ctx, dev) != 0)
      {
        err = "no NVIDIA device / CUDA context";
        return;
      }
      ok = true;
    }
  };

  inline bool env_on(const char *n)
  {
    const char *v = std::getenv(n);
    return v && std::atoi(v) != 0;
  }

  struct Params
  {
    int w = 0, h = 0;
    bool hevc = false;
    int preset = 4; // NVENC P1..P7
    int tuning = -1; // NV_ENC_TUNING_INFO, -1: from NVME_TUNING or 1
    int hints = 0;   // external hint candidates per 16x16 macroblock (H.264), 0: none
    bool operator==(const Params &o) const { return w == o.w && h == o.h && hevc == o.hevc && preset == o.preset && hints == o.hints; }
  };

  inline const GUID &preset_guid(int p)
  {
    static const GUID *g[7] = { &NV_ENC_PRESET_P1_GUID, &NV_ENC_PRESET_P2_GUID, &NV_ENC_PRESET_P3_GUID, &NV_ENC_PRESET_P4_GUID,
      &NV_ENC_PRESET_P5_GUID, &NV_ENC_PRESET_P6_GUID, &NV_ENC_PRESET_P7_GUID };
    return *g[std::min(std::max(p, 1), 7) - 1];
  }

  // statistics of the MV layout (probe): H.264 partition types, HEVC CU sizes
  struct Stats
  {
    long long part[4] = {}, intra = 0, same4_16x16 = 0, cu[4] = {}, hevc_units = 0, pm[8] = {}, cutype[4] = {}, last = 0;
    std::vector<std::string> dump;
  };

  class Session
  {
  public:
    Params p;
    std::string err;

    explicit Session(const Params &pp) : p(pp) {}
    ~Session() { close(); }

    bool open()
    {
      Driver &d = Driver::get();
      if (!d.ok)
      {
        err = d.err;
        return false;
      }
      const GUID codec = p.hevc ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
      NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op = {};
      op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
      op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
      op.device = d.ctx;
      op.apiVersion = NVENCAPI_VERSION;
      NVENCSTATUS s = d.api.nvEncOpenEncodeSessionEx(&op, &enc);
      if (s != NV_ENC_SUCCESS)
        return fail("nvEncOpenEncodeSessionEx", s);
      NV_ENC_CAPS_PARAM cp = {};
      cp.version = NV_ENC_CAPS_PARAM_VER;
      cp.capsToQuery = NV_ENC_CAPS_SUPPORT_MEONLY_MODE;
      int me = 0;
      if (d.api.nvEncGetEncodeCaps(enc, codec, &cp, &me) != NV_ENC_SUCCESS || !me)
        return fail("ME-only mode not supported for this codec", NV_ENC_ERR_UNSUPPORTED_DEVICE);
      NV_ENC_TUNING_INFO tuning = (NV_ENC_TUNING_INFO)(p.tuning >= 0 ? p.tuning : std::getenv("NVME_TUNING") ? std::atoi(std::getenv("NVME_TUNING")) : 1);
      NV_ENC_PRESET_CONFIG &pc = pc_;
      pc = {};
      pc.version = NV_ENC_PRESET_CONFIG_VER;
      pc.presetCfg.version = NV_ENC_CONFIG_VER;
      s = d.api.nvEncGetEncodePresetConfigEx(enc, codec, preset_guid(p.preset), tuning, &pc);
      if (s != NV_ENC_SUCCESS)
        return fail("nvEncGetEncodePresetConfigEx", s);
      pc.presetCfg.frameIntervalP = 1; // no B frames in ME-only mode
      pc.presetCfg.gopLength = NVENC_INFINITE_GOPLENGTH;
      // ME-only mode reads the codec union as NV_ENC_CONFIG_H264_MEONLY / _HEVC_MEONLY: clear what the encoder preset
      // put there (it would be read as "disable partition" / "disable intra" bits) and set the ME-only fields
      std::memset(&pc.presetCfg.encodeCodecConfig, 0, sizeof(pc.presetCfg.encodeCodecConfig));
      if (!p.hevc && std::getenv("NVME_LEGACY_UNION") == nullptr)
      {
        NV_ENC_CONFIG_H264_MEONLY &mo = pc.presetCfg.encodeCodecConfig.h264MeOnlyConfig;
        mo.disableIntraSearch = env_on("NVME_NOINTRA");
        if (env_on("NVME_P16")) // 16x16 partitions only
          mo.disablePartition8x16 = mo.disablePartition16x8 = mo.disablePartition8x8 = 1;
      }
      else if (p.hevc)
        pc.presetCfg.encodeCodecConfig.hevcMeOnlyConfig = {};
      NV_ENC_INITIALIZE_PARAMS &ip = ip_;
      ip = {};
      ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
      ip.encodeGUID = codec;
      ip.presetGUID = preset_guid(p.preset);
      ip.encodeWidth = ip.darWidth = ip.maxEncodeWidth = p.w;
      ip.encodeHeight = ip.darHeight = ip.maxEncodeHeight = p.h;
      ip.frameRateNum = 30;
      ip.frameRateDen = 1;
      ip.enablePTD = 1;
      ip.enableMEOnlyMode = 1;
      ip.encodeConfig = &pc.presetCfg;
      if (p.hints > 0 && !p.hevc)
      {
        ip.enableExternalMEHints = 1;
        ip.maxMEHintCountsPerBlock[0].numCandsPerBlk16x16 = p.hints;
      }
      ip.tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED; // ME-only mode rejects every tuning; the preset config above still uses one
      s = d.api.nvEncInitializeEncoder(enc, &ip);
      if (s != NV_ENC_SUCCESS)
        return fail("nvEncInitializeEncoder", s);
      for (int i = 0; i < 2; i++)
      {
        NV_ENC_CREATE_INPUT_BUFFER ib = {};
        ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        ib.width = p.w;
        ib.height = p.h;
        ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
        s = d.api.nvEncCreateInputBuffer(enc, &ib);
        if (s != NV_ENC_SUCCESS)
          return fail("nvEncCreateInputBuffer", s);
        in[i] = ib.inputBuffer;
        chroma_set[i] = false;
      }
      NV_ENC_CREATE_MV_BUFFER mb = {};
      mb.version = NV_ENC_CREATE_MV_BUFFER_VER;
      s = d.api.nvEncCreateMVBuffer(enc, &mb);
      if (s != NV_ENC_SUCCESS)
        return fail("nvEncCreateMVBuffer", s);
      mv = mb.mvBuffer;
      return true;
    }

    // cur / ref: 8-bit luma, w x h, row pitch `pitch`. out: vectors on the 8x8 grid ((w+15)/16*2 x (h+15)/16*2 for
    // H.264; (w+7)/8 x (h+7)/8 for HEVC), row-major, width gw().
    // cur_uv / ref_uv (optional): NV12 chroma, (h + 1) / 2 rows of w bytes (U, V interleaved); null: neutral grey
    bool run(const uint8_t *cur, const uint8_t *ref, int pitch, std::vector<Mv> &out, Stats *st = nullptr,
      const uint8_t *cur_uv = nullptr, const uint8_t *ref_uv = nullptr, const NVENC_EXTERNAL_ME_HINT *hints = nullptr)
    {
      Driver &d = Driver::get();
      // the session keeps state between calls (the previous vector field serves as predictors, and
      // nvEncReconfigureEncoder is not implemented in ME-only mode): with `fresh` every call gets a new session, so
      // the result depends on this pair only, not on the order in which the threads call
      if (fresh && used)
      {
        close();
        if (!open())
          return false;
      }
      used = true;
      if (d.cuCtxPushCurrent(d.ctx) != 0)
        return fail("cuCtxPushCurrent", NV_ENC_ERR_GENERIC);
      bool ok = upload(0, cur, pitch, cur_uv) && upload(1, ref, pitch, ref_uv);
      if (ok)
      {
        NV_ENC_MEONLY_PARAMS mp = {};
        mp.version = NV_ENC_MEONLY_PARAMS_VER;
        mp.inputWidth = p.w;
        mp.inputHeight = p.h;
        mp.inputBuffer = in[0];
        mp.referenceFrame = in[1];
        mp.mvBuffer = mv;
        mp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
        if (hints && p.hints > 0)
        {
          mp.meHintCountsPerBlock[0].numCandsPerBlk16x16 = p.hints;
          mp.meExternalHints = const_cast<NVENC_EXTERNAL_ME_HINT *>(hints);
        }
        NVENCSTATUS s = d.api.nvEncRunMotionEstimationOnly(enc, &mp);
        ok = s == NV_ENC_SUCCESS || fail("nvEncRunMotionEstimationOnly", s);
      }
      if (ok)
      {
        NV_ENC_LOCK_BITSTREAM lb = {};
        lb.version = NV_ENC_LOCK_BITSTREAM_VER;
        lb.outputBitstream = mv;
        NVENCSTATUS s = d.api.nvEncLockBitstream(enc, &lb);
        ok = s == NV_ENC_SUCCESS || fail("nvEncLockBitstream", s);
        if (ok)
        {
          ok = p.hevc ? decode_hevc((const NV_ENC_HEVC_MV_DATA *)lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes, out, st)
                      : decode_h264((const NV_ENC_H264_MV_DATA *)lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes, out, st);
          d.api.nvEncUnlockBitstream(enc, mv);
        }
      }
      CUcontext dummy;
      d.cuCtxPopCurrent(&dummy);
      return ok;
    }

    int gw() const { return p.hevc ? (p.w + 7) / 8 : (p.w + 15) / 16 * 2; }
    int gh() const { return p.hevc ? (p.h + 7) / 8 : (p.h + 15) / 16 * 2; }

    bool fresh = !std::getenv("NVME_FRESH") || std::atoi(std::getenv("NVME_FRESH")) != 0;

  private:
    NV_ENC_PRESET_CONFIG pc_ = {};
    NV_ENC_INITIALIZE_PARAMS ip_ = {};
    void *enc = nullptr;
    bool used = false;
    NV_ENC_INPUT_PTR in[2] = {};
    bool chroma_set[2] = {};
    NV_ENC_OUTPUT_PTR mv = nullptr;

    bool fail(const char *what, NVENCSTATUS s)
    {
      char b[256];
      const char *le = enc ? Driver::get().api.nvEncGetLastErrorString(enc) : nullptr;
      std::snprintf(b, sizeof b, "%s failed (NVENCSTATUS %d%s%s)", what, (int)s, le && *le ? ": " : "", le ? le : "");
      err = b;
      return false;
    }

    void close()
    {
      if (!enc)
        return;
      Driver &d = Driver::get();
      if (mv)
        d.api.nvEncDestroyMVBuffer(enc, mv);
      for (auto &b : in)
        if (b)
          d.api.nvEncDestroyInputBuffer(enc, b);
      d.api.nvEncDestroyEncoder(enc);
      enc = nullptr;
      mv = nullptr;
      in[0] = in[1] = nullptr;
    }

    bool upload(int i, const uint8_t *src, int pitch, const uint8_t *uv)
    {
      Driver &d = Driver::get();
      NV_ENC_LOCK_INPUT_BUFFER lk = {};
      lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
      lk.inputBuffer = in[i];
      NVENCSTATUS s = d.api.nvEncLockInputBuffer(enc, &lk);
      if (s != NV_ENC_SUCCESS)
        return fail("nvEncLockInputBuffer", s);
      uint8_t *dst = (uint8_t *)lk.bufferDataPtr;
      for (int y = 0; y < p.h; y++)
        std::memcpy(dst + (size_t)y * lk.pitch, src + (size_t)y * pitch, p.w);
      if (uv)
      {
        for (int y = 0; y < (p.h + 1) / 2; y++)
          std::memcpy(dst + (size_t)(p.h + y) * lk.pitch, uv + (size_t)y * p.w, p.w);
        chroma_set[i] = false;
      }
      else if (!chroma_set[i]) // neutral chroma, once per buffer
      {
        for (int y = 0; y < (p.h + 1) / 2; y++)
          std::memset(dst + (size_t)(p.h + y) * lk.pitch, 128, p.w);
        chroma_set[i] = true;
      }
      d.api.nvEncUnlockInputBuffer(enc, in[i]);
      return true;
    }

    bool decode_h264(const NV_ENC_H264_MV_DATA *md, uint32_t bytes, std::vector<Mv> &out, Stats *st)
    {
      const int mbw = (p.w + 15) / 16, mbh = (p.h + 15) / 16, gwid = mbw * 2;
      if (bytes < (uint32_t)(mbw * mbh) * sizeof(NV_ENC_H264_MV_DATA))
        return fail("MV buffer smaller than expected", NV_ENC_ERR_GENERIC);
      out.resize((size_t)gwid * mbh * 2);
      for (int my = 0; my < mbh; my++)
        for (int mx = 0; mx < mbw; mx++)
        {
          const NV_ENC_H264_MV_DATA &m = md[my * mbw + mx];
          if (st)
          {
            st->part[m.partitionType & 3]++;
            st->intra += m.mbType == 0 || m.mbType == 2;
            if (m.partitionType == 0)
              st->same4_16x16 += (m.mv[0].mvx == m.mv[1].mvx && m.mv[0].mvy == m.mv[1].mvy && m.mv[0].mvx == m.mv[3].mvx && m.mv[0].mvy == m.mv[3].mvy);
          }
          for (int q = 0; q < 4; q++)
          {
            const int qx = q & 1, qy = q >> 1;
            int k;
            switch (m.partitionType)
            {
            case 1: k = q; break;  // 8x8: raster
            case 2: k = qy; break; // 16x8: top, bottom
            case 3: k = qx; break; // 8x16: left, right
            default: k = 0; break; // 16x16
            }
            out[(size_t)(my * 2 + qy) * gwid + mx * 2 + qx] = { m.mv[k].mvx, m.mv[k].mvy };
          }
        }
      return true;
    }

    // CUs in CTB raster order (32x32 CTBs), inside a CTB in z-order; CUs wholly outside the picture are not coded
    bool decode_hevc(const NV_ENC_HEVC_MV_DATA *md, uint32_t bytes, std::vector<Mv> &out, Stats *st)
    {
      const int gwid = (p.w + 7) / 8, ghei = (p.h + 7) / 8;
      const int ctbw = (p.w + 31) / 32, ctbh = (p.h + 31) / 32;
      const size_t count = bytes / sizeof(NV_ENC_HEVC_MV_DATA);
      out.assign((size_t)gwid * ghei, Mv{ 0, 0 });
      size_t i = 0;
      for (int cy = 0; cy < ctbh; cy++)
        for (int cx = 0; cx < ctbw; cx++)
        {
          int z = 0; // z-order index of the next 8x8 unit inside the CTB (16 units)
          for (;;)
          {
            auto ux = [](int zz) { return (zz & 1) | ((zz >> 1) & 2); };
            auto uy = [](int zz) { return ((zz >> 1) & 1) | ((zz >> 2) & 2); };
            while (z < 16 && (cx * 32 + ux(z) * 8 >= p.w || cy * 32 + uy(z) * 8 >= p.h))
              z++; // outside the picture: not coded
            if (z >= 16)
              break;
            if (i >= count)
              return fail("HEVC MV buffer ended early", NV_ENC_ERR_GENERIC);
            const NV_ENC_HEVC_MV_DATA &m = md[i++];
            const int s = 8 << std::min<int>(m.cuSize, 2), units = (s / 8) * (s / 8);
            if (st)
            {
              st->cu[m.cuSize & 3]++;
              st->hevc_units += units;
              st->pm[m.partitionMode & 7]++;
              st->cutype[m.cuType & 3]++;
              st->last += m.lastCUInCTB != 0;
              if (st->dump.size() < 24)
              {
                char b[160];
                std::snprintf(b, sizeof b, "ctb %d,%d z %d: size %d type %d pm %d last %d mv (%d,%d) (%d,%d) (%d,%d) (%d,%d)", cx, cy, z, m.cuSize, m.cuType,
                  m.partitionMode, m.lastCUInCTB, m.mv[0].mvx, m.mv[0].mvy, m.mv[1].mvx, m.mv[1].mvy, m.mv[2].mvx, m.mv[2].mvy, m.mv[3].mvx, m.mv[3].mvy);
                st->dump.push_back(b);
              }
            }
            const int x0 = cx * 32 + ux(z) * 8, y0 = cy * 32 + uy(z) * 8;
            for (int sy = 0; sy < s; sy += 8)
              for (int sx = 0; sx < s; sx += 8)
              {
                const int px = sx + 4, py = sy + 4;
                int k = 0;
                if (s > 8)
                  switch (m.partitionMode)
                  {
                  case 1: k = py >= s / 2; break;
                  case 2: k = px >= s / 2; break;
                  case 3: k = (py >= s / 2) * 2 + (px >= s / 2); break;
                  case 4: k = py >= s / 4; break;
                  case 5: k = py >= 3 * s / 4; break;
                  case 6: k = px >= s / 4; break;
                  case 7: k = px >= 3 * s / 4; break;
                  default: k = 0; break;
                  }
                const int gx = (x0 + sx) / 8, gy = (y0 + sy) / 8;
                if (gx < gwid && gy < ghei)
                  out[(size_t)gy * gwid + gx] = { m.mv[k].mvx, m.mv[k].mvy };
              }
            z += units;
            if (m.lastCUInCTB || z >= 16)
              break;
          }
        }
      return true;
    }
  };

  // Sessions per Params, at most `cap` open at once; acquire waits for a free one (bounded by the running ME calls).
  class Pool
  {
  public:
    static Pool &get()
    {
      static Pool *p = new Pool; // never destroyed: no NVENC calls from static destructors at process exit
      return *p;
    }

    std::unique_ptr<Session> acquire(const Params &p, std::string &err)
    {
      std::unique_lock<std::mutex> lk(m);
      for (;;)
      {
        for (size_t i = 0; i < idle.size(); i++)
          if (idle[i]->p == p)
          {
            auto s = std::move(idle[i]);
            idle.erase(idle.begin() + i);
            return s;
          }
        if (open_count < cap)
          break;
        if (!idle.empty()) // other parameters: close one to make room
        {
          idle.erase(idle.begin());
          open_count--;
          continue;
        }
        cv.wait(lk);
      }
      open_count++;
      lk.unlock();
      auto s = std::make_unique<Session>(p);
      if (!s->open())
      {
        err = s->err;
        std::lock_guard<std::mutex> g(m);
        open_count--;
        cv.notify_one();
        return nullptr;
      }
      return s;
    }

    void release(std::unique_ptr<Session> s)
    {
      std::lock_guard<std::mutex> g(m);
      idle.push_back(std::move(s));
      cv.notify_one();
    }

  private:
    Pool()
    {
      const char *e = std::getenv("NVME_SESSIONS");
      cap = e ? std::max(1, std::atoi(e)) : 2;
    }
    std::mutex m;
    std::condition_variable cv;
    std::vector<std::unique_ptr<Session>> idle;
    int open_count = 0, cap = 2;
  };
  // 2x2 box downscale of an 8-bit picture
  inline void downscale(const std::vector<uint8_t> &src, int w, int h, std::vector<uint8_t> &dst, int &dw, int &dh)
  {
    dw = (w + 1) / 2;
    dh = (h + 1) / 2;
    dst.resize((size_t)dw * dh);
    for (int y = 0; y < dh; y++)
      for (int x = 0; x < dw; x++)
      {
        const int x0 = 2 * x, x1 = std::min(2 * x + 1, w - 1), y0 = 2 * y, y1 = std::min(2 * y + 1, h - 1);
        dst[(size_t)y * dw + x] = (uint8_t)((src[(size_t)y0 * w + x0] + src[(size_t)y0 * w + x1] + src[(size_t)y1 * w + x0]
          + src[(size_t)y1 * w + x1] + 2) >> 2);
      }
  }

  // Hints for the macroblocks of a picture twice the size of the coarse one: per MB the coarse vector under its centre
  // and the median of the 3x3 coarse cells around it, in integer pixels of the fine picture (2 candidates).
  inline void hints_from(const std::vector<Mv> &cg, int cgw, int cgh, int fw, int fh, std::vector<NVENC_EXTERNAL_ME_HINT> &out)
  {
    const int mbw = (fw + 15) / 16, mbh = (fh + 15) / 16;
    out.assign((size_t)mbw * mbh * 2, NVENC_EXTERNAL_ME_HINT{});
    auto conv = [](int q) { // coarse quarter pel -> fine integer pel: q * 2 / 4, rounded half away from zero
      return q >= 0 ? (q + 1) / 2 : -((-q + 1) / 2);
    };
    int vx[9], vy[9];
    for (int my = 0; my < mbh; my++)
      for (int mx = 0; mx < mbw; mx++)
      {
        const int gx = std::min((mx * 16 + 8) / 2 / 8, cgw - 1), gy = std::min((my * 16 + 8) / 2 / 8, cgh - 1);
        int n = 0;
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++)
          {
            const int x = std::min(std::max(gx + dx, 0), cgw - 1), y = std::min(std::max(gy + dy, 0), cgh - 1);
            vx[n] = cg[(size_t)y * cgw + x].x;
            vy[n++] = cg[(size_t)y * cgw + x].y;
          }
        std::nth_element(vx, vx + 4, vx + 9);
        std::nth_element(vy, vy + 4, vy + 9);
        const Mv c = cg[(size_t)gy * cgw + gx];
        const int cand[2][2] = { { conv(c.x), conv(c.y) }, { conv(vx[4]), conv(vy[4]) } };
        for (int k = 0; k < 2; k++)
        {
          NVENC_EXTERNAL_ME_HINT &h = out[((size_t)my * mbw + mx) * 2 + k];
          h.mvx = std::min(std::max(cand[k][0], -2047), 2047);
          h.mvy = std::min(std::max(cand[k][1], -511), 511);
          h.refidx = 0;
          h.dir = 0;
          h.partType = 0;
          h.lastofPart = 1;
          h.lastOfMB = k == 1;
        }
      }
  }

  // ME of (cur, ref) (8-bit luma, w x h, pitch w) through a pyramid of `levels` (1: plain); each finer level gets the
  // coarser level's vectors as hints. Result: the 8x8 grid of the full picture, quarter pel. H.264 only.
  inline bool pyramid_me(const std::vector<uint8_t> &cur, const std::vector<uint8_t> &ref, int w, int h, Params base, int levels,
    std::vector<Mv> &out, int &gw, int &gh, std::string &err)
  {
    std::vector<std::vector<uint8_t>> pc(1, cur), pr(1, ref);
    std::vector<int> ws(1, w), hs(1, h);
    for (int l = 1; l < levels && ws.back() >= 64 && hs.back() >= 64; l++)
    {
      std::vector<uint8_t> a, b;
      int dw, dh;
      downscale(pc.back(), ws.back(), hs.back(), a, dw, dh);
      downscale(pr.back(), ws.back(), hs.back(), b, dw, dh);
      pc.push_back(std::move(a));
      pr.push_back(std::move(b));
      ws.push_back(dw);
      hs.push_back(dh);
    }
    std::vector<Mv> grid;
    std::vector<NVENC_EXTERNAL_ME_HINT> hints;
    int cgw = 0, cgh = 0;
    for (int l = (int)ws.size() - 1; l >= 0; l--)
    {
      Params p = base;
      p.w = ws[l];
      p.h = hs[l];
      p.hints = l + 1 < (int)ws.size() ? 2 : 0;
      if (p.hints)
        hints_from(grid, cgw, cgh, p.w, p.h, hints);
      auto s = Pool::get().acquire(p, err);
      if (!s)
        return false;
      const bool ok = s->run(pc[l].data(), pr[l].data(), p.w, grid, nullptr, nullptr, nullptr, p.hints ? hints.data() : nullptr);
      cgw = s->gw();
      cgh = s->gh();
      if (!ok)
        err = s->err;
      Pool::get().release(std::move(s));
      if (!ok)
        return false;
    }
    out.swap(grid);
    gw = cgw;
    gh = cgh;
    return true;
  }
}
