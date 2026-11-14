// NVENC motion-estimation-only sessions (mvcuda_nvenc.h). The SDK header only declares types; the driver's encoder
// library is loaded at run time, so mvtools_cuda gets no new link dependency and works without it.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include "mvcuda_nvenc.h"

#ifdef __NVCC__
#pragma nv_diag_suppress 108 // signed one-bit fields in the SDK header
#endif
#include "third_party/nvenc/nvEncodeAPI.h"

#include "dyn_lib.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace mvc_nvenc
{
  namespace
  {
    struct Priv
    {
      void *enc = nullptr;
      NV_ENC_REGISTERED_PTR reg[9] = {};
      NV_ENC_OUTPUT_PTR mv = nullptr;
      bool used = false; // the encoder has run since it was opened
      bool busy = false;
      std::vector<NVENC_EXTERNAL_ME_HINT> hints;
    };

    std::mutex g_mutex;
    std::condition_variable g_cv;
    NV_ENCODE_API_FUNCTION_LIST g_f;
    int g_loaded = 0; // 0 not tried, 1 ok, -1 failed
    std::string g_load_err;
    std::vector<Session *> g_pool;
    int g_open = 0; // sessions with an open encoder or being opened

    typedef CUresult(CUDAAPI *PFN_ctx_get)(CUcontext *);
    PFN_ctx_get g_ctx_get = nullptr;

    int pool_cap()
    {
      static const int cap = [] {
        const char *e = std::getenv("MVTOOLS_NVENC_SESSIONS");
        const int v = e ? std::atoi(e) : 0;
        return v > 0 ? v : 4;
      }();
      return cap;
    }

    bool load_locked()
    {
      if (g_loaded)
        return g_loaded > 0;
      g_loaded = -1;
      void *m = dynlib::open(dynlib::nvenc_lib_name);
      if (!m)
      {
        g_load_err = std::string(dynlib::nvenc_lib_name) + " not found (" + dynlib::last_error() + ")";
        return false;
      }
      typedef NVENCSTATUS(NVENCAPI * PFN_max)(uint32_t *);
      typedef NVENCSTATUS(NVENCAPI * PFN_create)(NV_ENCODE_API_FUNCTION_LIST *);
      auto maxv = (PFN_max)dynlib::symbol(m, "NvEncodeAPIGetMaxSupportedVersion");
      auto create = (PFN_create)dynlib::symbol(m, "NvEncodeAPICreateInstance");
      uint32_t v = 0;
      if (!maxv || !create || maxv(&v) != NV_ENC_SUCCESS)
      {
        g_load_err = "the NVENC library has no usable entry points";
        return false;
      }
      // driver version: major << 4 | minor; the header's NVENCAPI_VERSION: major | minor << 24
      if ((int)(v >> 4) < NVENCAPI_MAJOR_VERSION || ((int)(v >> 4) == NVENCAPI_MAJOR_VERSION && (int)(v & 15) < NVENCAPI_MINOR_VERSION))
      {
        char b[160];
        std::snprintf(b, sizeof b, "the driver supports NVENC API %u.%u, %d.%d is needed (update the NVIDIA driver)", v >> 4, v & 15,
          NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
        g_load_err = b;
        return false;
      }
      std::memset(&g_f, 0, sizeof g_f);
      g_f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
      cudaDriverEntryPointQueryResult q;
      if (create(&g_f) != NV_ENC_SUCCESS
        || cudaGetDriverEntryPointByVersion("cuCtxGetCurrent", (void **)&g_ctx_get, 12000, cudaEnableDefault, &q) != cudaSuccess
        || !g_ctx_get)
      {
        g_load_err = "NvEncodeAPICreateInstance / cuCtxGetCurrent failed";
        return false;
      }
      g_loaded = 1;
      return true;
    }

    std::string status(void *enc, NVENCSTATUS s, const char *what)
    {
      const char *le = enc ? g_f.nvEncGetLastErrorString(enc) : nullptr;
      char b[384];
      std::snprintf(b, sizeof b, "%s failed (NVENCSTATUS %d%s%s)", what, (int)s, le && *le ? ": " : "", le ? le : "");
      return b;
    }

    GUID preset_guid(int p)
    {
      static const GUID g[7] = { NV_ENC_PRESET_P1_GUID, NV_ENC_PRESET_P2_GUID, NV_ENC_PRESET_P3_GUID, NV_ENC_PRESET_P4_GUID,
        NV_ENC_PRESET_P5_GUID, NV_ENC_PRESET_P6_GUID, NV_ENC_PRESET_P7_GUID };
      return g[std::min(std::max(p, 1), 7) - 1];
    }

    void close_enc(Session *s)
    {
      Priv *p = (Priv *)s->priv;
      if (!p->enc)
        return;
      for (auto &r : p->reg)
        if (r)
        {
          g_f.nvEncUnregisterResource(p->enc, r);
          r = nullptr;
        }
      if (p->mv)
        g_f.nvEncDestroyMVBuffer(p->enc, p->mv);
      p->mv = nullptr;
      g_f.nvEncDestroyEncoder(p->enc);
      p->enc = nullptr;
    }

    bool open_enc(Session *s, std::string &err)
    {
      Priv *p = (Priv *)s->priv;
      const Config &c = s->cfg;
      CUcontext ctx = nullptr;
      g_ctx_get(&ctx);
      if (!ctx)
      {
        err = "no current CUDA context";
        return false;
      }
      const GUID codec = c.hevc ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
      NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op = {};
      op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
      op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
      op.device = ctx;
      op.apiVersion = NVENCAPI_VERSION;
      NVENCSTATUS r = g_f.nvEncOpenEncodeSessionEx(&op, &p->enc);
      if (r != NV_ENC_SUCCESS)
      {
        err = status(nullptr, r, "nvEncOpenEncodeSessionEx");
        p->enc = nullptr;
        return false;
      }
      NV_ENC_CAPS_PARAM cp = {};
      cp.version = NV_ENC_CAPS_PARAM_VER;
      cp.capsToQuery = NV_ENC_CAPS_SUPPORT_MEONLY_MODE;
      int me = 0;
      if (g_f.nvEncGetEncodeCaps(p->enc, codec, &cp, &me) != NV_ENC_SUCCESS || !me)
      {
        err = std::string("this GPU has no motion-estimation-only mode for ") + (c.hevc ? "HEVC" : "H.264");
        close_enc(s);
        return false;
      }
      NV_ENC_PRESET_CONFIG pc = {};
      pc.version = NV_ENC_PRESET_CONFIG_VER;
      pc.presetCfg.version = NV_ENC_CONFIG_VER;
      if ((r = g_f.nvEncGetEncodePresetConfigEx(p->enc, codec, preset_guid(c.preset), NV_ENC_TUNING_INFO_HIGH_QUALITY, &pc)) != NV_ENC_SUCCESS)
      {
        err = status(p->enc, r, "nvEncGetEncodePresetConfigEx");
        close_enc(s);
        return false;
      }
      pc.presetCfg.frameIntervalP = 1; // no B frames in ME-only mode
      pc.presetCfg.gopLength = NVENC_INFINITE_GOPLENGTH;
      if (c.qp >= 0)
      {
        pc.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        pc.presetCfg.rcParams.constQP.qpInterP = pc.presetCfg.rcParams.constQP.qpInterB = pc.presetCfg.rcParams.constQP.qpIntra = (uint32_t)c.qp;
      }
      // ME-only mode reads the codec union as NV_ENC_CONFIG_H264_MEONLY / _HEVC_MEONLY: clear what the preset put there
      std::memset(&pc.presetCfg.encodeCodecConfig, 0, sizeof(pc.presetCfg.encodeCodecConfig));
      if (!c.hevc)
      {
        NV_ENC_CONFIG_H264_MEONLY &mo = pc.presetCfg.encodeCodecConfig.h264MeOnlyConfig;
        mo.disableIntraSearch = c.no_intra ? 1 : 0;
        if (c.p16)
          mo.disablePartition8x16 = mo.disablePartition16x8 = mo.disablePartition8x8 = 1;
      }
      NV_ENC_INITIALIZE_PARAMS ip = {};
      ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
      ip.encodeGUID = codec;
      ip.presetGUID = preset_guid(c.preset);
      ip.encodeWidth = ip.darWidth = ip.maxEncodeWidth = (uint32_t)c.width;
      ip.encodeHeight = ip.darHeight = ip.maxEncodeHeight = (uint32_t)c.height;
      ip.frameRateNum = 30;
      ip.frameRateDen = 1;
      ip.enablePTD = 1;
      ip.enableMEOnlyMode = 1;
      ip.encodeConfig = &pc.presetCfg;
      ip.tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED; // ME-only mode rejects every tuning; the preset config above used one
      if (c.hints > 0 && !c.hevc)
      {
        ip.enableExternalMEHints = 1;
        ip.maxMEHintCountsPerBlock[0].numCandsPerBlk16x16 = (uint32_t)c.hints;
      }
      if ((r = g_f.nvEncInitializeEncoder(p->enc, &ip)) != NV_ENC_SUCCESS)
      {
        err = status(p->enc, r, "nvEncInitializeEncoder");
        close_enc(s);
        return false;
      }
      for (int i = 0; i <= c.slots; i++) // the last one: the flat picture of reset runs
      {
        NV_ENC_REGISTER_RESOURCE rr = {};
        rr.version = NV_ENC_REGISTER_RESOURCE_VER;
        rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        rr.width = (uint32_t)c.width;
        rr.height = (uint32_t)c.height;
        rr.pitch = (uint32_t)s->pitch;
        rr.resourceToRegister = s->pic[i];
        rr.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
        rr.bufferUsage = NV_ENC_INPUT_IMAGE;
        if ((r = g_f.nvEncRegisterResource(p->enc, &rr)) != NV_ENC_SUCCESS)
        {
          err = status(p->enc, r, "nvEncRegisterResource");
          close_enc(s);
          return false;
        }
        p->reg[i] = rr.registeredResource;
      }
      NV_ENC_CREATE_MV_BUFFER mb = {};
      mb.version = NV_ENC_CREATE_MV_BUFFER_VER;
      if ((r = g_f.nvEncCreateMVBuffer(p->enc, &mb)) != NV_ENC_SUCCESS)
      {
        err = status(p->enc, r, "nvEncCreateMVBuffer");
        close_enc(s);
        return false;
      }
      p->mv = mb.mvBuffer;
      p->used = false;
      return true;
    }

    void destroy(Session *s)
    {
      close_enc(s);
      for (auto &b : s->pic)
        if (b)
          cudaFree(b);
      delete (Priv *)s->priv;
      delete s;
    }

    Session *create(const Config &c, std::string &err)
    {
      Session *s = new Session();
      s->cfg = c;
      s->priv = new Priv();
      s->gw = c.hevc ? (c.width + 7) / 8 : (c.width + 15) / 16 * 2;
      s->gh = c.hevc ? (c.height + 7) / 8 : (c.height + 15) / 16 * 2;
      for (int i = 0; i <= c.slots; i++) // slot c.slots: flat grey (reset runs)
      {
        void *b = nullptr;
        size_t pitch = 0;
        if (cudaMallocPitch(&b, &pitch, (size_t)c.width, (size_t)c.height * 3 / 2) != cudaSuccess
          || cudaMemset2D((uint8_t *)b + (i == c.slots ? 0 : pitch * c.height), pitch, 128, (size_t)c.width,
               (size_t)(i == c.slots ? c.height * 3 / 2 : c.height / 2)) != cudaSuccess)
        {
          cudaGetLastError();
          if (b)
            cudaFree(b);
          err = "out of device memory for the NVENC pictures";
          destroy(s);
          return nullptr;
        }
        s->pic[i] = (uint8_t *)b;
        s->pitch = pitch;
      }
      if (cudaDeviceSynchronize() != cudaSuccess || !open_enc(s, err))
      {
        if (err.empty())
          err = "CUDA error while preparing the NVENC pictures";
        destroy(s);
        return nullptr;
      }
      return s;
    }

    // H.264: four 8x8 vectors per macroblock, by partition type
    bool decode_h264(const Session *s, const NV_ENC_H264_MV_DATA *md, uint32_t bytes, Mv *out)
    {
      const int mbw = (s->cfg.width + 15) / 16, mbh = (s->cfg.height + 15) / 16, gw = mbw * 2;
      if (bytes < (uint32_t)(mbw * mbh) * sizeof(NV_ENC_H264_MV_DATA))
        return false;
      for (int my = 0; my < mbh; my++)
        for (int mx = 0; mx < mbw; mx++)
        {
          const NV_ENC_H264_MV_DATA &m = md[my * mbw + mx];
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
            out[(size_t)(my * 2 + qy) * gw + mx * 2 + qx] = { m.mv[k].mvx, m.mv[k].mvy };
          }
        }
      return true;
    }

    // HEVC: CUs in CTB raster order (32x32 CTBs), inside a CTB in z-order; CUs wholly outside the picture are not coded
    bool decode_hevc(const Session *s, const NV_ENC_HEVC_MV_DATA *md, uint32_t bytes, Mv *out)
    {
      const int w = s->cfg.width, h = s->cfg.height, gw = s->gw, gh = s->gh;
      const int ctbw = (w + 31) / 32, ctbh = (h + 31) / 32;
      const size_t count = bytes / sizeof(NV_ENC_HEVC_MV_DATA);
      std::fill(out, out + (size_t)gw * gh, Mv{ 0, 0 });
      size_t i = 0;
      auto ux = [](int z) { return (z & 1) | ((z >> 1) & 2); };
      auto uy = [](int z) { return ((z >> 1) & 1) | ((z >> 2) & 2); };
      for (int cy = 0; cy < ctbh; cy++)
        for (int cx = 0; cx < ctbw; cx++)
        {
          int z = 0; // z-order index of the next 8x8 unit inside the CTB (16 units)
          for (;;)
          {
            while (z < 16 && (cx * 32 + ux(z) * 8 >= w || cy * 32 + uy(z) * 8 >= h))
              z++;
            if (z >= 16)
              break;
            if (i >= count)
              return false;
            const NV_ENC_HEVC_MV_DATA &m = md[i++];
            const int sz = 8 << std::min<int>(m.cuSize, 2), units = (sz / 8) * (sz / 8);
            const int bx = cx * 4 + ux(z), by = cy * 4 + uy(z), side = sz / 8;
            for (int yy = 0; yy < side; yy++)
              for (int xx = 0; xx < side; xx++)
                if (bx + xx < gw && by + yy < gh)
                  out[(size_t)(by + yy) * gw + bx + xx] = { m.mv[0].mvx, m.mv[0].mvy };
            z += units;
          }
        }
      return true;
    }
  }

  Session *acquire(const Config &c, std::string &err)
  {
    std::unique_lock<std::mutex> lk(g_mutex);
    if (!load_locked())
    {
      err = g_load_err;
      return nullptr;
    }
    for (;;)
    {
      for (Session *s : g_pool)
      {
        Priv *p = (Priv *)s->priv;
        if (!p->busy && s->cfg == c)
        {
          p->busy = true;
          return s;
        }
      }
      if (g_open < pool_cap())
        break;
      // the cap is reached: free an idle session of another configuration, else wait for one
      for (size_t i = 0; i < g_pool.size(); i++)
      {
        Session *s = g_pool[i];
        if (!((Priv *)s->priv)->busy)
        {
          g_pool.erase(g_pool.begin() + (std::ptrdiff_t)i);
          g_open--;
          destroy(s);
          break;
        }
      }
      if (g_open < pool_cap())
        break;
      g_cv.wait(lk);
    }
    g_open++;
    lk.unlock();
    Session *s = create(c, err);
    lk.lock();
    if (!s)
    {
      g_open--;
      g_cv.notify_all();
      return nullptr;
    }
    ((Priv *)s->priv)->busy = true;
    g_pool.push_back(s);
    return s;
  }

  int run(Session *s, int cur, int ref, int mode, const Mv *hints, Mv *out, std::string &err)
  {
    Priv *p = (Priv *)s->priv;
    const Config &c = s->cfg;
    if (cur < 0 || ref < 0 || cur > c.slots || ref > c.slots)
      return -4;
    if (mode == 2 && p->enc && p->used && cur != c.slots)
    {
      // reset: a run on the flat picture first, so the encoder's predictors are the same (zero) field for every call
      std::vector<Mv> dummy((size_t)s->gw * s->gh);
      const int r = run(s, c.slots, c.slots, 0, nullptr, dummy.data(), err);
      if (r)
        return r;
    }
    const bool fresh = mode == 1;
    if (fresh && p->used)
    {
      close_enc(s);
      if (!open_enc(s, err))
        return -2;
    }
    if (!p->enc && !open_enc(s, err))
      return -2;
    p->used = true;
    NV_ENC_MAP_INPUT_RESOURCE mi[2] = {};
    const int slot[2] = { cur, ref };
    NVENCSTATUS r = NV_ENC_SUCCESS;
    int mapped = 0;
    for (; mapped < 2; mapped++)
    {
      mi[mapped].version = NV_ENC_MAP_INPUT_RESOURCE_VER;
      mi[mapped].registeredResource = p->reg[slot[mapped]];
      if ((r = g_f.nvEncMapInputResource(p->enc, &mi[mapped])) != NV_ENC_SUCCESS)
      {
        err = status(p->enc, r, "nvEncMapInputResource");
        break;
      }
    }
    int rc = mapped == 2 ? 0 : -2;
    if (!rc)
    {
      NV_ENC_MEONLY_PARAMS mp = {};
      mp.version = NV_ENC_MEONLY_PARAMS_VER;
      mp.inputWidth = (uint32_t)c.width;
      mp.inputHeight = (uint32_t)c.height;
      mp.inputBuffer = mi[0].mappedResource;
      mp.referenceFrame = mi[1].mappedResource;
      mp.mvBuffer = p->mv;
      mp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
      // an encoder made with external hints rejects a run without them (NVENCSTATUS 8): runs without hints of their own
      // (reset, swapped) get zero vectors
      if (c.hints > 0 && !c.hevc)
      {
        const int mbw = (c.width + 15) / 16, mbh = (c.height + 15) / 16, n = c.hints;
        p->hints.assign((size_t)mbw * mbh * n, NVENC_EXTERNAL_ME_HINT{});
        for (size_t mb = 0; mb < (size_t)mbw * mbh; mb++)
          for (int k = 0; k < n; k++)
          {
            NVENC_EXTERNAL_ME_HINT &h = p->hints[mb * n + k];
            const Mv v = hints ? hints[mb * n + k] : Mv{ 0, 0 };
            h.mvx = std::min(std::max((int)v.x, -2047), 2047);
            h.mvy = std::min(std::max((int)v.y, -511), 511);
            h.refidx = 0;
            h.dir = 0;
            h.partType = 0;
            h.lastofPart = 1;
            h.lastOfMB = k == n - 1;
          }
        mp.meHintCountsPerBlock[0].numCandsPerBlk16x16 = (uint32_t)n;
        mp.meExternalHints = p->hints.data();
      }
      if ((r = g_f.nvEncRunMotionEstimationOnly(p->enc, &mp)) != NV_ENC_SUCCESS)
      {
        err = status(p->enc, r, "nvEncRunMotionEstimationOnly");
        rc = -2;
      }
    }
    if (!rc)
    {
      NV_ENC_LOCK_BITSTREAM lb = {};
      lb.version = NV_ENC_LOCK_BITSTREAM_VER;
      lb.outputBitstream = p->mv;
      if ((r = g_f.nvEncLockBitstream(p->enc, &lb)) != NV_ENC_SUCCESS)
      {
        err = status(p->enc, r, "nvEncLockBitstream");
        rc = -2;
      }
      else
      {
        const bool ok = c.hevc ? decode_hevc(s, (const NV_ENC_HEVC_MV_DATA *)lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes, out)
                               : decode_h264(s, (const NV_ENC_H264_MV_DATA *)lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes, out);
        g_f.nvEncUnlockBitstream(p->enc, p->mv);
        if (!ok)
        {
          err = "the NVENC vector buffer is smaller than expected";
          rc = -2;
        }
      }
    }
    for (int i = 0; i < mapped; i++)
      g_f.nvEncUnmapInputResource(p->enc, mi[i].mappedResource);
    return rc;
  }

  void release(Session *s)
  {
    if (!s)
      return;
    std::lock_guard<std::mutex> lk(g_mutex);
    ((Priv *)s->priv)->busy = false;
    g_cv.notify_all();
  }

  int available(int hevc, std::string &err)
  {
    static int result[2] = { -1, -1 }; // under g_mutex
    static std::string why[2];
    hevc = hevc ? 1 : 0;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      if (result[hevc] >= 0)
      {
        err = why[hevc];
        return result[hevc];
      }
    }
    cudaFree(nullptr); // makes the primary context current on this thread
    Config c;
    c.width = 160;
    c.height = 128;
    c.hevc = hevc;
    std::string e;
    Session *s = acquire(c, e);
    release(s);
    std::lock_guard<std::mutex> lk(g_mutex);
    result[hevc] = s ? 1 : 0;
    why[hevc] = e;
    err = e;
    return result[hevc];
  }
}
