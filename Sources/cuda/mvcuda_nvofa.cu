// NVOFA sessions for the MAnalyse experiment (mvcuda_nvofa.h). The SDK headers (third_party/nvof, NVIDIA, BSD-3) only
// declare types; nvofapi64.dll is loaded at run time, so the DLL has no new link dependency and works without it.
#include "mvcuda_nvofa.h"

#include "third_party/nvof/nvOpticalFlowCuda.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace mvc_nvofa
{
  namespace
  {
    struct Priv
    {
      NvOFHandle h = nullptr;
      NvOFGPUBufferHandle b_in[2] = {}, b_out = nullptr;
      int level = 0;
      bool busy = false;
    };

    std::mutex g_mutex;
    NV_OF_CUDA_API_FUNCTION_LIST g_f;
    int g_loaded = 0; // 0 not tried, 1 ok, -1 failed
    std::vector<Session *> g_pool;

    typedef CUresult(CUDAAPI *PFN_ctx_get)(CUcontext *);
    PFN_ctx_get g_ctx_get = nullptr;

    bool load_locked()
    {
      if (g_loaded)
        return g_loaded > 0;
      g_loaded = -1;
      HMODULE m = LoadLibraryW(L"nvofapi64.dll");
      if (!m)
      {
        fprintf(stderr, "mvcuda nvofa: nvofapi64.dll not found\n");
        return false;
      }
      typedef NV_OF_STATUS(NVOFAPI * PFN_create)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *);
      auto create = (PFN_create)GetProcAddress(m, "NvOFAPICreateInstanceCuda");
      cudaDriverEntryPointQueryResult q;
      if (!create || create(NV_OF_API_VERSION, &g_f) != NV_OF_SUCCESS
        || cudaGetDriverEntryPointByVersion("cuCtxGetCurrent", (void **)&g_ctx_get, 12000, cudaEnableDefault, &q) != cudaSuccess
        || !g_ctx_get)
      {
        fprintf(stderr, "mvcuda nvofa: NvOFAPICreateInstanceCuda / cuCtxGetCurrent failed\n");
        return false;
      }
      g_loaded = 1;
      return true;
    }

    void report(NvOFHandle h, NV_OF_STATUS s, const char *what)
    {
      char msg[512] = "";
      uint32_t n = sizeof msg;
      if (h)
        g_f.nvOFGetLastError(h, msg, &n);
      fprintf(stderr, "mvcuda nvofa: %s failed (status %d) %s\n", what, (int)s, msg);
    }

    void destroy(Session *s)
    {
      Priv *p = (Priv *)s->priv;
      for (auto b : p->b_in)
        if (b)
          g_f.nvOFDestroyGPUBufferCuda(b);
      if (p->b_out)
        g_f.nvOFDestroyGPUBufferCuda(p->b_out);
      if (p->h)
        g_f.nvOFDestroy(p->h);
      delete p;
      delete s;
    }

    Session *create_session(int width, int height, int level)
    {
      CUcontext ctx = nullptr;
      g_ctx_get(&ctx);
      if (!ctx)
      {
        fprintf(stderr, "mvcuda nvofa: no current CUDA context\n");
        return nullptr;
      }
      Session *s = new Session();
      Priv *p = new Priv();
      s->priv = p;
      p->level = level;
      NV_OF_STATUS r = g_f.nvCreateOpticalFlowCuda(ctx, &p->h);
      if (r)
      {
        report(nullptr, r, "nvCreateOpticalFlowCuda");
        destroy(s);
        return nullptr;
      }
      const int grid = 4;
      NV_OF_INIT_PARAMS ip = {};
      ip.width = (uint32_t)width;
      ip.height = (uint32_t)height;
      ip.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
      ip.mode = NV_OF_MODE_OPTICALFLOW;
      ip.perfLevel = (NV_OF_PERF_LEVEL)level;
      if ((r = g_f.nvOFInit(p->h, &ip)) != NV_OF_SUCCESS)
      {
        report(p->h, r, "nvOFInit");
        destroy(s);
        return nullptr;
      }
      const int gw = (width + grid - 1) / grid, gh = (height + grid - 1) / grid;
      NV_OF_BUFFER_DESCRIPTOR di = { (uint32_t)width, (uint32_t)height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8 };
      NV_OF_BUFFER_DESCRIPTOR dout = { (uint32_t)gw, (uint32_t)gh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2 };
      for (int i = 0; i < 2 && !r; i++)
        r = g_f.nvOFCreateGPUBufferCuda(p->h, &di, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &p->b_in[i]);
      if (!r)
        r = g_f.nvOFCreateGPUBufferCuda(p->h, &dout, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &p->b_out);
      NV_OF_CUDA_BUFFER_STRIDE_INFO si = {}, so = {};
      if (!r)
        r = g_f.nvOFGPUBufferGetStrideInfo(p->b_in[0], &si);
      if (!r)
        r = g_f.nvOFGPUBufferGetStrideInfo(p->b_out, &so);
      if (r)
      {
        report(p->h, r, "buffer creation");
        destroy(s);
        return nullptr;
      }
      for (int i = 0; i < 2; i++)
        s->in[i] = (uint8_t *)g_f.nvOFGPUBufferGetCUdeviceptr(p->b_in[i]);
      s->in_pitch = si.strideInfo[0].strideXInBytes;
      s->flow = (const short2 *)g_f.nvOFGPUBufferGetCUdeviceptr(p->b_out);
      s->flow_pitch = so.strideInfo[0].strideXInBytes;
      s->width = width;
      s->height = height;
      s->grid = grid;
      s->gw = gw;
      s->gh = gh;
      return s;
    }
  }

  int perf_level()
  {
    static const int level = [] {
      const char *e = getenv("MVCUDA_NVOFA");
      if (!e || !*e || e[0] == '0')
        return 0;
      if (!_stricmp(e, "fast"))
        return (int)NV_OF_PERF_LEVEL_FAST;
      if (!_stricmp(e, "slow"))
        return (int)NV_OF_PERF_LEVEL_SLOW;
      return (int)NV_OF_PERF_LEVEL_MEDIUM;
    }();
    return level;
  }

  Session *acquire(int width, int height)
  {
    const int level = perf_level();
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!level || !load_locked())
      return nullptr;
    for (Session *s : g_pool)
    {
      Priv *p = (Priv *)s->priv;
      if (!p->busy && s->width == width && s->height == height && p->level == level)
      {
        p->busy = true;
        return s;
      }
    }
    Session *s = create_session(width, height, level);
    if (s)
    {
      ((Priv *)s->priv)->busy = true;
      g_pool.push_back(s);
    }
    return s;
  }

  int execute(Session *s, cudaStream_t st)
  {
    Priv *p = (Priv *)s->priv;
    NV_OF_STATUS r = g_f.nvOFSetIOCudaStreams(p->h, (CUstream)st, (CUstream)st);
    if (r)
    {
      report(p->h, r, "nvOFSetIOCudaStreams");
      return -2;
    }
    NV_OF_EXECUTE_INPUT_PARAMS ei = {};
    ei.inputFrame = p->b_in[0];
    ei.referenceFrame = p->b_in[1];
    // calls arrive in any order (both directions, Prefetch threads): the previous call's flow is no predictor
    ei.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eo = {};
    eo.outputBuffer = p->b_out;
    if ((r = g_f.nvOFExecute(p->h, &ei, &eo)) != NV_OF_SUCCESS)
    {
      report(p->h, r, "nvOFExecute");
      return -2;
    }
    return 0;
  }

  void release(Session *s)
  {
    if (!s)
      return;
    std::lock_guard<std::mutex> lk(g_mutex);
    ((Priv *)s->priv)->busy = false;
  }
}
