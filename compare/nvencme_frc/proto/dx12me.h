// D3D12 video motion estimator (ID3D12VideoMotionEstimator), the API DTL2020's MAnalyse uses with DX12_ME
// (optSearchOption 5 / 6): NV12 in, one vector per 8x8 or 16x16 block, quarter pel. Same calls as DTL's
// Init_DX12_ME / EstimateMotion / ResolveMotionVectorHeap, without the d3dx12 helpers.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <d3d12video.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

namespace dx12me
{
  using Microsoft::WRL::ComPtr;

  struct Mv
  {
    int16_t x, y; // quarter pel, current -> reference
  };

  class Estimator
  {
  public:
    std::string err, info;
    int w = 0, h = 0, bs = 8;

    bool open(int width, int height, int block)
    {
      w = width;
      h = height;
      bs = block;
      ComPtr<IDXGIFactory6> f;
      if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f))))
        return fail("CreateDXGIFactory2");
      ComPtr<IDXGIAdapter1> ad;
      for (UINT i = 0; f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad)) != DXGI_ERROR_NOT_FOUND; i++)
      {
        DXGI_ADAPTER_DESC1 d;
        ad->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
          continue;
        if (SUCCEEDED(D3D12CreateDevice(ad.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev))))
        {
          char name[128];
          WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof name, nullptr, nullptr);
          info = name;
          break;
        }
      }
      if (!dev)
        return fail("no D3D12 hardware device");
      if (FAILED(dev->QueryInterface(IID_PPV_ARGS(&vdev))))
        return fail("ID3D12VideoDevice1 not available");
      D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR fs = {};
      fs.NodeIndex = 0;
      fs.InputFormat = DXGI_FORMAT_NV12;
      if (FAILED(vdev->CheckFeatureSupport(D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR, &fs, sizeof fs)))
        return fail("CheckFeatureSupport(VIDEO_MOTION_ESTIMATOR)");
      char b[256];
      std::snprintf(b, sizeof b, "%s; block sizes flags 0x%x (1 = 8x8, 2 = 16x16), precision flags 0x%x, size %ux%u..%ux%u", info.c_str(),
        (unsigned)fs.BlockSizeFlags, (unsigned)fs.PrecisionFlags, fs.SizeRange.MinWidth, fs.SizeRange.MinHeight, fs.SizeRange.MaxWidth,
        fs.SizeRange.MaxHeight);
      info = b;
      const D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE bsz =
        bs == 16 ? D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16 : D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_8X8;
      D3D12_VIDEO_MOTION_ESTIMATOR_DESC md = { 0, DXGI_FORMAT_NV12, bsz, D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL,
        { (UINT)w, (UINT)h, (UINT)w, (UINT)h } };
      if (FAILED(vdev->CreateVideoMotionEstimator(&md, nullptr, IID_PPV_ARGS(&me))))
        return fail("CreateVideoMotionEstimator");
      D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC hd = { 0, DXGI_FORMAT_NV12, bsz, D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL,
        { (UINT)w, (UINT)h, (UINT)w, (UINT)h } };
      if (FAILED(vdev->CreateVideoMotionVectorHeap(&hd, nullptr, IID_PPV_ARGS(&heap))))
        return fail("CreateVideoMotionVectorHeap");

      D3D12_COMMAND_QUEUE_DESC qd = {};
      qd.Type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
      if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&qv))))
        return fail("CreateCommandQueue(video encode)");
      qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&qg))))
        return fail("CreateCommandQueue(direct)");
      if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE, IID_PPV_ARGS(&av)))
        || FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ag))))
        return fail("CreateCommandAllocator");
      if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE, av.Get(), nullptr, IID_PPV_ARGS(&lv)))
        || FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ag.Get(), nullptr, IID_PPV_ARGS(&lg))))
        return fail("CreateCommandList");
      lv->Close();
      lg->Close();
      if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return fail("CreateFence");
      ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);

      // NV12 textures, upload buffer, resolved MV texture, readback buffer
      D3D12_HEAP_PROPERTIES hdef = { D3D12_HEAP_TYPE_DEFAULT }, hup = { D3D12_HEAP_TYPE_UPLOAD }, hrb = { D3D12_HEAP_TYPE_READBACK };
      D3D12_RESOURCE_DESC td = {};
      td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      td.Width = w;
      td.Height = h;
      td.DepthOrArraySize = 1;
      td.MipLevels = 1;
      td.Format = DXGI_FORMAT_NV12;
      td.SampleDesc.Count = 1;
      for (auto &t : tex)
        if (FAILED(dev->CreateCommittedResource(&hdef, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t))))
          return fail("CreateCommittedResource(NV12)");
      dev->GetCopyableFootprints(&td, 0, 2, 0, fp, rows, rowbytes, &up_size);
      D3D12_RESOURCE_DESC bd = {};
      bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      bd.Width = up_size * 2;
      bd.Height = 1;
      bd.DepthOrArraySize = 1;
      bd.MipLevels = 1;
      bd.SampleDesc.Count = 1;
      bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      if (FAILED(dev->CreateCommittedResource(&hup, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&up))))
        return fail("CreateCommittedResource(upload)");
      gw = (w + bs - 1) / bs;
      gh = (h + bs - 1) / bs;
      D3D12_RESOURCE_DESC md2 = td;
      md2.Width = gw;
      md2.Height = gh;
      md2.Format = DXGI_FORMAT_R16G16_SINT;
      if (FAILED(dev->CreateCommittedResource(&hdef, D3D12_HEAP_FLAG_NONE, &md2, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mvtex))))
        return fail("CreateCommittedResource(MV texture)");
      UINT64 mvsize = 0;
      dev->GetCopyableFootprints(&md2, 0, 1, 0, &mvfp, nullptr, nullptr, &mvsize);
      bd.Width = mvsize;
      if (FAILED(dev->CreateCommittedResource(&hrb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb))))
        return fail("CreateCommittedResource(readback)");
      return true;
    }

    // cur / ref: 8-bit luma (w x h, pitch w); uv: NV12 chroma ((h + 1) / 2 rows of w bytes) or null for grey
    bool run(const uint8_t *cur, const uint8_t *ref, const uint8_t *cur_uv, const uint8_t *ref_uv, std::vector<Mv> &out)
    {
      // upload both pictures
      uint8_t *p = nullptr;
      if (FAILED(up->Map(0, nullptr, (void **)&p)))
        return fail("Map(upload)");
      const uint8_t *ys[2] = { cur, ref }, *uvs[2] = { cur_uv, ref_uv };
      for (int i = 0; i < 2; i++)
      {
        uint8_t *base = p + up_size * i;
        for (UINT y = 0; y < rows[0]; y++)
          std::memcpy(base + fp[0].Offset + (size_t)y * fp[0].Footprint.RowPitch, ys[i] + (size_t)y * w, w);
        for (UINT y = 0; y < rows[1]; y++)
        {
          uint8_t *d = base + fp[1].Offset + (size_t)y * fp[1].Footprint.RowPitch;
          if (uvs[i])
            std::memcpy(d, uvs[i] + (size_t)y * w, w);
          else
            std::memset(d, 128, w);
        }
      }
      up->Unmap(0, nullptr);
      ag->Reset();
      lg->Reset(ag.Get(), nullptr);
      for (int i = 0; i < 2; i++)
      {
        barrier(lg.Get(), tex[i].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        for (UINT s = 0; s < 2; s++)
        {
          D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
          dst.pResource = tex[i].Get();
          dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          dst.SubresourceIndex = s;
          src.pResource = up.Get();
          src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
          src.PlacedFootprint = fp[s];
          src.PlacedFootprint.Offset += up_size * i;
          lg->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        barrier(lg.Get(), tex[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
      }
      lg->Close();
      if (!exec(qg.Get(), lg.Get()))
        return false;

      // motion estimation + resolve
      av->Reset();
      lv->Reset(av.Get());
      vbarrier(tex[0].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ);
      vbarrier(tex[1].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ);
      vbarrier(mvtex.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE);
      const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT o = { heap.Get() };
      const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT in = { tex[0].Get(), 0, tex[1].Get(), 0, nullptr };
      lv->EstimateMotion(me.Get(), &o, &in);
      const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT ro = { mvtex.Get(), {} };
      const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT ri = { heap.Get(), (UINT)w, (UINT)h };
      lv->ResolveMotionVectorHeap(&ro, &ri);
      vbarrier(mvtex.Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, D3D12_RESOURCE_STATE_COMMON);
      vbarrier(tex[0].Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, D3D12_RESOURCE_STATE_COMMON);
      vbarrier(tex[1].Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, D3D12_RESOURCE_STATE_COMMON);
      if (FAILED(lv->Close()))
        return fail("video command list Close");
      if (!exec(qv.Get(), lv.Get()))
        return false;

      // read back
      ag->Reset();
      lg->Reset(ag.Get(), nullptr);
      barrier(lg.Get(), mvtex.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
      D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
      dst.pResource = rb.Get();
      dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      dst.PlacedFootprint = mvfp;
      src.pResource = mvtex.Get();
      src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      lg->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
      barrier(lg.Get(), mvtex.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
      lg->Close();
      if (!exec(qg.Get(), lg.Get()))
        return false;
      uint8_t *r = nullptr;
      if (FAILED(rb->Map(0, nullptr, (void **)&r)))
        return fail("Map(readback)");
      out.resize((size_t)gw * gh);
      for (int y = 0; y < gh; y++)
        std::memcpy(&out[(size_t)y * gw], r + mvfp.Offset + (size_t)y * mvfp.Footprint.RowPitch, (size_t)gw * sizeof(Mv));
      rb->Unmap(0, nullptr);
      return true;
    }

    int gw = 0, gh = 0;

  private:
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12VideoDevice1> vdev;
    ComPtr<ID3D12VideoMotionEstimator> me;
    ComPtr<ID3D12VideoMotionVectorHeap> heap;
    ComPtr<ID3D12CommandQueue> qv, qg;
    ComPtr<ID3D12CommandAllocator> av, ag;
    ComPtr<ID3D12VideoEncodeCommandList> lv;
    ComPtr<ID3D12GraphicsCommandList> lg;
    ComPtr<ID3D12Fence> fence;
    HANDLE ev = nullptr;
    UINT64 fval = 0;
    ComPtr<ID3D12Resource> tex[2], up, mvtex, rb;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2] = {}, mvfp = {};
    UINT rows[2] = {};
    UINT64 rowbytes[2] = {}, up_size = 0;

    bool fail(const char *what)
    {
      err = what;
      return false;
    }

    static void barrier(ID3D12GraphicsCommandList *l, ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
    {
      D3D12_RESOURCE_BARRIER br = {};
      br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      br.Transition.pResource = r;
      br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      br.Transition.StateBefore = a;
      br.Transition.StateAfter = b;
      l->ResourceBarrier(1, &br);
    }

    void vbarrier(ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
    {
      D3D12_RESOURCE_BARRIER br = {};
      br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      br.Transition.pResource = r;
      br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      br.Transition.StateBefore = a;
      br.Transition.StateAfter = b;
      lv->ResourceBarrier(1, &br);
    }

    bool exec(ID3D12CommandQueue *q, ID3D12CommandList *l)
    {
      q->ExecuteCommandLists(1, &l);
      const UINT64 v = ++fval;
      if (FAILED(q->Signal(fence.Get(), v)))
        return fail("Signal");
      if (fence->GetCompletedValue() < v)
      {
        fence->SetEventOnCompletion(v, ev);
        if (WaitForSingleObject(ev, 10000) != WAIT_OBJECT_0) // bounded: never hang
          return fail("GPU timeout");
      }
      if (FAILED(dev->GetDeviceRemovedReason()))
        return fail("device removed");
      return true;
    }
  };
}
