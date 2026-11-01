// mvtools_vsbridge: the AviSynth+ host side under VapourSynth. See bridge_core.h and VAPOURSYNTH_PLAN.md.
//
// Pieces, in file order:
//   frames      VideoFrame / VideoFrameBuffer construction (core mode), wrapping VS frames as AviSynth frames and back
//   registry    the full AviSynth VideoInfo of every node the bridge creates (MSuper / MAnalyse side channel)
//   VSClip      an input VS node seen as an AviSynth clip; frames come from the request set of the calling filter
//   BridgeEnv   IScriptEnvironment: frames, errors, frame properties, Invoke between registered functions
//   filters     a VS filter around an AviSynth clip: learned prefetch, instance pool for MT_MULTI_INSTANCE
//   functions   AddFunction -> VS registerFunction, argument translation
//   entry       mvbridge_register_functions, called from each plugin's VapourSynthPluginInit2

#include "bridge_core.h"
#include "mt_port.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#define MVBRIDGE_EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#define MVBRIDGE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{

// ---------------------------------------------------------------------------------------------------------------------
// globals

const VSAPI *g_vsapi = nullptr; // the same table for every core of the process; set before any frame exists

bool debug_log()
{
  static const bool on = [] {
    const char *v = std::getenv("MVBRIDGE_DEBUG");
    return v && *v && *v != '0';
  }();
  return on;
}

#ifndef _WIN32
// MVBRIDGE_DEBUG only: on a crash print each frame as module + offset (resolve with addr2line -f -e <module> <offset>)
void crash_handler(int sig, siginfo_t *si, void *)
{
  void *frames[64];
  const int n = backtrace(frames, 64);
  char line[512];
  int len = snprintf(line, sizeof(line), "mvtools_vsbridge: signal %d at address %p\n", sig, si ? si->si_addr : nullptr);
  (void)!write(2, line, (size_t)len);
  for (int i = 0; i < n; ++i)
  {
    Dl_info info{};
    if (dladdr(frames[i], &info) && info.dli_fname)
      len = snprintf(line, sizeof(line), "  #%d %s +0x%zx\n", i, info.dli_fname,
                     (size_t)((const char *)frames[i] - (const char *)info.dli_fbase));
    else
      len = snprintf(line, sizeof(line), "  #%d %p\n", i, frames[i]);
    (void)!write(2, line, (size_t)len);
  }
  signal(sig, SIG_DFL);
  raise(sig);
}

void install_crash_handler()
{
  static bool done = false;
  if (done || !debug_log())
    return;
  done = true;
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = crash_handler;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
}
#else
void install_crash_handler() {}
#endif

void *aligned_alloc64(size_t size)
{
#ifdef _WIN32
  return _aligned_malloc(size, 64);
#else
  void *p = nullptr;
  return posix_memalign(&p, 64, size) == 0 ? p : nullptr;
#endif
}

void aligned_free64(void *p)
{
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

void bitblt(uint8_t *dst, ptrdiff_t dst_pitch, const uint8_t *src, ptrdiff_t src_pitch, size_t row_size, int height)
{
  if (height <= 0 || row_size == 0)
    return;
  if (dst_pitch == src_pitch && (size_t)src_pitch == row_size)
  {
    memcpy(dst, src, row_size * (size_t)height);
    return;
  }
  for (int y = 0; y < height; ++y)
    memcpy(dst + y * dst_pitch, src + y * src_pitch, row_size);
}

int muldiv(int a, int b, int c) { return c ? (int)(((int64_t)a * b + c / 2) / c) : 0; }

} // namespace

struct BridgeFilter;

namespace
{

// per-thread call context: which filter is computing which frame, with which VS frame context
struct CallCtx
{
  BridgeFilter *filter;
  VSFrameContext *ctx;
  int n;
  VSCore *core;
};
thread_local CallCtx *t_call = nullptr;
thread_local VSCore *t_core = nullptr; // during filter creation

VSCore *current_core() { return t_call ? t_call->core : t_core; }

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// frames

// A frame buffer of the bridge: either a VS frame (wrapped without copying) or a contiguous buffer of its own.
struct BridgeVFB : VideoFrameBuffer
{
  const VSFrame *vsframe = nullptr; // reference owned by this buffer
  bool vs_writable = false;         // vsframe came from newVideoFrame (the bridge may write to it)
  void *owned = nullptr;            // contiguous buffer, aligned_alloc64
  std::atomic<bool> readonly{false};
};

AVSMap::~AVSMap()
{
  if (owned && rw && g_vsapi)
    g_vsapi->freeMap(rw);
}

void bridge_release_buffer(VideoFrameBuffer *vfb)
{
  BridgeVFB *b = static_cast<BridgeVFB *>(vfb);
  if (b->vsframe)
    g_vsapi->freeFrame(b->vsframe);
  if (b->owned)
    aligned_free64(b->owned);
  delete b;
}

bool bridge_buffer_readonly(const VideoFrameBuffer *vfb) { return static_cast<const BridgeVFB *>(vfb)->readonly; }

// core-mode definitions that AviSynth+ keeps in avisynth.cpp
VideoFrameBuffer::VideoFrameBuffer() : data(nullptr), data_size(0), sequence_number(0), refcount(0), device(nullptr) {}
VideoFrameBuffer::~VideoFrameBuffer() {}
void VideoFrameBuffer::DESTRUCTOR() {}

void *VideoFrame::operator new(size_t size) { return ::operator new(size); }

VideoFrame::VideoFrame(VideoFrameBuffer *_vfb, AVSMap *avsmap, int _offset, int _pitch, int _row_size, int _height,
                       int _pixel_type)
    : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height), offsetU(_offset),
      offsetV(_offset), pitchUV(0), row_sizeUV(0), heightUV(0), offsetA(0), pitchA(0), row_sizeA(0),
      properties(avsmap), pixel_type(_pixel_type)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer *_vfb, AVSMap *avsmap, int _offset, int _pitch, int _row_size, int _height,
                       int _offsetU, int _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV, int _pixel_type)
    : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height), offsetU(_offsetU),
      offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV), offsetA(0), pitchA(0),
      row_sizeA(0), properties(avsmap), pixel_type(_pixel_type)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer *_vfb, AVSMap *avsmap, int _offset, int _pitch, int _row_size, int _height,
                       int _offsetU, int _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV, int _offsetA,
                       int _pixel_type)
    : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height), offsetU(_offsetU),
      offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV), offsetA(_offsetA),
      pitchA(_pitch), row_sizeA(_row_size), properties(avsmap), pixel_type(_pixel_type)
{
  InterlockedIncrement(&vfb->refcount);
}

static AVSMap *copy_props(const AVSMap *src)
{
  AVSMap *m = new AVSMap;
  m->rw = g_vsapi->createMap();
  m->owned = true;
  if (src && src->get())
    g_vsapi->copyMap(src->get(), m->rw);
  return m;
}

VideoFrame *VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height) const
{
  return new VideoFrame(vfb, copy_props(properties), offset + rel_offset, new_pitch, new_row_size, new_height,
                        pixel_type);
}

VideoFrame *VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU,
                                 int rel_offsetV, int new_pitchUV) const
{
  const int new_row_sizeUV = !row_size ? 0 : muldiv(new_row_size, row_sizeUV, row_size);
  const int new_heightUV = !height ? 0 : muldiv(new_height, heightUV, height);
  return new VideoFrame(vfb, copy_props(properties), offset + rel_offset, new_pitch, new_row_size, new_height,
                        rel_offsetU + offsetU, rel_offsetV + offsetV, new_pitchUV, new_row_sizeUV, new_heightUV,
                        pixel_type);
}

VideoFrame *VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU,
                                 int rel_offsetV, int new_pitchUV, int rel_offsetA) const
{
  const int new_row_sizeUV = !row_size ? 0 : muldiv(new_row_size, row_sizeUV, row_size);
  const int new_heightUV = !height ? 0 : muldiv(new_height, heightUV, height);
  return new VideoFrame(vfb, copy_props(properties), offset + rel_offset, new_pitch, new_row_size, new_height,
                        rel_offsetU + offsetU, rel_offsetV + offsetV, new_pitchUV, new_row_sizeUV, new_heightUV,
                        rel_offsetA + offsetA, pixel_type);
}

// The friend of VideoFrame / VideoFrameBuffer that AviSynth+ calls ScriptEnvironment: private access for the bridge.
class ScriptEnvironment
{
public:
  static void set_buffer(VideoFrameBuffer *vfb, BYTE *data, size_t size)
  {
    vfb->data = data;
    vfb->data_size = size > (size_t)INT_MAX ? INT_MAX : (int)size;
  }
  static VideoFrame *frame1(VideoFrameBuffer *vfb, AVSMap *props, int pitch, int row, int h, int pixel_type)
  {
    return new VideoFrame(vfb, props, 0, pitch, row, h, pixel_type);
  }
  static VideoFrame *frame3(VideoFrameBuffer *vfb, AVSMap *props, int pitch, int row, int h, int offU, int offV,
                            int pitchUV, int rowUV, int hUV, int pixel_type)
  {
    return new VideoFrame(vfb, props, 0, pitch, row, h, offU, offV, pitchUV, rowUV, hUV, pixel_type);
  }
  static AVSMap *&props(VideoFrame *f) { return f->properties; }
  static const AVSMap *props(const VideoFrame *f) { return f->properties; }
  // a new view on the same buffer with its own properties (MakePropertyWritable)
  static VideoFrame *clone_view(const VideoFrame *f)
  {
    VideoFrame *c = new VideoFrame(f->vfb, copy_props(f->properties), f->offset, f->pitch, f->row_size, f->height,
                                   f->offsetU, f->offsetV, f->pitchUV, f->row_sizeUV, f->heightUV, f->pixel_type);
    c->offsetA = f->offsetA;
    c->pitchA = f->pitchA;
    c->row_sizeA = f->row_sizeA;
    return c;
  }
};

namespace
{

// How the planes of an AviSynth frame map to the planes of the VS frame that carries it.
//   packed (vector clips: BGR32 / BGR24 / YUY2, ...): one Gray8 plane, width = bytes per row
//   Y: one plane; YUV: Y,U,V = VS 0,1,2; planar RGB: AviSynth G,B,R = VS 1,2,0
struct Geometry
{
  int nplanes = 0;
  int vsplane[3] = {0, 1, 2};
  int avsplane[3] = {0, 0, 0}; // plane constant for GetReadPtr / GetPitch
  int row[3] = {0, 0, 0};      // bytes
  int height[3] = {0, 0, 0};
};

bool is_packed(const VideoInfo &vi) { return !vi.IsPlanar() || vi.IsYUY2(); }

Geometry geometry(const VideoInfo &vi)
{
  Geometry g;
  if (is_packed(vi))
  {
    g.nplanes = 1;
    g.row[0] = vi.RowSize();
    g.height[0] = vi.height;
    return g;
  }
  const int cs = vi.ComponentSize();
  if (vi.IsY())
  {
    g.nplanes = 1;
    g.avsplane[0] = PLANAR_Y;
    g.row[0] = vi.width * cs;
    g.height[0] = vi.height;
    return g;
  }
  g.nplanes = 3;
  if (vi.IsPlanarRGB())
  {
    g.vsplane[0] = 1; // G
    g.vsplane[1] = 2; // B
    g.vsplane[2] = 0; // R
    g.avsplane[0] = PLANAR_G;
    g.avsplane[1] = PLANAR_B;
    g.avsplane[2] = PLANAR_R;
    for (int k = 0; k < 3; ++k)
    {
      g.row[k] = vi.width * cs;
      g.height[k] = vi.height;
    }
    return g;
  }
  g.avsplane[0] = PLANAR_Y;
  g.avsplane[1] = PLANAR_U;
  g.avsplane[2] = PLANAR_V;
  g.row[0] = vi.width * cs;
  g.height[0] = vi.height;
  const int sw = vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int sh = vi.GetPlaneHeightSubsampling(PLANAR_U);
  g.row[1] = g.row[2] = (vi.width >> sw) * cs;
  g.height[1] = g.height[2] = vi.height >> sh;
  return g;
}

// AviSynth pixel type of a VS format; 0 if AviSynth has none
int avs_pixel_type(const VSVideoFormat &f)
{
  int bits;
  if (f.sampleType == stFloat)
  {
    if (f.bitsPerSample != 32)
      return 0;
    bits = VideoInfo::CS_Sample_Bits_32;
  }
  else
  {
    switch (f.bitsPerSample)
    {
    case 8: bits = VideoInfo::CS_Sample_Bits_8; break;
    case 10: bits = VideoInfo::CS_Sample_Bits_10; break;
    case 12: bits = VideoInfo::CS_Sample_Bits_12; break;
    case 14: bits = VideoInfo::CS_Sample_Bits_14; break;
    case 16: bits = VideoInfo::CS_Sample_Bits_16; break;
    default: return 0;
    }
  }
  switch (f.colorFamily)
  {
  case cfGray: return VideoInfo::CS_GENERIC_Y | bits;
  case cfRGB: return (f.subSamplingW || f.subSamplingH) ? 0 : (VideoInfo::CS_GENERIC_RGBP | bits);
  case cfYUV:
    if (f.subSamplingW == 0 && f.subSamplingH == 0) return VideoInfo::CS_GENERIC_YUV444 | bits;
    if (f.subSamplingW == 1 && f.subSamplingH == 0) return VideoInfo::CS_GENERIC_YUV422 | bits;
    if (f.subSamplingW == 1 && f.subSamplingH == 1) return VideoInfo::CS_GENERIC_YUV420 | bits;
    if (f.subSamplingW == 2 && f.subSamplingH == 0 && f.bitsPerSample == 8) return VideoInfo::CS_YV411;
    if (f.subSamplingW == 2 && f.subSamplingH == 2 && f.bitsPerSample == 8) return VideoInfo::CS_YUV9;
    return 0;
  default: return 0;
  }
}

// VS format and frame size that carry an AviSynth VideoInfo; false (and a reason) if VS cannot carry it
bool vs_format(const VideoInfo &vi, VSCore *core, VSVideoFormat &fmt, int &width, int &height, std::string &why)
{
  height = vi.height;
  if (is_packed(vi))
  {
    width = vi.RowSize(); // bytes per row as Gray8 samples
    if (!g_vsapi->queryVideoFormat(&fmt, cfGray, stInteger, 8, 0, 0, core))
    {
      why = "no Gray8 format";
      return false;
    }
    return true;
  }
  width = vi.width;
  if (vi.IsYUVA() || vi.IsPlanarRGBA())
  {
    why = "formats with alpha are not supported under VapourSynth";
    return false;
  }
  const int bits = vi.BitsPerComponent();
  const int st = bits == 32 ? stFloat : stInteger;
  int cf, sw = 0, sh = 0;
  if (vi.IsY())
    cf = cfGray;
  else if (vi.IsPlanarRGB())
    cf = cfRGB;
  else if (vi.IsYUV())
  {
    cf = cfYUV;
    sw = vi.GetPlaneWidthSubsampling(PLANAR_U);
    sh = vi.GetPlaneHeightSubsampling(PLANAR_U);
  }
  else
  {
    why = "unsupported pixel type";
    return false;
  }
  if (!g_vsapi->queryVideoFormat(&fmt, cf, st, bits, sw, sh, core))
  {
    why = "no matching VapourSynth format";
    return false;
  }
  return true;
}

// AviSynth VideoInfo of a VS clip that did not come from the bridge
bool avs_video_info(const VSVideoInfo &vsvi, VideoInfo &vi, std::string &why)
{
  memset(&vi, 0, sizeof(vi));
  if (vsvi.format.colorFamily == cfUndefined || vsvi.width <= 0 || vsvi.height <= 0)
  {
    why = "clips with variable format or size are not supported";
    return false;
  }
  vi.pixel_type = avs_pixel_type(vsvi.format);
  if (!vi.pixel_type)
  {
    why = "the clip's format has no AviSynth equivalent (Gray / YUV 4:2:0 / 4:2:2 / 4:4:4 / planar RGB, 8-16 bit "
          "integer or 32 bit float)";
    return false;
  }
  vi.width = vsvi.width;
  vi.height = vsvi.height;
  vi.num_frames = vsvi.numFrames;
  if (vsvi.fpsNum > 0 && vsvi.fpsDen > 0 && vsvi.fpsNum <= UINT_MAX && vsvi.fpsDen <= UINT_MAX)
  {
    vi.fps_numerator = (unsigned)vsvi.fpsNum;
    vi.fps_denominator = (unsigned)vsvi.fpsDen;
  }
  else
  {
    vi.fps_numerator = 25; // variable frame rate: AviSynth has none
    vi.fps_denominator = 1;
  }
  return true;
}

// An AviSynth frame over a VS frame. Takes over the reference f. Zero-copy when the plane distances fit AviSynth's int
// offsets, else a contiguous copy.
PVideoFrame wrap_vs_frame(const VSFrame *f, bool writable, const VideoInfo &vi)
{
  const Geometry g = geometry(vi);
  BridgeVFB *b = new BridgeVFB;
  AVSMap *props = new AVSMap;
  const uint8_t *p[3] = {};
  int pitch[3] = {};
  for (int k = 0; k < g.nplanes; ++k)
  {
    p[k] = writable ? g_vsapi->getWritePtr(const_cast<VSFrame *>(f), g.vsplane[k])
                    : g_vsapi->getReadPtr(f, g.vsplane[k]);
    pitch[k] = (int)g_vsapi->getStride(f, g.vsplane[k]);
  }
  bool fits = true;
  ptrdiff_t off[3] = {0, 0, 0};
  for (int k = 1; k < g.nplanes; ++k)
  {
    // int offsets, with room for subframe offsets on top
    off[k] = p[k] - p[0];
    if (off[k] < (ptrdiff_t)INT_MIN + (1 << 28) || off[k] > (ptrdiff_t)INT_MAX - (1 << 28))
      fits = false;
  }
  b->readonly = !writable; // input frames belong to VS's cache
  if (fits)
  {
    b->vsframe = f;
    b->vs_writable = writable;
    ScriptEnvironment::set_buffer(b, const_cast<BYTE *>(p[0]), (size_t)pitch[0] * g.height[0]);
    if (writable)
      props->rw = g_vsapi->getFramePropertiesRW(const_cast<VSFrame *>(f));
    else
      props->ro = g_vsapi->getFramePropertiesRO(f);
    VideoFrame *vf = g.nplanes == 1
                         ? ScriptEnvironment::frame1(b, props, pitch[0], g.row[0], g.height[0], vi.pixel_type)
                         : ScriptEnvironment::frame3(b, props, pitch[0], g.row[0], g.height[0], (int)off[1],
                                                     (int)off[2], pitch[1], g.row[1], g.height[1], vi.pixel_type);
    return PVideoFrame(vf);
  }
  // contiguous copy with the VS frame's own strides: mvtools assumes that frames of one format share one pitch (e.g.
  // FlowInterExtra reads src with the ref frame's pitch), so the copy must not change it
  int dpitch[3];
  size_t size = 0, start[3];
  for (int k = 0; k < g.nplanes; ++k)
  {
    dpitch[k] = pitch[k];
    start[k] = size;
    size += (size_t)dpitch[k] * g.height[k];
  }
  b->owned = aligned_alloc64(size ? size : 64);
  uint8_t *base = static_cast<uint8_t *>(b->owned);
  for (int k = 0; k < g.nplanes; ++k)
    bitblt(base + start[k], dpitch[k], p[k], pitch[k], g.row[k], g.height[k]);
  ScriptEnvironment::set_buffer(b, base, size);
  props->rw = g_vsapi->createMap();
  props->owned = true;
  g_vsapi->copyMap(g_vsapi->getFramePropertiesRO(f), props->rw);
  g_vsapi->freeFrame(f);
  if (debug_log())
    fprintf(stderr, "mvtools_vsbridge: VS planes too far apart for AviSynth offsets, frame copied\n");
  VideoFrame *vf = g.nplanes == 1
                       ? ScriptEnvironment::frame1(b, props, dpitch[0], g.row[0], g.height[0], vi.pixel_type)
                       : ScriptEnvironment::frame3(b, props, dpitch[0], g.row[0], g.height[0], (int)start[1],
                                                   (int)start[2], dpitch[1], g.row[1], g.height[1], vi.pixel_type);
  return PVideoFrame(vf);
}

// ---------------------------------------------------------------------------------------------------------------------
// registry: the AviSynth VideoInfo of every node the bridge created (decision 2)

std::mutex g_registry_mutex;
std::unordered_map<VSNode *, VideoInfo> g_registry;

bool registry_get(VSNode *node, VideoInfo &vi)
{
  std::lock_guard<std::mutex> l(g_registry_mutex);
  auto it = g_registry.find(node);
  if (it == g_registry.end())
    return false;
  vi = it->second;
  return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// VSClip: a VS node as an AviSynth clip

class VSClip;
std::mutex g_clips_mutex;
std::unordered_set<IClip *> g_clips; // live VSClips, to recognise a filter that returns one of its inputs

// One VSClip per VS node, shared by every filter that takes the node as an argument: in AviSynth+ all consumers of a
// script variable hold the same IClip, and the GPU path keys its shared uploads by that pointer
// (GpuBackend::frame_key(child, n)), so one wrapper per consumer would upload each super picture once per consumer.
// The map holds a reference of its own plus the number of filters using the entry (acquire_vsclip / release_vsclip), so
// a lookup never revives a clip whose last reference is just going away.
struct NodeClip
{
  PClip clip;
  int users = 0;
};
std::unordered_map<VSNode *, NodeClip> g_node_clips; // under g_clips_mutex

class VSClip : public IClip
{
public:
  VSNode *node;
  VideoInfo vi;

  VSClip(VSNode *n, const VideoInfo &v) : node(n), vi(v) {} // only through acquire_vsclip (under g_clips_mutex)
  ~VSClip() override
  {
    {
      std::lock_guard<std::mutex> l(g_clips_mutex);
      g_clips.erase(this);
    }
    g_vsapi->freeNode(node);
  }
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override;
  bool __stdcall GetParity(int) override { return vi.IsTFF(); }
  void __stdcall GetAudio(void *, int64_t, int64_t, IScriptEnvironment *) override {}
  int __stdcall SetCacheHints(int cachehints, int) override
  {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }
  const VideoInfo &__stdcall GetVideoInfo() override { return vi; }
};

bool is_vsclip(IClip *c)
{
  std::lock_guard<std::mutex> l(g_clips_mutex);
  return g_clips.count(c) != 0;
}

// The shared VSClip of node, for one more filter. Takes over the node reference.
VSClip *acquire_vsclip(VSNode *node, const VideoInfo &vi)
{
  std::lock_guard<std::mutex> l(g_clips_mutex);
  NodeClip &e = g_node_clips[node];
  if (e.users++ == 0)
  {
    VSClip *c = new VSClip(node, vi);
    g_clips.insert(c);
    e.clip = c;
  }
  else
    g_vsapi->freeNode(node); // the existing VSClip holds a reference
  if (debug_log())
    fprintf(stderr, "mvtools_vsbridge: node %p -> clip %p, %d users\n", (void *)node, (void *)e.clip.operator->(),
            e.users);
  return static_cast<VSClip *>(e.clip.operator->());
}

// One filter less uses c; the last one drops the map's reference (the clip goes when the plugin's references go).
void release_vsclip(VSClip *c)
{
  PClip drop;
  {
    std::lock_guard<std::mutex> l(g_clips_mutex);
    auto it = g_node_clips.find(c->node);
    if (it == g_node_clips.end() || --it->second.users != 0)
      return;
    drop = it->second.clip;
    g_node_clips.erase(it);
  }
  // drop released here, outside the lock: ~VSClip takes it
}

// releases the VSClips of a create call that did not become a filter
struct ClipUses
{
  std::vector<VSClip *> v;
  ~ClipUses()
  {
    for (VSClip *c : v)
      release_vsclip(c);
  }
};

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// filters

struct FuncDef;

struct InputRule
{
  VSNode *node = nullptr;
  int frames = 0;
  std::mutex m;
  std::vector<int> offsets{0}; // learned: input frame - base(n)
};

struct BridgeFilter
{
  std::string name;
  const FuncDef *def = nullptr;
  AVSValue args;            // to create more instances
  int mtmode = 0;
  std::mutex pool_m;
  std::vector<PClip> all;   // every instance, all[0] the first
  std::vector<IClip *> idle; // instances not in a GetFrame call (MT_MULTI_INSTANCE)
  VideoInfo avs_vi;
  VSVideoInfo vs_vi;
  Geometry geo;
  std::vector<std::unique_ptr<InputRule>> inputs;
  std::vector<VSClip *> input_clips;
  int out_frames = 0;
  int64_t dur_num = 0, dur_den = 0; // set on every output frame when the rate differs from the first input's
  VSNode *self = nullptr;
  std::atomic<int> misses{0};
  std::atomic<int> last_miss{-1}; // highest output frame that had a miss (length of the warm-up)
  std::mutex create_m; // one new instance at a time, as AviSynth+'s MT guard does per filter

  int base(int k, int n) const
  {
    const InputRule &r = *inputs[k];
    return out_frames ? (int)((int64_t)n * r.frames / out_frames) : n;
  }
  void learn(int k, int offset)
  {
    InputRule &r = *inputs[k];
    std::lock_guard<std::mutex> l(r.m);
    if (r.offsets.size() < 64 && std::find(r.offsets.begin(), r.offsets.end(), offset) == r.offsets.end())
      r.offsets.push_back(offset);
  }
  // index of input c (the first, when the same node is given twice), -1 when c is not an input of this filter
  int input_of(const VSClip *c) const
  {
    for (size_t k = 0; k < input_clips.size(); ++k)
      if (input_clips[k] == c)
        return (int)k;
    return -1;
  }
  ~BridgeFilter()
  {
    for (VSClip *c : input_clips)
      release_vsclip(c);
  }
};

namespace
{

PVideoFrame VSClip::GetFrame(int n, IScriptEnvironment *env)
{
  n = std::min(std::max(0, n), vi.num_frames - 1);
  CallCtx *c = t_call;
  const VSFrame *f = nullptr;
  if (c && c->ctx)
    f = g_vsapi->getFrameFilter(n, node, c->ctx);
  if (!f)
  {
    // not in the request set: learn the offset for the next frames, then fetch synchronously (safe in VS: the pool
    // thread is parked as reserveThread, see VAPOURSYNTH_PLAN.md step 1)
    // learned only for the filter whose GetFrame this is (not e.g. a producer regenerating a GPU frame on its behalf)
    BridgeFilter *owner = c ? c->filter : nullptr;
    const int input_index = owner ? owner->input_of(this) : -1;
    if (input_index >= 0)
    {
      const int off = n - owner->base(input_index, c->n);
      owner->learn(input_index, off);
      const int m = ++owner->misses;
      int last = owner->last_miss.load();
      while (c->n > last && !owner->last_miss.compare_exchange_weak(last, c->n))
      {
      }
      if (debug_log() && m <= 50)
        fprintf(stderr, "mvtools_vsbridge: %s frame %d: input %d frame %d not prefetched (offset %+d learned)\n",
                owner->name.c_str(), c->n, input_index, n, off);
    }
    char err[1024] = {};
    f = g_vsapi->getFrame(n, node, err, sizeof(err));
    if (!f)
      env->ThrowError("%s", err[0] ? err : "VapourSynth: getFrame failed");
  }
  return wrap_vs_frame(f, false, vi);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// functions registered by plugins

struct ArgDef
{
  std::string name; // VS argument name
  char type;        // c i f b s
  bool array;       // '*' or '+'
  bool required;
};

struct FuncDef
{
  std::string name;
  std::string params;
  IScriptEnvironment::ApplyFunc apply;
  void *user_data;
  std::vector<ArgDef> args;
};

namespace
{

std::mutex g_funcs_mutex;
std::map<std::string, FuncDef *> g_funcs; // lower-case name -> first registration
std::mutex g_register_mutex; // one plugin registration at a time

std::string lower(std::string s)
{
  for (char &ch : s)
    ch = (char)tolower((unsigned char)ch);
  return s;
}

const FuncDef *find_func(const char *name)
{
  std::lock_guard<std::mutex> l(g_funcs_mutex);
  auto it = g_funcs.find(lower(name));
  return it == g_funcs.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------------------------------------------------------------
// BridgeEnv

class BridgeEnv : public IScriptEnvironment
{
  std::mutex strings_m;
  std::deque<std::unique_ptr<char[]>> strings;
  std::mutex locks_m;
  std::map<std::string, std::unique_ptr<std::mutex>> locks;

public:
  // set while a plugin's AvisynthPluginInit3 runs; its functions are registered with VS after it returns
  VSPlugin *reg_plugin = nullptr;
  const VSPLUGINAPI *reg_api = nullptr;
  std::vector<FuncDef *> reg_pending;

  ~BridgeEnv() override {}

  int __stdcall GetCPUFlags() override { return ::GetCPUFlags(); }
  int64_t __stdcall GetCPUFlagsEx() override { return (int64_t)(unsigned)::GetCPUFlags(); }

  char *__stdcall SaveString(const char *s, int length = -1) override
  {
    if (!s)
      return nullptr;
    const size_t len = length < 0 ? strlen(s) : (size_t)length;
    std::unique_ptr<char[]> p(new char[len + 1]);
    memcpy(p.get(), s, len);
    p[len] = 0;
    std::lock_guard<std::mutex> l(strings_m);
    strings.push_back(std::move(p));
    return strings.back().get();
  }
  char *__stdcall VSprintf(const char *fmt, va_list val) override
  {
    va_list copy;
    va_copy(copy, val);
    const int len = vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    if (len < 0)
      return SaveString(fmt);
    std::vector<char> buf((size_t)len + 1);
    vsnprintf(buf.data(), buf.size(), fmt, val);
    return SaveString(buf.data(), len);
  }
  char *Sprintf(const char *fmt, ...) override
  {
    va_list val;
    va_start(val, fmt);
    char *r = VSprintf(fmt, val);
    va_end(val);
    return r;
  }
  void ThrowError(const char *fmt, ...) override
  {
    va_list val;
    va_start(val, fmt);
    char *msg = VSprintf(fmt, val);
    va_end(val);
    throw AvisynthError(msg);
  }

  void __stdcall AddFunction(const char *name, const char *params, ApplyFunc apply, void *user_data) override;
  bool __stdcall FunctionExists(const char *name) override
  {
    return find_func(name) || !strcmp(lower(name).c_str(), "internalcache") || !strcmp(lower(name).c_str(), "cache");
  }
  AVSValue __stdcall Invoke(const char *name, const AVSValue args, const char *const *arg_names = 0) override;
  bool __stdcall InvokeTry(AVSValue *result, const char *name, const AVSValue &args,
                           const char *const *arg_names = 0) override
  {
    try
    {
      *result = Invoke(name, args, arg_names);
      return true;
    }
    catch (const NotFound &)
    {
      return false;
    }
  }
  AVSValue __stdcall Invoke2(const AVSValue &, const char *name, const AVSValue args,
                             const char *const *arg_names = 0) override
  {
    return Invoke(name, args, arg_names);
  }
  bool __stdcall Invoke2Try(AVSValue *result, const AVSValue &, const char *name, const AVSValue args,
                            const char *const *arg_names = 0) override
  {
    return InvokeTry(result, name, args, arg_names);
  }
  AVSValue __stdcall Invoke3(const AVSValue &, const PFunction &, const AVSValue, const char *const * = 0) override
  {
    ThrowError("Invoke3 (function objects) is not supported under VapourSynth");
    return AVSValue();
  }
  bool __stdcall Invoke3Try(AVSValue *, const AVSValue &, const PFunction &, const AVSValue,
                            const char *const * = 0) override
  {
    return false;
  }

  // script variables: none under VapourSynth
  AVSValue __stdcall GetVar(const char *) override { throw NotFound(); }
  AVSValue __stdcall GetVarDef(const char *, const AVSValue &def = AVSValue()) override { return def; }
  bool __stdcall GetVarTry(const char *, AVSValue *) const override { return false; }
  bool __stdcall GetVarBool(const char *, bool def) const override { return def; }
  int __stdcall GetVarInt(const char *, int def) const override { return def; }
  double __stdcall GetVarDouble(const char *, double def) const override { return def; }
  const char *__stdcall GetVarString(const char *, const char *def) const override { return def; }
  int64_t __stdcall GetVarLong(const char *, int64_t def) const override { return def; }
  bool __stdcall SetVar(const char *, const AVSValue &) override { return true; }
  bool __stdcall SetGlobalVar(const char *, const AVSValue &) override { return true; }
  void __stdcall PushContext(int = 0) override {}
  void __stdcall PopContext() override {}

  PVideoFrame __stdcall NewVideoFrame(const VideoInfo &vi, int align = FRAME_ALIGN) override
  {
    return NewVideoFrameP(vi, nullptr, align);
  }
  PVideoFrame __stdcall NewVideoFrameP(const VideoInfo &vi, const PVideoFrame *prop_src, int = FRAME_ALIGN) override
  {
    VSCore *core = current_core();
    if (!core)
      ThrowError("mvtools_vsbridge: NewVideoFrame outside a filter call");
    VSVideoFormat fmt;
    int w, h;
    std::string why;
    if (!vs_format(vi, core, fmt, w, h, why))
      ThrowError("mvtools_vsbridge: NewVideoFrame: %s", why.c_str());
    const VSFrame *psrc = nullptr;
    const AVSMap *pmap = nullptr;
    if (prop_src && *prop_src)
    {
      const VideoFrame *sf = prop_src->operator->();
      const BridgeVFB *sb = static_cast<const BridgeVFB *>(sf->GetFrameBuffer());
      pmap = ScriptEnvironment::props(sf);
      // the VS frame's own properties only when this view did not change them (owned map = changed or copied)
      if (sb->vsframe && pmap && !pmap->owned)
        psrc = sb->vsframe;
    }
    VSFrame *dst = g_vsapi->newVideoFrame(&fmt, w, h, psrc, core);
    if (!dst)
      ThrowError("mvtools_vsbridge: newVideoFrame failed (%dx%d)", w, h);
    if (!psrc && pmap && pmap->get())
      g_vsapi->copyMap(pmap->get(), g_vsapi->getFramePropertiesRW(dst));
    // vector clips: MAnalyse leaves the header gap (bytes 4 + sizeof(MVAnalysisData) .. 255) unwritten; zero it so the
    // clip is deterministic and equal to AviSynth+'s (small frames, cheap)
    if (is_packed(vi))
      memset(g_vsapi->getWritePtr(dst, 0), 0, (size_t)g_vsapi->getStride(dst, 0) * h);
    return wrap_vs_frame(dst, true, vi);
  }

  bool __stdcall MakeWritable(PVideoFrame *pvf) override
  {
    VideoFrame *f = pvf->operator->();
    if (f->IsWritable())
      return false;
    VideoInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.pixel_type = f->GetPixelType();
    vi.width = f->GetRowSize() / std::max(1, vi.BytesFromPixels(1));
    vi.height = f->GetHeight();
    PVideoFrame dst = NewVideoFrameP(vi, pvf);
    const Geometry g = geometry(vi);
    for (int k = 0; k < g.nplanes; ++k)
    {
      const int pl = g.avsplane[k];
      BitBlt(dst->GetWritePtr(pl), dst->GetPitch(pl), f->GetReadPtr(pl), f->GetPitch(pl), f->GetRowSize(pl),
             f->GetHeight(pl));
    }
    *pvf = dst;
    return true;
  }
  bool __stdcall MakePropertyWritable(PVideoFrame *pvf) override
  {
    VideoFrame *f = pvf->operator->();
    if (f->IsPropertyWritable() && ScriptEnvironment::props(f) && ScriptEnvironment::props(f)->rw)
      return false;
    *pvf = PVideoFrame(ScriptEnvironment::clone_view(f));
    return true;
  }

  void __stdcall BitBlt(BYTE *dstp, int dst_pitch, const BYTE *srcp, int src_pitch, int row_size, int height) override
  {
    bitblt(dstp, dst_pitch, srcp, src_pitch, (size_t)std::max(0, row_size), height);
  }
  void __stdcall AtExit(ShutdownFunc, void *) override {}
  void __stdcall CheckVersion(int version = AVISYNTH_INTERFACE_VERSION) override
  {
    if (version > AVISYNTH_INTERFACE_VERSION)
      ThrowError("Plugin was designed for a later version of AviSynth (%d)", version);
  }
  PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
                                 int new_height) override
  {
    return PVideoFrame(src->Subframe(rel_offset, new_pitch, new_row_size, new_height));
  }
  PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
                                       int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) override
  {
    return PVideoFrame(
        src->Subframe(rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV));
  }
  PVideoFrame __stdcall SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
                                        int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV,
                                        int rel_offsetA) override
  {
    return PVideoFrame(src->Subframe(rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV,
                                     new_pitchUV, rel_offsetA));
  }
  int __stdcall SetMemoryMax(int) override { return 0; }
  int __stdcall SetWorkingDir(const char *) override { return -1; }
  void *__stdcall ManageCache(int, void *) override { return nullptr; }
  bool __stdcall PlanarChromaAlignment(PlanarChromaAlignmentMode) override { return true; }
  void __stdcall DeleteScriptEnvironment() override {}
  void __stdcall ApplyMessage(PVideoFrame *, const VideoInfo &, const char *, int, int, int, int) override {}
  void __stdcall ApplyMessageEx(PVideoFrame *, const VideoInfo &, const char *, int, int, int, int, bool) override {}
  const AVS_Linkage *__stdcall GetAVSLinkage() override { return bridge_linkage(); }

  size_t __stdcall GetEnvProperty(AvsEnvProperty prop) override
  {
    switch (prop)
    {
    case AEP_INTERFACE_VERSION: return AVISYNTH_INTERFACE_VERSION;
    case AEP_INTERFACE_BUGFIX: return AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION;
    case AEP_PHYSICAL_CPUS:
    case AEP_LOGICAL_CPUS: return std::max(1u, std::thread::hardware_concurrency());
    case AEP_THREADPOOL_THREADS:
    {
      VSCore *core = current_core();
      if (!core)
        return 1;
      VSCoreInfo ci;
      g_vsapi->getCoreInfo(core, &ci);
      return (size_t)std::max(1, ci.numThreads);
    }
    case AEP_FILTERCHAIN_THREADS: return 1;
    case AEP_THREAD_ID: return 0;
    case AEP_NUM_DEVICES: return 1;
    case AEP_FRAME_ALIGN:
    case AEP_PLANE_ALIGN: return FRAME_ALIGN;
    default: return 0;
    }
  }
  void *__stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType) override
  {
    if (alignment <= 64)
      return aligned_alloc64(nBytes ? nBytes : 1);
#ifdef _WIN32
    return _aligned_malloc(nBytes ? nBytes : 1, alignment);
#else
    void *p = nullptr;
    return posix_memalign(&p, alignment, nBytes ? nBytes : 1) == 0 ? p : nullptr;
#endif
  }
  void __stdcall Free(void *ptr) override
  {
    if (ptr)
      aligned_free64(ptr);
  }
  bool __stdcall AcquireGlobalLock(const char *name) override
  {
    std::mutex *m;
    {
      std::lock_guard<std::mutex> l(locks_m);
      auto &slot = locks[name ? name : ""];
      if (!slot)
        slot.reset(new std::mutex);
      m = slot.get();
    }
    m->lock();
    return true;
  }
  void __stdcall ReleaseGlobalLock(const char *name) override
  {
    std::lock_guard<std::mutex> l(locks_m);
    auto it = locks.find(name ? name : "");
    if (it != locks.end())
      it->second->unlock();
  }

  // frame properties
  void __stdcall copyFrameProps(const PVideoFrame &src, PVideoFrame &dst) override
  {
    const AVSMap *s = ScriptEnvironment::props(src.operator->());
    AVSMap *d = getFramePropsRW(dst);
    if (!d || !d->rw)
      return;
    g_vsapi->clearMap(d->rw);
    if (s && s->get())
      g_vsapi->copyMap(s->get(), d->rw);
  }
  const AVSMap *__stdcall getFramePropsRO(const PVideoFrame &frame) override
  {
    return ScriptEnvironment::props(frame.operator->());
  }
  AVSMap *__stdcall getFramePropsRW(PVideoFrame &frame) override
  {
    AVSMap *&m = ScriptEnvironment::props(frame.operator->());
    if (!m)
      m = copy_props(nullptr);
    else if (!m->rw)
    {
      // read-only view of an input frame: this frame gets a copy of its own
      AVSMap *c = copy_props(m);
      delete m;
      m = c;
    }
    return m;
  }
  int __stdcall propNumKeys(const AVSMap *map) override { return map ? g_vsapi->mapNumKeys(map->get()) : 0; }
  const char *__stdcall propGetKey(const AVSMap *map, int index) override
  {
    return g_vsapi->mapGetKey(map->get(), index);
  }
  int __stdcall propNumElements(const AVSMap *map, const char *key) override
  {
    return g_vsapi->mapNumElements(map->get(), key);
  }
  char __stdcall propGetType(const AVSMap *map, const char *key) override
  {
    switch (g_vsapi->mapGetType(map->get(), key))
    {
    case ptInt: return PROPTYPE_INT;
    case ptFloat: return PROPTYPE_FLOAT;
    case ptData: return PROPTYPE_DATA;
    case ptVideoNode: return PROPTYPE_CLIP;
    case ptVideoFrame: return PROPTYPE_FRAME;
    default: return PROPTYPE_UNSET;
    }
  }
  int64_t __stdcall propGetInt(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetInt(map->get(), key, index, error);
  }
  int __stdcall propGetIntSaturated(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetIntSaturated(map->get(), key, index, error);
  }
  double __stdcall propGetFloat(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetFloat(map->get(), key, index, error);
  }
  float __stdcall propGetFloatSaturated(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetFloatSaturated(map->get(), key, index, error);
  }
  const char *__stdcall propGetData(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetData(map->get(), key, index, error);
  }
  int __stdcall propGetDataSize(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetDataSize(map->get(), key, index, error);
  }
  int __stdcall propGetDataTypeHint(const AVSMap *map, const char *key, int index, int *error) override
  {
    return g_vsapi->mapGetDataTypeHint(map->get(), key, index, error);
  }
  PClip __stdcall propGetClip(const AVSMap *, const char *, int, int *error) override
  {
    if (error)
      *error = GETPROPERROR_TYPE;
    return PClip();
  }
  const PVideoFrame __stdcall propGetFrame(const AVSMap *, const char *, int, int *error) override
  {
    if (error)
      *error = GETPROPERROR_TYPE;
    return PVideoFrame();
  }
  const int64_t *__stdcall propGetIntArray(const AVSMap *map, const char *key, int *error) override
  {
    return g_vsapi->mapGetIntArray(map->get(), key, error);
  }
  const double *__stdcall propGetFloatArray(const AVSMap *map, const char *key, int *error) override
  {
    return g_vsapi->mapGetFloatArray(map->get(), key, error);
  }
  int __stdcall propDeleteKey(AVSMap *map, const char *key) override
  {
    return map->rw ? g_vsapi->mapDeleteKey(map->rw, key) : 0;
  }
  int __stdcall propSetInt(AVSMap *map, const char *key, int64_t i, int append) override
  {
    if (!map->rw || append == PROPAPPENDMODE_TOUCH)
      return 0;
    return g_vsapi->mapSetInt(map->rw, key, i, append);
  }
  int __stdcall propSetFloat(AVSMap *map, const char *key, double d, int append) override
  {
    if (!map->rw || append == PROPAPPENDMODE_TOUCH)
      return 0;
    return g_vsapi->mapSetFloat(map->rw, key, d, append);
  }
  int __stdcall propSetData(AVSMap *map, const char *key, const char *d, int length, int append) override
  {
    return propSetDataH(map, key, d, length, dtUnknown, append);
  }
  int __stdcall propSetDataH(AVSMap *map, const char *key, const char *d, int length, int type, int append) override
  {
    if (!map->rw || append == PROPAPPENDMODE_TOUCH)
      return 0;
    return g_vsapi->mapSetData(map->rw, key, d, length, type, append);
  }
  int __stdcall propSetClip(AVSMap *, const char *, PClip &, int) override { return 1; }
  int __stdcall propSetFrame(AVSMap *, const char *, const PVideoFrame &, int) override { return 1; }
  int __stdcall propSetIntArray(AVSMap *map, const char *key, const int64_t *i, int size) override
  {
    return map->rw ? g_vsapi->mapSetIntArray(map->rw, key, i, size) : 1;
  }
  int __stdcall propSetFloatArray(AVSMap *map, const char *key, const double *d, int size) override
  {
    return map->rw ? g_vsapi->mapSetFloatArray(map->rw, key, d, size) : 1;
  }
  AVSMap *__stdcall createMap() override { return copy_props(nullptr); }
  void __stdcall freeMap(AVSMap *map) override { delete map; }
  void __stdcall clearMap(AVSMap *map) override
  {
    if (map && map->rw)
      g_vsapi->clearMap(map->rw);
  }
};

BridgeEnv g_env;

// Calls a registered function with AviSynth arguments: positional values in order, named ones by name.
AVSValue call_function(const FuncDef &def, const AVSValue &args, const char *const *arg_names)
{
  const int n_in = args.IsArray() ? args.ArraySize() : 1;
  std::vector<AVSValue> out(def.args.size());
  size_t pos = 0;
  for (int i = 0; i < n_in; ++i)
  {
    const AVSValue &v = args.IsArray() ? args[i] : args;
    const char *nm = arg_names ? arg_names[i] : nullptr;
    if (nm)
    {
      size_t k = 0;
      while (k < def.args.size() && lower(def.args[k].name) != lower(nm))
        ++k;
      if (k == def.args.size())
        g_env.ThrowError("%s: unknown argument '%s'", def.name.c_str(), nm);
      out[k] = v;
    }
    else
    {
      if (pos >= def.args.size())
        g_env.ThrowError("%s: too many arguments", def.name.c_str());
      if (def.args[pos].array && !v.IsArray())
      {
        // varargs: gather the remaining positional values
        std::vector<AVSValue> rest;
        for (; i < n_in && !(arg_names && arg_names[i]); ++i)
          rest.push_back(args.IsArray() ? args[i] : args);
        --i;
        out[pos++] = AVSValue(rest.data(), (int)rest.size());
      }
      else
        out[pos++] = v;
    }
  }
  return def.apply(AVSValue(out.data(), (int)out.size()), def.user_data, &g_env);
}

// AviSynth core filters that cudafilters Invoke as their CPU fallback. Each cudafilter builds its fallback in the
// constructor (and takes its VideoInfo from it) even when the GPU computes every frame; it pulls fallback frames only
// when the GPU is unusable (library or device missing, format not covered) or fails for a frame (device allocation
// failed, kernel error). A stand-in has the VideoInfo the AviSynth+ filter would report, so the GPU path runs and stays
// bit-identical. What a fallback frame is, is a global choice (VAPOURSYNTH_PLAN.md, step 4):
//   MVBRIDGE_FALLBACK=strict   (default) an error naming the missing filter: VS output is AviSynth+'s or an error
//   MVBRIDGE_FALLBACK=nearest  the closest VapourSynth filter (std.Merge, ...): not bit-identical to AviSynth+; names
//                              without a mapping stay an error
// Read when the stand-in is created, so a script can set it (os.environ) before creating the filters.
enum class FallbackMode
{
  strict,
  nearest
};

FallbackMode fallback_mode()
{
  const char *v = std::getenv("MVBRIDGE_FALLBACK");
  if (v && !strcmp(lower(v).c_str(), "nearest"))
    return FallbackMode::nearest;
  if (v && *v && strcmp(lower(v).c_str(), "strict"))
  {
    static std::once_flag warned;
    std::call_once(warned, [v] {
      fprintf(stderr, "mvtools_vsbridge: MVBRIDGE_FALLBACK=%s not known (strict or nearest), using strict\n", v);
    });
  }
  return FallbackMode::strict;
}

// a VS node computing the AviSynth clip c (defined after create_filter)
VSNode *clip_node(const PClip &c, const std::string &name, VSCore *core, std::string &err);
// the VideoInfo of a node the bridge did not create; acquire_vsclip takes the node
VSClip *node_clip(VSNode *node, std::string &err)
{
  VideoInfo vi;
  if (!registry_get(node, vi) && !avs_video_info(*g_vsapi->getVideoInfo(node), vi, err))
  {
    g_vsapi->freeNode(node);
    return nullptr;
  }
  return acquire_vsclip(node, vi);
}

class StandInClip : public IClip
{
  std::string name, nearest_name; // nearest_name: the VS filter behind `nearest`, "" when strict / no mapping
  PClip child, nearest;           // nearest: that filter's node as a clip (a shared VSClip, released at the end)
  VideoInfo vi;
  bool has_mapping;
  std::once_flag logged;

public:
  StandInClip(const std::string &n, const PClip &c, const VideoInfo &v, bool mapping)
      : name(n), child(c), vi(v), has_mapping(mapping)
  {
  }
  ~StandInClip() override
  {
    if (nearest)
      release_vsclip(static_cast<VSClip *>(nearest.operator->()));
  }
  void set_nearest(VSClip *c, const std::string &vs_name)
  {
    nearest = c;
    nearest_name = vs_name;
  }
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
  {
    if (nearest)
    {
      std::call_once(logged, [&] {
        fprintf(stderr, "mvtools_vsbridge: %s (CPU fallback of a cudafilters filter) from %s, first at frame %d: not "
                        "bit-identical to AviSynth+ (MVBRIDGE_FALLBACK=nearest)\n",
                name.c_str(), nearest_name.c_str(), n);
      });
      return nearest->GetFrame(n, env);
    }
    env->ThrowError("%s (CPU fallback of a cudafilters filter) is not available under VapourSynth; frame %d needed "
                    "it (GPU library missing, format not supported on the GPU, or the GPU failed for this frame; "
                    "MVTOOLS_GPU_LOG=1 tells which).%s",
                    name.c_str(), n,
                    has_mapping ? " MVBRIDGE_FALLBACK=nearest uses the closest VapourSynth filter instead (not "
                                  "bit-identical to AviSynth+)."
                                : "");
    return PVideoFrame();
  }
  bool __stdcall GetParity(int n) override { return child->GetParity(n); }
  void __stdcall GetAudio(void *, int64_t, int64_t, IScriptEnvironment *) override {}
  int __stdcall SetCacheHints(int cachehints, int) override
  {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }
  const VideoInfo &__stdcall GetVideoInfo() override { return vi; }
};

// Invoke arguments by AviSynth parameter position: positional values first, then named ones at the position of their
// name in `params` (lower case); undefined where not given
std::vector<AVSValue> invoke_args(const AVSValue &args, const char *const *arg_names,
                                  const std::vector<std::string> &params)
{
  std::vector<AVSValue> out(params.size());
  const int n_in = args.IsArray() ? args.ArraySize() : 1;
  size_t pos = 0;
  for (int i = 0; i < n_in; ++i)
  {
    const AVSValue &v = args.IsArray() ? args[i] : args;
    const char *nm = arg_names ? arg_names[i] : nullptr;
    if (nm)
    {
      const auto it = std::find(params.begin(), params.end(), lower(nm));
      if (it != params.end())
        out[(size_t)(it - params.begin())] = v;
    }
    else if (pos < params.size())
      out[pos++] = v;
  }
  return out;
}

// A filter of a VapourSynth core plugin (com.vapoursynth.std / .resize) on `in` (freed here): its node, or null and err.
VSNode *vs_call(const char *plugin_id, const char *func, VSMap *in, std::string &err)
{
  VSCore *core = current_core();
  VSPlugin *p = core ? g_vsapi->getPluginByID(plugin_id, core) : nullptr;
  if (!p)
  {
    g_vsapi->freeMap(in);
    err = std::string("VapourSynth plugin ") + plugin_id + " not found";
    return nullptr;
  }
  VSMap *out = g_vsapi->invoke(p, func, in);
  g_vsapi->freeMap(in);
  if (const char *e = g_vsapi->mapGetError(out))
  {
    err = std::string(func) + ": " + e;
    g_vsapi->freeMap(out);
    return nullptr;
  }
  VSNode *node = g_vsapi->mapGetNode(out, "clip", 0, nullptr);
  g_vsapi->freeMap(out);
  return node;
}

// the input clips of a nearest mapping as VS nodes appended to `key` of `in`; false and err on failure
bool append_clips(VSMap *in, const char *key, const std::vector<PClip> &clips, const char *name, std::string &err)
{
  VSCore *core = current_core();
  if (!core)
  {
    err = "no VapourSynth core";
    return false;
  }
  for (const PClip &c : clips)
  {
    VSNode *n = clip_node(c, std::string(name) + "FallbackInput", core, err);
    if (!n)
      return false;
    g_vsapi->mapConsumeNode(in, key, n, maAppend);
  }
  return true;
}

// masktools plane mode (Y / U / V: 3 process, 2 copy, 1 none, 4 copy second) -> the std.Expr expression of the plane;
// "" for modes without one (memset values)
std::string mt_plane_expr(const AVSValue &v, int def, const std::string &process, int nclips)
{
  const int m = v.Defined() ? (int)v.AsFloat((float)def) : def;
  return m == 3 ? process : (m == 2 || m == 1) ? "x" : (m == 4 && nclips > 1) ? "y" : "";
}

// std.Expr over clips with one expression per plane (masktools Y / U / V modes at a[iy], a[iy + 1], a[iy + 2]); null
// and err when a plane mode has no expression
VSNode *nearest_mt_expr(const std::vector<PClip> &clips, const std::vector<AVSValue> &a, size_t iy,
                        const std::string &process, const char *name, std::string &err)
{
  const int np = clips[0]->GetVideoInfo().NumComponents();
  std::string ex[3];
  for (int p = 0; p < np; ++p)
  {
    ex[p] = mt_plane_expr(a[iy + p], p == 0 ? 3 : 1, process, (int)clips.size());
    if (ex[p].empty())
    {
      err = "this plane mode has no VapourSynth equivalent";
      return nullptr;
    }
  }
  VSMap *in = g_vsapi->createMap();
  if (!append_clips(in, "clips", clips, name, err))
  {
    g_vsapi->freeMap(in);
    return nullptr;
  }
  for (int p = 0; p < np; ++p)
    g_vsapi->mapSetData(in, "expr", ex[p].c_str(), (int)ex[p].size(), dtUtf8, maAppend);
  return vs_call("com.vapoursynth.std", "Expr", in, err);
}

// The stand-in for AviSynth core / plugin filter `l` (lower case) that cudafilters Invoke as their CPU fallback:
// AviSynth parameter names (positional ones first, in order), the VideoInfo rule (the same VideoInfo calls as the
// filter's constructor; checked against AviSynth+ by vsbridge/test/stand_ins.py) and the nearest VS mapping, if any.
struct StandInSpec
{
  const char *name;
  std::vector<std::string> params;
  bool (*video_info)(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err);
  // nearest: the VS node computing an approximation, or null and err (null pointer: no mapping)
  VSNode *(*nearest)(const std::string &l, const std::vector<AVSValue> &a, std::string &err);
  const char *vs_name; // the VS filter(s) behind nearest, for the log
};

bool vi_first(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &)
{
  vi = a[0].AsClip()->GetVideoInfo();
  return true;
}

// same format and size as the first clip (masktools' and Merge's requirement), for clips at the given positions
bool vi_first_same(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err, size_t nclips)
{
  vi = a[0].AsClip()->GetVideoInfo();
  for (size_t k = 1; k < nclips; ++k)
  {
    if (!a[k].IsClip())
      continue;
    const VideoInfo &v = a[k].AsClip()->GetVideoInfo();
    if (v.pixel_type != vi.pixel_type || v.width != vi.width || v.height != vi.height)
    {
      err = "the clips must have the same format and size";
      return false;
    }
  }
  return true;
}
bool vi_first2(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err) { return vi_first_same(a, vi, err, 2); }
bool vi_first3(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err) { return vi_first_same(a, vi, err, 3); }

// AviSynth+ Bob (field.cpp, Create_Bob): SeparateFields when frame-based, FilteredResizeV to `height`, Fieldwise
bool vi_bob(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err)
{
  vi = a[0].AsClip()->GetVideoInfo();
  if (!vi.IsFieldBased())
  {
    if (vi.height & 1)
      return err = "SeparateFields: height must be even", false;
    if (vi.Is420() && vi.height & 3)
      return err = "SeparateFields: YUV420 height must be multiple of 4", false;
    vi.height >>= 1;
    vi.MulDivFPS(2, 1);
    vi.num_frames *= 2;
    if (vi.num_frames < 0)
      return err = "SeparateFields: Maximum number of frames exceeded.", false;
    vi.SetFieldBased(true);
  }
  const int h = a[3].AsInt(vi.height * 2);
  if (h <= 0)
    return err = "Resize: Height must be greater than 0.", false;
  if (vi.IsPlanar() && !vi.IsY() && !vi.IsRGB() && (h & ((1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1)))
    return err = "Resize: Planar destination height must be a multiple of " +
                 std::to_string(1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) + ".",
           false;
  vi.height = h;
  vi.SetFieldBased(false); // Fieldwise: frame-based, IT_BFF / IT_TFF kept
  return true;
}

// AviSynth+ resizers (resample.cpp, FilteredResizeH / V): the target size, with their size checks
bool vi_resize(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err)
{
  vi = a[0].AsClip()->GetVideoInfo();
  const int w = a[1].AsInt(0), h = a[2].AsInt(0);
  if (w <= 0)
    return err = "Resize: Width must be greater than 0.", false;
  if (h <= 0)
    return err = "Resize: Height must be greater than 0.", false;
  if (vi.IsPlanar() && !vi.IsY() && !vi.IsRGB())
  {
    const int mw = (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1, mh = (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1;
    if ((w & mw) && w != vi.width)
      return err = "Resize: Planar destination height must be a multiple of " + std::to_string(mw + 1) + ".", false;
    if ((h & mh) && h != vi.height)
      return err = "Resize: Planar destination height must be a multiple of " + std::to_string(mh + 1) + ".", false;
  }
  vi.width = w;
  vi.height = h;
  return true;
}

// nnedi3 (plugins_JPSDR; measured on AviSynth+ 3.7.5, vsbridge/test/stand_ins.py): field -2 / 2 / 3 double the rate,
// dh doubles the height (field -1 / 0 / 1 only), the result is frame-based (IT_BFF / IT_TFF kept)
bool vi_nnedi3(const std::vector<AVSValue> &a, VideoInfo &vi, std::string &err)
{
  vi = a[0].AsClip()->GetVideoInfo();
  const int field = a[1].AsInt(-1);
  const bool dh = a[2].AsBool(false);
  if (field < -2 || field > 3)
    return err = "nnedi3: field must be set to -2, -1, 0, 1, 2, or 3!", false;
  if (dh && (field < -1 || field > 1))
    return err = "nnedi3: field must be set to -1, 0, or 1 when dh=true!", false;
  if (field > 1 || field == -2)
  {
    vi.num_frames *= 2;
    vi.MulDivFPS(2, 1);
  }
  if (dh)
    vi.height *= 2;
  vi.SetFieldBased(false);
  return true;
}

// nearest for Merge / MergeLuma / MergeChroma: std.Merge with per-plane weights (AviSynth+: Merge weights all planes,
// MergeLuma only Y, MergeChroma only U and V; default weight 0.5 / 1.0 / 1.0)
VSNode *nearest_merge(const std::string &l, const std::vector<AVSValue> &a, std::string &err)
{
  const double w = a[2].Defined() ? a[2].AsFloat() : (l == "merge" ? 0.5 : 1.0);
  VSMap *in = g_vsapi->createMap();
  if (!append_clips(in, "clipa", {a[0].AsClip()}, "Merge", err) || !append_clips(in, "clipb", {a[1].AsClip()}, "Merge", err))
  {
    g_vsapi->freeMap(in);
    return nullptr;
  }
  const bool gray = a[0].AsClip()->GetVideoInfo().NumComponents() == 1;
  if (l == "merge" || gray)
    g_vsapi->mapSetFloat(in, "weight", l == "mergechroma" ? 0.0 : w, maAppend);
  else
  {
    g_vsapi->mapSetFloat(in, "weight", l == "mergeluma" ? w : 0.0, maAppend);
    g_vsapi->mapSetFloat(in, "weight", l == "mergeluma" ? 0.0 : w, maAppend);
  }
  return vs_call("com.vapoursynth.std", "Merge", in, err);
}

// nearest for Bob: std.SeparateFields, each field resized with resize.Bicubic (b, c) and shifted by -/+0.25 lines as
// AviSynth+ does (Fieldwise: top fields +0.25, bottom fields -0.25), frame-based input only
VSNode *nearest_bob(const std::string &, const std::vector<AVSValue> &a, std::string &err)
{
  const PClip c = a[0].AsClip();
  const VideoInfo &vi = c->GetVideoInfo();
  if (vi.IsFieldBased())
  {
    err = "field-based input has no VapourSynth equivalent";
    return nullptr;
  }
  const bool tff = c->GetParity(0);
  VSMap *in = g_vsapi->createMap();
  if (!append_clips(in, "clip", {c}, "Bob", err))
  {
    g_vsapi->freeMap(in);
    return nullptr;
  }
  g_vsapi->mapSetInt(in, "tff", tff ? 1 : 0, maAppend);
  VSNode *sep = vs_call("com.vapoursynth.std", "SeparateFields", in, err);
  if (!sep)
    return nullptr;
  const int fh = vi.height / 2, h = a[3].AsInt(vi.height);
  VSNode *shifted[2] = {}; // [0]: -0.25 (bottom fields), [1]: +0.25 (top fields)
  for (int k = 0; k < 2 && !err.size(); ++k)
  {
    VSMap *r = g_vsapi->createMap();
    g_vsapi->mapSetNode(r, "clip", sep, maAppend);
    g_vsapi->mapSetInt(r, "width", vi.width, maAppend);
    g_vsapi->mapSetInt(r, "height", h, maAppend);
    g_vsapi->mapSetFloat(r, "src_top", k ? 0.25 : -0.25, maAppend);
    g_vsapi->mapSetFloat(r, "src_height", fh, maAppend);
    g_vsapi->mapSetFloat(r, "filter_param_a", a[1].AsDblDef(1. / 3.), maAppend);
    g_vsapi->mapSetFloat(r, "filter_param_b", a[2].AsDblDef(1. / 3.), maAppend);
    shifted[k] = vs_call("com.vapoursynth.resize", "Bicubic", r, err);
  }
  g_vsapi->freeNode(sep);
  if (!shifted[0] || !shifted[1])
  {
    for (VSNode *n : shifted)
      if (n)
        g_vsapi->freeNode(n);
    return nullptr;
  }
  // field 2k is the top field when tff: take it from the +0.25 clip, field 2k + 1 from the -0.25 clip
  VSNode *sel[2];
  for (int k = 0; k < 2; ++k)
  {
    VSMap *s = g_vsapi->createMap();
    g_vsapi->mapSetNode(s, "clip", shifted[k == 0 ? (tff ? 1 : 0) : (tff ? 0 : 1)], maAppend);
    g_vsapi->mapSetInt(s, "cycle", 2, maAppend);
    g_vsapi->mapSetInt(s, "offsets", k, maAppend);
    sel[k] = vs_call("com.vapoursynth.std", "SelectEvery", s, err);
  }
  g_vsapi->freeNode(shifted[0]);
  g_vsapi->freeNode(shifted[1]);
  if (!sel[0] || !sel[1])
  {
    for (VSNode *n : sel)
      if (n)
        g_vsapi->freeNode(n);
    return nullptr;
  }
  VSMap *il = g_vsapi->createMap();
  g_vsapi->mapConsumeNode(il, "clips", sel[0], maAppend);
  g_vsapi->mapConsumeNode(il, "clips", sel[1], maAppend);
  return vs_call("com.vapoursynth.std", "Interleave", il, err);
}

// nearest for BilinearResize: resize.Bilinear with the same source window (AviSynth's src_width / src_height <= 0 count
// from the right / bottom edge)
VSNode *nearest_bilinear(const std::string &, const std::vector<AVSValue> &a, std::string &err)
{
  const PClip c = a[0].AsClip();
  const VideoInfo &vi = c->GetVideoInfo();
  const double left = a[3].AsDblDef(0), top = a[4].AsDblDef(0);
  double sw = a[5].AsDblDef(vi.width), sh = a[6].AsDblDef(vi.height);
  if (sw <= 0)
    sw = vi.width - left + sw;
  if (sh <= 0)
    sh = vi.height - top + sh;
  VSMap *in = g_vsapi->createMap();
  if (!append_clips(in, "clip", {c}, "BilinearResize", err))
  {
    g_vsapi->freeMap(in);
    return nullptr;
  }
  g_vsapi->mapSetInt(in, "width", a[1].AsInt(), maAppend);
  g_vsapi->mapSetInt(in, "height", a[2].AsInt(), maAppend);
  g_vsapi->mapSetFloat(in, "src_left", left, maAppend);
  g_vsapi->mapSetFloat(in, "src_top", top, maAppend);
  g_vsapi->mapSetFloat(in, "src_width", sw, maAppend);
  g_vsapi->mapSetFloat(in, "src_height", sh, maAppend);
  return vs_call("com.vapoursynth.resize", "Bilinear", in, err);
}

// nearest for mt_logic "min" / "max" (others have none): std.Expr
VSNode *nearest_mt_logic(const std::string &, const std::vector<AVSValue> &a, std::string &err)
{
  const std::string mode = lower(a[2].AsString("and"));
  if (mode != "min" && mode != "max")
  {
    err = "mode \"" + mode + "\" has no VapourSynth equivalent";
    return nullptr;
  }
  return nearest_mt_expr({a[0].AsClip(), a[1].AsClip()}, a, 5, "x y " + mode, "mt_logic", err);
}

// nearest for mt_clamp: std.Expr, max(min(x, bright + overshoot), dark - undershoot) in masktools' order (clamp.cpp:
// the upper limit first), the limits scaled as masktools scales them (paramscale "i8")
VSNode *nearest_mt_clamp(const std::string &, const std::vector<AVSValue> &a, std::string &err)
{
  const int bits = a[0].AsClip()->GetVideoInfo().BitsPerComponent();
  auto scale = [&](int v) { return v >= 255 ? (1 << bits) - 1 : v << (bits - 8); };
  const std::string e = "x y " + std::to_string(scale(a[3].AsInt(0))) + " + min z " +
                        std::to_string(scale(a[4].AsInt(0))) + " - max";
  return nearest_mt_expr({a[0].AsClip(), a[1].AsClip(), a[2].AsClip()}, a, 5, e, "mt_clamp", err);
}

// nearest for mt_makediff: std.Expr x - y + half
VSNode *nearest_mt_makediff(const std::string &, const std::vector<AVSValue> &a, std::string &err)
{
  const int bits = a[0].AsClip()->GetVideoInfo().BitsPerComponent();
  return nearest_mt_expr({a[0].AsClip(), a[1].AsClip()}, a, 2, "x y - " + std::to_string(128 << (bits - 8)) + " +",
                         "mt_makediff", err);
}

// nearest for mt_expand / mt_inpand: std.Maximum / Minimum over the mode's neighbours (thresholds left at their default)
VSNode *nearest_mt_morph(const std::string &l, const std::vector<AVSValue> &a, std::string &err)
{
  const std::string mode = lower(a[3].AsString("square"));
  static const std::map<std::string, std::vector<int>> coords = {{"square", {1, 1, 1, 1, 1, 1, 1, 1}},
                                                                 {"vertical", {0, 1, 0, 0, 0, 0, 1, 0}},
                                                                 {"horizontal", {0, 0, 0, 1, 1, 0, 0, 0}},
                                                                 {"both", {0, 1, 0, 1, 1, 0, 1, 0}}};
  const auto it = coords.find(mode);
  if (it == coords.end() || (a[1].Defined() && a[1].AsInt(255) != 255) || (a[2].Defined() && a[2].AsInt(255) != 255))
  {
    err = "mode \"" + mode + "\" or the thresholds have no VapourSynth equivalent";
    return nullptr;
  }
  const PClip c = a[0].AsClip();
  const int np = c->GetVideoInfo().NumComponents();
  VSMap *in = g_vsapi->createMap();
  if (!append_clips(in, "clip", {c}, l.c_str(), err))
  {
    g_vsapi->freeMap(in);
    return nullptr;
  }
  for (int p = 0; p < np; ++p)
  {
    const AVSValue &m = a[4 + p];
    const int v = m.Defined() ? (int)m.AsFloat(3) : (p == 0 ? 3 : 1);
    if (v == 3)
      g_vsapi->mapSetInt(in, "planes", p, maAppend);
    else if (v != 1 && v != 2)
    {
      g_vsapi->freeMap(in);
      err = "this plane mode has no VapourSynth equivalent";
      return nullptr;
    }
  }
  for (int k : it->second)
    g_vsapi->mapSetInt(in, "coordinates", k, maAppend);
  return vs_call("com.vapoursynth.std", l == "mt_expand" ? "Maximum" : "Minimum", in, err);
}

const std::vector<StandInSpec> &stand_in_specs()
{
  static const std::vector<StandInSpec> specs = {
      {"merge", {"clip", "clip2", "weight"}, vi_first2, nearest_merge, "std.Merge"},
      {"mergeluma", {"clip", "clip2", "weight"}, vi_first2, nearest_merge, "std.Merge"},
      {"mergechroma", {"clip", "clip2", "weight"}, vi_first2, nearest_merge, "std.Merge"},
      {"bob", {"clip", "b", "c", "height"}, vi_bob, nearest_bob, "std.SeparateFields + resize.Bicubic"},
      {"temporalsoften", {"clip", "radius", "luma_threshold", "chroma_threshold", "scenechange", "mode"}, vi_first, nullptr, ""},
      {"gaussresize",
       {"clip", "target_width", "target_height", "src_left", "src_top", "src_width", "src_height", "p", "b", "s", "force",
        "keep_center", "placement"},
       vi_resize, nullptr, ""},
      {"bilinearresize",
       {"clip", "target_width", "target_height", "src_left", "src_top", "src_width", "src_height", "force", "keep_center",
        "placement"},
       vi_resize, nearest_bilinear, "resize.Bilinear"},
      {"removegrain", {"clip", "mode", "modeu", "modev", "planar"}, vi_first, nullptr, ""},
      {"nnedi3",
       {"clip", "field", "dh", "y", "u", "v", "nsize", "nns", "qual", "etype", "pscrn", "threads", "opt", "fapprox"},
       vi_nnedi3, nullptr, ""},
      {"mt_logic", {"clip", "clip2", "mode", "th1", "th2", "y", "u", "v"}, vi_first2, nearest_mt_logic, "std.Expr"},
      {"mt_clamp", {"clip", "bright_limit", "dark_limit", "overshoot", "undershoot", "y", "u", "v"}, vi_first3,
       nearest_mt_clamp, "std.Expr"},
      {"mt_makediff", {"clip", "clip2", "y", "u", "v"}, vi_first2, nearest_mt_makediff, "std.Expr"},
      {"mt_expand", {"clip", "thy", "thc", "mode", "y", "u", "v"}, vi_first, nearest_mt_morph, "std.Maximum"},
      {"mt_inpand", {"clip", "thy", "thc", "mode", "y", "u", "v"}, vi_first, nearest_mt_morph, "std.Minimum"},
      {"qtgmc_keeponlybobshimmerfixes", {"input", "ref", "rep", "chroma"}, vi_first2, nullptr, ""},
  };
  return specs;
}

// the stand-in for name, or null when name has none; args / arg_names as given to Invoke
PClip stand_in(const std::string &l, const char *name, const AVSValue &args, const char *const *arg_names)
{
  const auto &specs = stand_in_specs();
  const auto spec = std::find_if(specs.begin(), specs.end(), [&](const StandInSpec &s) { return l == s.name; });
  if (spec == specs.end())
    return PClip();
  const std::vector<AVSValue> a = invoke_args(args, arg_names, spec->params);
  if (!a[0].IsClip())
    return PClip();
  VideoInfo vi;
  std::string err;
  if (!spec->video_info(a, vi, err))
    g_env.ThrowError("%s", err.c_str());
  if (debug_log())
    fprintf(stderr, "mvtools_vsbridge: stand-in %s: %dx%d pixel_type %d fps %u/%u frames %d image_type %d\n", name,
            vi.width, vi.height, vi.pixel_type, vi.fps_numerator, vi.fps_denominator, vi.num_frames, vi.image_type);
  StandInClip *s = new StandInClip(name, a[0].AsClip(), vi, spec->nearest != nullptr);
  PClip keep(s);
  if (spec->nearest && fallback_mode() == FallbackMode::nearest)
  {
    VSNode *node = spec->nearest(l, a, err);
    VSClip *c = node ? node_clip(node, err) : nullptr;
    if (!c)
      g_env.ThrowError("%s: MVBRIDGE_FALLBACK=nearest: %s", name, err.c_str());
    s->set_nearest(c, spec->vs_name); // the stand-in owns c from here
    const VideoInfo &nv = c->GetVideoInfo();
    if (nv.width != vi.width || nv.height != vi.height || nv.num_frames != vi.num_frames || nv.pixel_type != vi.pixel_type)
      g_env.ThrowError("%s: MVBRIDGE_FALLBACK=nearest: %s gives %dx%d, %d frames instead of %dx%d, %d frames", name,
                       spec->vs_name, nv.width, nv.height, nv.num_frames, vi.width, vi.height, vi.num_frames);
  }
  return keep;
}

AVSValue __stdcall BridgeEnv::Invoke(const char *name, const AVSValue args, const char *const *arg_names)
{
  const std::string l = lower(name);
  if (l == "internalcache" || l == "cache")
    return args.IsArray() ? args[0] : args;
  const FuncDef *def = find_func(name);
  if (!def)
  {
    // masktools filters the bridge computes itself, exactly (mt_port.h); then the stand-ins
    if (PClip p = mt_port_invoke(l, args, arg_names, this))
      return p;
    if (PClip s = stand_in(l, name, args, arg_names))
      return s;
    throw NotFound();
  }
  return call_function(*def, args, arg_names);
}

// ---------------------------------------------------------------------------------------------------------------------
// the VS filter around an AviSynth clip

const VSFrame *to_vs_frame(BridgeFilter &f, PVideoFrame &pf, VSCore *core)
{
  VideoFrame *fr = pf.operator->();
  BridgeVFB *b = static_cast<BridgeVFB *>(fr->GetFrameBuffer());
  const Geometry &g = f.geo;

  // the frame is exactly a VS frame of the output format: hand it over without copying
  if (b->vsframe)
  {
    const VSVideoFormat *ff = g_vsapi->getVideoFrameFormat(b->vsframe);
    bool whole = ff->colorFamily == f.vs_vi.format.colorFamily && ff->sampleType == f.vs_vi.format.sampleType &&
                 ff->bitsPerSample == f.vs_vi.format.bitsPerSample &&
                 ff->subSamplingW == f.vs_vi.format.subSamplingW && ff->subSamplingH == f.vs_vi.format.subSamplingH &&
                 g_vsapi->getFrameWidth(b->vsframe, 0) == f.vs_vi.width &&
                 g_vsapi->getFrameHeight(b->vsframe, 0) == f.vs_vi.height;
    for (int k = 0; whole && k < g.nplanes; ++k)
    {
      const int pl = g.avsplane[k];
      whole = fr->GetReadPtr(pl) == g_vsapi->getReadPtr(b->vsframe, g.vsplane[k]) &&
              fr->GetPitch(pl) == (int)g_vsapi->getStride(b->vsframe, g.vsplane[k]) &&
              fr->GetRowSize(pl) == g.row[k] && fr->GetHeight(pl) == g.height[k];
    }
    // its properties must be the VS frame's own (a view with a changed / copied map is copied below)
    const AVSMap *pm = ScriptEnvironment::props(fr);
    whole = whole && pm && !pm->owned;
    if (whole)
    {
      if (f.dur_num)
      {
        VSFrame *c = g_vsapi->copyFrame(b->vsframe, core);
        VSMap *p = g_vsapi->getFramePropertiesRW(c);
        g_vsapi->mapSetInt(p, "_DurationNum", f.dur_num, maReplace);
        g_vsapi->mapSetInt(p, "_DurationDen", f.dur_den, maReplace);
        b->readonly = true;
        return c;
      }
      b->readonly = true; // VS holds it now; the plugin may keep its reference but no longer write
      return g_vsapi->addFrameRef(b->vsframe);
    }
  }

  // otherwise copy into a new VS frame
  VSFrame *dst = g_vsapi->newVideoFrame(&f.vs_vi.format, f.vs_vi.width, f.vs_vi.height, nullptr, core);
  for (int k = 0; k < g.nplanes; ++k)
  {
    const int pl = g.avsplane[k];
    const int rows = std::min(g.height[k], fr->GetHeight(pl));
    const size_t row = (size_t)std::min(g.row[k], fr->GetRowSize(pl));
    bitblt(g_vsapi->getWritePtr(dst, g.vsplane[k]), g_vsapi->getStride(dst, g.vsplane[k]), fr->GetReadPtr(pl),
           fr->GetPitch(pl), row, rows);
  }
  VSMap *p = g_vsapi->getFramePropertiesRW(dst);
  const AVSMap *pm = ScriptEnvironment::props(fr);
  if (pm && pm->get())
    g_vsapi->copyMap(pm->get(), p);
  if (f.dur_num)
  {
    g_vsapi->mapSetInt(p, "_DurationNum", f.dur_num, maReplace);
    g_vsapi->mapSetInt(p, "_DurationDen", f.dur_den, maReplace);
  }
  return dst;
}

// An instance for one GetFrame call. MT_MULTI_INSTANCE: one call per instance at a time, so an idle one is taken or a
// new one created (as AviSynth+ does per prefetch thread); other modes share the first instance.
IClip *acquire(BridgeFilter &f)
{
  if (f.mtmode != MT_MULTI_INSTANCE)
    return f.all[0].operator->();
  {
    std::lock_guard<std::mutex> l(f.pool_m);
    if (!f.idle.empty())
    {
      IClip *c = f.idle.back();
      f.idle.pop_back();
      return c;
    }
  }
  PClip c;
  {
    // per filter, not global: the constructor may fetch frames synchronously, and a worker that computes them may need
    // a new instance of another filter
    std::lock_guard<std::mutex> l(f.create_m);
    AVSValue r = f.def->apply(f.args, f.def->user_data, &g_env);
    if (!r.IsClip())
      g_env.ThrowError("%s: a new instance returned no clip", f.name.c_str());
    c = r.AsClip();
  }
  const VideoInfo &vi = c->GetVideoInfo();
  if (vi.num_frames != f.avs_vi.num_frames || vi.pixel_type != f.avs_vi.pixel_type || vi.width != f.avs_vi.width ||
      vi.height != f.avs_vi.height)
    g_env.ThrowError("%s: a new instance has a different VideoInfo", f.name.c_str());
  std::lock_guard<std::mutex> l(f.pool_m);
  f.all.push_back(c);
  if (debug_log())
    fprintf(stderr, "mvtools_vsbridge: %s: instance %d\n", f.name.c_str(), (int)f.all.size());
  return c.operator->();
}

void release(BridgeFilter &f, IClip *c)
{
  if (f.mtmode != MT_MULTI_INSTANCE)
    return;
  std::lock_guard<std::mutex> l(f.pool_m);
  f.idle.push_back(c);
}

const VSFrame *VS_CC filter_get_frame(int n, int ar, void *instance, void **, VSFrameContext *ctx, VSCore *core,
                                      const VSAPI *vsapi)
{
  BridgeFilter &f = *static_cast<BridgeFilter *>(instance);
  if (ar == arInitial && !f.inputs.empty())
  {
    int offs[64];
    for (size_t k = 0; k < f.inputs.size(); ++k)
    {
      InputRule &r = *f.inputs[k];
      int cnt;
      {
        std::lock_guard<std::mutex> l(r.m);
        cnt = (int)r.offsets.size();
        std::copy(r.offsets.begin(), r.offsets.end(), offs);
      }
      const int base = f.base((int)k, n);
      for (int i = 0; i < cnt; ++i)
        vsapi->requestFrameFilter(std::min(std::max(0, base + offs[i]), r.frames - 1), r.node, ctx);
    }
    return nullptr;
  }
  if (ar != arInitial && ar != arAllFramesReady)
    return nullptr;

  CallCtx call{&f, ctx, n, core};
  CallCtx *prev = t_call;
  t_call = &call;
  const VSFrame *out = nullptr;
  IClip *inst = nullptr;
  try
  {
    inst = acquire(f);
    PVideoFrame pf = inst->GetFrame(n, &g_env);
    if (!pf)
      g_env.ThrowError("%s: no frame returned", f.name.c_str());
    out = to_vs_frame(f, pf, core);
  }
  catch (const AvisynthError &e)
  {
    vsapi->setFilterError(e.msg, ctx);
  }
  catch (const IScriptEnvironment::NotFound &)
  {
    vsapi->setFilterError((f.name + ": a function it needs is not available under VapourSynth").c_str(), ctx);
  }
  catch (const std::exception &e)
  {
    vsapi->setFilterError((f.name + ": " + e.what()).c_str(), ctx);
  }
  catch (...)
  {
    vsapi->setFilterError((f.name + ": unknown exception").c_str(), ctx);
  }
  if (inst)
    release(f, inst);
  t_call = prev;
  return out;
}

void VS_CC filter_free(void *instance, VSCore *, const VSAPI *)
{
  BridgeFilter *f = static_cast<BridgeFilter *>(instance);
  {
    std::lock_guard<std::mutex> l(g_registry_mutex);
    g_registry.erase(f->self);
  }
  if (debug_log())
    fprintf(stderr, "mvtools_vsbridge: %s freed (%d instances, %d frames not prefetched, last at output frame %d)\n",
            f->name.c_str(), (int)f->all.size(), f->misses.load(), f->last_miss.load());
  delete f; // may free upstream nodes, whose free callbacks run nested
}

// VS argument -> AviSynth value
bool vs_arg(const VSMap *in, const ArgDef &a, VSCore *core, std::vector<VSClip *> &clips, AVSValue &out,
            std::string &err)
{
  const int n = g_vsapi->mapNumElements(in, a.name.c_str());
  if (n <= 0)
    return true; // not given: undefined, the plugin's default applies
  std::vector<AVSValue> vals;
  for (int i = 0; i < n; ++i)
  {
    switch (a.type)
    {
    case 'c':
    {
      VSNode *node = g_vsapi->mapGetNode(in, a.name.c_str(), i, nullptr);
      VideoInfo vi;
      if (!registry_get(node, vi))
      {
        std::string why;
        if (!avs_video_info(*g_vsapi->getVideoInfo(node), vi, why))
        {
          g_vsapi->freeNode(node);
          err = "argument " + a.name + ": " + why;
          return false;
        }
      }
      VSClip *c = acquire_vsclip(node, vi);
      clips.push_back(c);
      vals.emplace_back(c);
      break;
    }
    case 'i': vals.emplace_back(g_vsapi->mapGetIntSaturated(in, a.name.c_str(), i, nullptr)); break;
    case 'f': vals.emplace_back(g_vsapi->mapGetFloat(in, a.name.c_str(), i, nullptr)); break;
    case 'b': vals.emplace_back(g_vsapi->mapGetInt(in, a.name.c_str(), i, nullptr) != 0); break;
    case 's':
      vals.emplace_back(g_env.SaveString(g_vsapi->mapGetData(in, a.name.c_str(), i, nullptr),
                                         g_vsapi->mapGetDataSize(in, a.name.c_str(), i, nullptr)));
      break;
    default: err = "argument " + a.name + ": unsupported type"; return false;
    }
  }
  out = a.array ? AVSValue(vals.data(), (int)vals.size()) : vals[0];
  (void)core;
  return true;
}

// the output side of a filter: AviSynth and VS VideoInfo, plane geometry, frame count
bool set_output_info(BridgeFilter &f, const VideoInfo &vi, VSCore *core, std::string &why)
{
  f.avs_vi = vi;
  f.geo = geometry(f.avs_vi);
  VSVideoInfo &v = f.vs_vi;
  memset(&v, 0, sizeof(v));
  if (!vs_format(f.avs_vi, core, v.format, v.width, v.height, why))
    return false;
  v.numFrames = f.avs_vi.num_frames;
  v.fpsNum = f.avs_vi.fps_numerator;
  v.fpsDen = f.avs_vi.fps_denominator;
  if (v.fpsNum && v.fpsDen)
  {
    int64_t x = v.fpsNum, y = v.fpsDen;
    while (y)
    {
      const int64_t t = x % y;
      x = y;
      y = t;
    }
    v.fpsNum /= x;
    v.fpsDen /= x;
  }
  f.out_frames = f.avs_vi.num_frames;
  return true;
}

void VS_CC create_filter(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
  g_vsapi = vsapi;
  const FuncDef &def = *static_cast<const FuncDef *>(user_data);
  VSCore *prev_core = t_core;
  t_core = core;
  struct Restore
  {
    VSCore *c;
    ~Restore() { t_core = c; }
  } restore{prev_core};

  std::vector<AVSValue> a(def.args.size());
  ClipUses uses; // until they belong to the filter
  std::vector<VSClip *> &clips = uses.v;
  std::string err;
  for (size_t i = 0; i < def.args.size(); ++i)
    if (!vs_arg(in, def.args[i], core, clips, a[i], err))
    {
      vsapi->mapSetError(out, (def.name + ": " + err).c_str());
      return;
    }
  AVSValue args(a.data(), (int)a.size());
  a.clear(); // args holds the references now

  AVSValue ret;
  try
  {
    ret = def.apply(args, def.user_data, &g_env);
  }
  catch (const AvisynthError &e)
  {
    vsapi->mapSetError(out, e.msg);
    return;
  }
  catch (const IScriptEnvironment::NotFound &)
  {
    vsapi->mapSetError(out, (def.name + ": a function it needs is not available under VapourSynth").c_str());
    return;
  }
  catch (const std::exception &e)
  {
    vsapi->mapSetError(out, (def.name + ": " + e.what()).c_str());
    return;
  }
  catch (...)
  {
    vsapi->mapSetError(out, (def.name + ": unknown exception").c_str());
    return;
  }

  if (!ret.IsClip())
  {
    if (ret.IsBool())
      vsapi->mapSetInt(out, "val", ret.AsBool(), maReplace);
    else if (ret.IsInt())
      vsapi->mapSetInt(out, "val", ret.AsLong(), maReplace);
    else if (ret.IsFloat())
      vsapi->mapSetFloat(out, "val", ret.AsFloat(), maReplace);
    else if (ret.IsString())
      vsapi->mapSetData(out, "val", ret.AsString(), -1, dtUtf8, maReplace);
    return;
  }
  PClip clip = ret.AsClip();
  IClip *raw = clip.operator->();

  // the function returned one of its inputs unchanged
  if (is_vsclip(raw))
  {
    vsapi->mapSetNode(out, "clip", static_cast<VSClip *>(raw)->node, maAppend);
    return;
  }

  std::unique_ptr<BridgeFilter> f(new BridgeFilter);
  f->name = def.name;
  f->def = &def;
  f->args = args;
  {
    std::string why;
    if (!set_output_info(*f, clip->GetVideoInfo(), core, why))
    {
      vsapi->mapSetError(out, (def.name + ": " + why).c_str());
      return;
    }
  }
  f->mtmode = clip->SetCacheHints(CACHE_GET_MTMODE, 0);
  // debugging: MVBRIDGE_SERIAL=MAnalyse,MFlowInter runs the named functions as one serialised instance
  if (const char *serial = std::getenv("MVBRIDGE_SERIAL"))
  {
    const std::string list = "," + lower(serial) + ",";
    if (list.find("," + lower(def.name) + ",") != std::string::npos || list == ",all,")
      f->mtmode = MT_SERIALIZED;
  }
  f->all.push_back(clip);
  if (f->mtmode == MT_MULTI_INSTANCE)
    f->idle.push_back(raw);

  std::vector<VSFilterDependency> deps;
  for (VSClip *c : clips)
  {
    std::unique_ptr<InputRule> r(new InputRule);
    r->node = c->node;
    r->frames = c->vi.num_frames;
    f->inputs.push_back(std::move(r));
    f->input_clips.push_back(c);
    deps.push_back({c->node, rpGeneral});
  }
  if (!clips.empty())
  {
    const VSVideoInfo *first = vsapi->getVideoInfo(clips[0]->node);
    if (f->vs_vi.fpsNum && (first->fpsNum != f->vs_vi.fpsNum || first->fpsDen != f->vs_vi.fpsDen))
    {
      f->dur_num = f->vs_vi.fpsDen;
      f->dur_den = f->vs_vi.fpsNum;
    }
  }
  clips.clear(); // released by ~BridgeFilter from here on

  const int mode = f->mtmode == MT_NICE_FILTER || f->mtmode == MT_MULTI_INSTANCE ? fmParallel : fmParallelRequests;
  BridgeFilter *fp = f.get();
  VSNode *node = vsapi->createVideoFilter2(def.name.c_str(), &fp->vs_vi, filter_get_frame, filter_free, mode,
                                           deps.data(), (int)deps.size(), fp, core);
  if (!node)
  {
    vsapi->mapSetError(out, (def.name + ": VapourSynth refused the clip (" + std::to_string(fp->vs_vi.width) + "x" +
                             std::to_string(fp->vs_vi.height) + ", " + std::to_string(fp->vs_vi.numFrames) + " frames)")
                                .c_str());
    return; // the free callback does not run for a refused filter: f is deleted here
  }
  f.release(); // owned by VS now (filter_free)
  fp->self = node;
  {
    std::lock_guard<std::mutex> l(g_registry_mutex);
    g_registry[node] = fp->avs_vi;
  }
  vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

// A VS node computing an AviSynth clip that the bridge did not get from a registered function (the inputs of a nearest
// fallback, e.g. cudafilters' CudaDownload in front of a VS node): one instance (no function to create more), no
// declared inputs (its fetches take the synchronous path; a fallback only).
VSNode *clip_node(const PClip &c, const std::string &name, VSCore *core, std::string &err)
{
  if (is_vsclip(c.operator->()))
    return g_vsapi->addNodeRef(static_cast<VSClip *>(c.operator->())->node);
  std::unique_ptr<BridgeFilter> f(new BridgeFilter);
  f->name = name;
  if (!set_output_info(*f, c->GetVideoInfo(), core, err))
    return nullptr;
  const int hint = c->SetCacheHints(CACHE_GET_MTMODE, 0);
  f->mtmode = hint == MT_NICE_FILTER ? MT_NICE_FILTER : MT_SERIALIZED;
  f->all.push_back(c);
  BridgeFilter *fp = f.get();
  VSNode *node = g_vsapi->createVideoFilter2(name.c_str(), &fp->vs_vi, filter_get_frame, filter_free,
                                             fp->mtmode == MT_NICE_FILTER ? fmParallel : fmParallelRequests, nullptr,
                                             0, fp, core);
  if (!node)
  {
    err = "VapourSynth refused the clip";
    return nullptr;
  }
  f.release(); // owned by VS now (filter_free)
  fp->self = node;
  {
    std::lock_guard<std::mutex> l(g_registry_mutex);
    g_registry[node] = fp->avs_vi;
  }
  return node;
}

// ---------------------------------------------------------------------------------------------------------------------
// AddFunction: AviSynth parameter string -> VS argument string

// names for the positional (unnamed) parameters, in order; the AviSynth documentation's names
std::vector<std::string> positional_names(const std::string &fname, size_t count)
{
  static const std::map<std::string, std::vector<std::string>> table = {
      {"mshow", {"super", "vectors"}},
      {"manalyse", {"super"}},
      {"mmask", {"clip", "vectors"}},
      {"mcompensate", {"clip", "super", "vectors"}},
      {"mscdetection", {"clip", "vectors"}},
      {"mdepan", {"clip", "vectors"}},
      {"mflow", {"clip", "super", "vectors"}},
      {"mflowinter", {"clip", "super", "mvbw", "mvfw"}},
      {"mflowfps", {"clip", "super", "mvbw", "mvfw"}},
      {"mflowblur", {"clip", "super", "mvbw", "mvfw"}},
      {"mblockfps", {"clip", "super", "mvbw", "mvfw"}},
      {"mdegrainn", {"clip", "super", "mvmulti", "tr"}},
      {"mrecalculate", {"super", "vectors"}},
      {"msuper", {"clip"}},
      {"mstorevect", {"clips"}},
      {"mrestorevect", {"clip"}},
      {"mscalevect", {"vectors"}},
  };
  const std::string l = lower(fname);
  std::vector<std::string> names;
  auto it = table.find(l);
  if (it != table.end())
    names = it->second;
  else if (l.size() == 9 && l.compare(0, 8, "mdegrain") == 0 && l[8] >= '1' && l[8] <= '9')
  {
    names = {"clip", "super"};
    for (int k = 1; k <= l[8] - '0'; ++k)
    {
      const std::string s = k == 1 ? "" : std::to_string(k);
      names.push_back("mvbw" + s);
      names.push_back("mvfw" + s);
    }
  }
  if (names.size() != count)
  {
    names.clear();
    for (size_t i = 0; i < count; ++i)
      names.push_back("arg" + std::to_string(i + 1));
  }
  return names;
}

bool parse_params(const std::string &fname, const char *params, std::vector<ArgDef> &args, std::string &err)
{
  struct Raw
  {
    std::string name;
    char type;
    bool array, optional;
  };
  std::vector<Raw> raw;
  size_t unnamed = 0;
  for (const char *p = params; *p;)
  {
    Raw r{"", 0, false, false};
    if (*p == '[')
    {
      const char *e = strchr(p, ']');
      if (!e)
      {
        err = "bad parameter string";
        return false;
      }
      r.name.assign(p + 1, e);
      r.optional = true;
      p = e + 1;
    }
    r.type = *p ? *p++ : 0;
    if (!strchr("cifbs", r.type))
    {
      err = std::string("parameter type '") + r.type + "' is not supported under VapourSynth";
      return false;
    }
    if (*p == '*' || *p == '+')
    {
      r.array = true;
      r.optional = r.optional || *p == '*';
      ++p;
    }
    if (r.name.empty())
      ++unnamed;
    raw.push_back(r);
  }
  const std::vector<std::string> names = positional_names(fname, unnamed);
  size_t k = 0;
  for (const Raw &r : raw)
    args.push_back({r.name.empty() ? names[k++] : r.name, r.type, r.array, !r.optional});
  return true;
}

std::string vs_arg_string(const std::vector<ArgDef> &args)
{
  std::string s;
  for (const ArgDef &a : args)
  {
    s += a.name + ":";
    switch (a.type)
    {
    case 'c': s += "vnode"; break;
    case 'i':
    case 'b': s += "int"; break;
    case 'f': s += "float"; break;
    case 's': s += "data"; break;
    }
    if (a.array)
      s += "[]";
    if (!a.required)
      s += ":opt";
    s += ";";
  }
  return s;
}

void __stdcall BridgeEnv::AddFunction(const char *name, const char *params, ApplyFunc apply, void *user_data)
{
  std::unique_ptr<FuncDef> d(new FuncDef);
  d->name = name;
  d->params = params;
  d->apply = apply;
  d->user_data = user_data;
  std::string err;
  if (!parse_params(name, params, d->args, err))
  {
    fprintf(stderr, "mvtools_vsbridge: %s not registered: %s\n", name, err.c_str());
    return;
  }
  FuncDef *dp = d.get();
  {
    std::lock_guard<std::mutex> l(g_funcs_mutex);
    auto &slot = g_funcs[lower(name)];
    if (slot)
      return; // first registration wins (Invoke target); VS functions are per plugin anyway
    slot = d.release();
  }
  if (reg_api && reg_plugin)
    reg_pending.push_back(dp);
}

// AviSynth argument names are case-insensitive, VS's are not, and plugins are not consistent (MRecalculate "thsad",
// MCompensate / MDegrain "thSAD"). A name that occurs in several casings among a plugin's functions gets the most
// frequent one (ties: the first seen), so the casing of the documentation works everywhere.
void unify_arg_case(std::vector<FuncDef *> &funcs)
{
  std::map<std::string, std::vector<std::pair<std::string, int>>> seen; // lower -> (casing, count) in first-seen order
  for (FuncDef *d : funcs)
    for (const ArgDef &a : d->args)
    {
      auto &v = seen[lower(a.name)];
      auto it = std::find_if(v.begin(), v.end(), [&](const std::pair<std::string, int> &e) { return e.first == a.name; });
      if (it == v.end())
        v.emplace_back(a.name, 1);
      else
        ++it->second;
    }
  for (FuncDef *d : funcs)
    for (ArgDef &a : d->args)
    {
      const auto &v = seen[lower(a.name)];
      const auto best = std::max_element(v.begin(), v.end(), [](const std::pair<std::string, int> &x,
                                                                const std::pair<std::string, int> &y) {
        return x.second < y.second;
      });
      a.name = best->first;
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// entry

typedef const char *(__stdcall *AvsPluginInit3)(IScriptEnvironment *env, const AVS_Linkage *const vectors);

// Called from a plugin's VapourSynthPluginInit2 after configPlugin. Returns 0 on success, else writes a message to err.
// abi: MVBRIDGE_ABI of the caller's VapourSynthEntry.cpp; init3: the plugin's AvisynthPluginInit3.
MVBRIDGE_EXPORT int mvbridge_register_functions(int abi, VSPlugin *plugin, const VSPLUGINAPI *vspapi,
                                                void *init3, char *err, int errlen)
{
  auto fail = [&](const std::string &m) {
    if (err && errlen > 0)
    {
      strncpy(err, m.c_str(), (size_t)errlen - 1);
      err[errlen - 1] = 0;
    }
    return 1;
  };
  if (abi != 1)
    return fail("mvtools_vsbridge: ABI " + std::to_string(abi) + " requested, this bridge has 1");
  if (!plugin || !vspapi || !init3)
    return fail("mvtools_vsbridge: bad arguments");
  std::lock_guard<std::mutex> l(g_register_mutex);
  install_crash_handler();
  g_env.reg_plugin = plugin;
  g_env.reg_api = vspapi;
  int r = 0;
  g_env.reg_pending.clear();
  try
  {
    reinterpret_cast<AvsPluginInit3>(init3)(&g_env, bridge_linkage());
    unify_arg_case(g_env.reg_pending);
    for (FuncDef *d : g_env.reg_pending)
      vspapi->registerFunction(d->name.c_str(), vs_arg_string(d->args).c_str(), "any", create_filter, d, plugin);
  }
  catch (const AvisynthError &e)
  {
    r = fail(std::string("AvisynthPluginInit3: ") + e.msg);
  }
  catch (...)
  {
    r = fail("AvisynthPluginInit3: unknown exception");
  }
  g_env.reg_plugin = nullptr;
  g_env.reg_api = nullptr;
  g_env.reg_pending.clear();
  return r;
}
