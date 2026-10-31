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
    off[k] = p[k] - p[0];
    if (off[k] < INT_MIN / 2 || off[k] > INT_MAX / 2)
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
  // contiguous copy, AviSynth layout (pitch aligned to 64)
  int dpitch[3];
  size_t size = 0, start[3];
  for (int k = 0; k < g.nplanes; ++k)
  {
    dpitch[k] = (g.row[k] + 63) & ~63;
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

class VSClip : public IClip
{
public:
  VSNode *node;
  VideoInfo vi;
  BridgeFilter *owner = nullptr; // the filter whose argument this clip is (learned prefetch), set after creation
  int input_index = -1;

  VSClip(VSNode *n, const VideoInfo &v) : node(n), vi(v)
  {
    std::lock_guard<std::mutex> l(g_clips_mutex);
    g_clips.insert(this);
  }
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
  ~BridgeFilter()
  {
    for (VSClip *c : input_clips)
      c->owner = nullptr;
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
    if (c && owner && c->filter == owner)
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

AVSValue __stdcall BridgeEnv::Invoke(const char *name, const AVSValue args, const char *const *arg_names)
{
  const std::string l = lower(name);
  if (l == "internalcache" || l == "cache")
    return args.IsArray() ? args[0] : args;
  const FuncDef *def = find_func(name);
  if (!def)
    throw NotFound();
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
      VSClip *c = new VSClip(node, vi);
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
  std::vector<VSClip *> clips;
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
  f->avs_vi = clip->GetVideoInfo();
  f->geo = geometry(f->avs_vi);
  {
    VSVideoInfo &v = f->vs_vi;
    memset(&v, 0, sizeof(v));
    std::string why;
    if (!vs_format(f->avs_vi, core, v.format, v.width, v.height, why))
    {
      vsapi->mapSetError(out, (def.name + ": " + why).c_str());
      return;
    }
    v.numFrames = f->avs_vi.num_frames;
    v.fpsNum = f->avs_vi.fps_numerator;
    v.fpsDen = f->avs_vi.fps_denominator;
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
  }
  f->out_frames = f->avs_vi.num_frames;
  f->mtmode = clip->SetCacheHints(CACHE_GET_MTMODE, 0);
  f->all.push_back(clip);
  if (f->mtmode == MT_MULTI_INSTANCE)
    f->idle.push_back(raw);

  std::vector<VSFilterDependency> deps;
  for (VSClip *c : clips)
  {
    std::unique_ptr<InputRule> r(new InputRule);
    r->node = c->node;
    r->frames = c->vi.num_frames;
    c->owner = f.get();
    c->input_index = (int)f->inputs.size();
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
