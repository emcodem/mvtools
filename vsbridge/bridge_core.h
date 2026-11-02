// mvtools_vsbridge: a small AviSynth+ host on top of VapourSynth (API 4), so that AviSynth+ plugins of this repo
// (mvtools2, cudafilters) run unchanged under VapourSynth, bit-identical with AviSynth+. Design: frames are requested
// as base(n) = n * frames_in / frames_out plus offsets learned per input (a miss fetches synchronously, which only
// costs speed); the AviSynth VideoInfo of every node sits in a registry keyed by VSNode*; vector clips travel as
// Gray8 carriers; VS functions keep the AviSynth function and parameter names.
//
// This library compiles avisynth.h in core mode (BUILDING_AVSCORE, set by CMake): the member functions of VideoInfo,
// VideoFrame, AVSValue, ... are defined here (avs_interface.cpp, from AviSynth+) and handed to the plugin as its
// AVS_Linkage table. The plugin itself is compiled in plugin mode and calls everything through that table, so the two
// never share a definition.

#ifndef MV_VSBRIDGE_CORE_H
#define MV_VSBRIDGE_CORE_H

#include <avisynth.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif
#include <VapourSynth4.h>

#include <cassert>

#ifndef _ASSERTE
#define _ASSERTE(x) assert(x)
#endif
#ifndef _ASSERT
#define _ASSERT(x) assert(x)
#endif

// Frame properties: a view on a VapourSynth property map. Input frames expose their read-only map; frames the bridge
// allocates expose the writable map of their VSFrame, or a map of their own (owned) when they are not backed by one.
struct AVSMap
{
  const VSMap *ro = nullptr;
  VSMap *rw = nullptr;
  bool owned = false; // rw was created by the bridge and is freed with this object

  const VSMap *get() const { return rw ? rw : ro; }
  ~AVSMap();
};

// AviSynth+ devices (Neo): the bridge has only the CPU
class Device
{
public:
  AvsDeviceType device_type = DEV_TYPE_CPU;
  int device_id = 0;
  int device_index = 0;
  const char *GetName() const { return "CPU"; }
};

// AviSynth+ function objects (Neo; AviSynth+ defines the class in its internal function.h). Only reference counting is
// needed here: no function objects exist under the bridge.
class IFunction
{
public:
  IFunction() : refcnt(0) {}
  virtual ~IFunction() {}
  void AddRef() { InterlockedIncrement(&refcnt); }
  void Release()
  {
    if (!InterlockedDecrement(&refcnt))
      delete this;
  }

private:
  volatile long refcnt;
};

// avs_interface.cpp -> bridge.cpp
void bridge_release_buffer(VideoFrameBuffer *vfb);    // last frame on the buffer released
bool bridge_buffer_readonly(const VideoFrameBuffer *vfb); // wraps a VS frame the bridge may not write to
const AVS_Linkage *bridge_linkage();

// avs_cpuid.cpp (AviSynth+)
int GetCPUFlags();

#endif
