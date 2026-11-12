// The built CudaConvolution.dll inside AviSynth+: automatic format conversion, 8-bit input, clip limits, frame properties,
// parameter errors and the self-test through the DLL. Needs AviSynth+ (AviSynth.dll) on the build machine.
#define NOMINMAX
#include <windows.h>

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include "avisynth/avisynth.h"

const AVS_Linkage* AVS_linkage = nullptr;

namespace {

// One script environment for the whole run, with CudaConvolution.dll and the test sources loaded.
class Avs {
 public:
  static Avs& get() {
    static Avs avs;
    return avs;
  }

  IScriptEnvironment* env() { return env_; }

  AVSValue eval(const std::string& script) {
    const AVSValue arg[1] = {AVSValue(script.c_str())};
    return env_->Invoke("Eval", AVSValue(arg, 1));
  }
  PClip clip(const std::string& script) { return eval(script).AsClip(); }

  // message of the AviSynth error the script throws, or "" if it does not
  std::string error_of(const std::string& script) {
    try {
      PClip c = clip(script);
      c->GetFrame(0, env_);
    } catch (const AvisynthError& e) {
      return e.msg;
    }
    return "";
  }

  bool gpu() const { return gpu_; }
  // engine argument for CudaConvolution: the GPU if this machine has one, else the CPU engine
  const char* engine() const { return gpu_ ? "" : ", cpu=true"; }

 private:
  Avs() {
    HMODULE avs = LoadLibraryW(L"AviSynth.dll");
    if (!avs) FAIL("AviSynth.dll not found: install AviSynth+ 3.7+ on the build machine (this test loads CudaConvolution.dll in it)");
    using Create = IScriptEnvironment*(__stdcall*)(int);
    auto create = reinterpret_cast<Create>(GetProcAddress(avs, "CreateScriptEnvironment"));
    if (!create) FAIL("AviSynth.dll has no CreateScriptEnvironment");
    env_ = create(AVISYNTH_INTERFACE_VERSION);
    if (!env_) FAIL("AviSynth.dll is too old: interface version " << AVISYNTH_INTERFACE_VERSION << " needed");
    AVS_linkage = env_->GetAVSLinkage();
    eval(std::string("LoadPlugin(\"") + CUDACONVOLUTION_DLL + "\")");
    // Test material: colour bars plus a texture (8 bit) or fine detail (16 bit), 4 different frames, all limited
    // range YUV 4:4:4. 512x288: factor-2 output 256x144.
    eval(R"(
      c = ColorBarsHD(width=512, height=288).Trim(0, -1)
      t = Expr(c, "x sx 7 * sy 13 * + 64 % 32 - +", "x sx 5 * sy 3 * + 32 % 16 - +", "x sx 3 * sy 11 * + 32 % 16 - +")
      global src8 = Interleave(t, t.FlipHorizontal(), t.FlipVertical(), t.Turn180())
      global src16 = Expr(src8.ConvertBits(16), "x sx 37 * sy 91 * + 256 % +")
    )");
    gpu_ = std::string(eval("CudaConvolution_Version()").AsString()).find(" - GPU: ") != std::string::npos;
  }

  IScriptEnvironment* env_ = nullptr;
  bool gpu_ = false;
};

std::string pixel_type(const VideoInfo& vi) {
  const int bits = vi.BitsPerComponent();
  const std::string depth = bits == 32 ? "S" : std::to_string(bits);
  if (vi.IsPlanarRGB()) return "RGBP" + depth;
  return std::string("YUV") + (vi.Is422() ? "422" : vi.Is444() ? "444" : "?") + "P" + depth;
}

int planes(const VideoInfo& vi, int i) {
  static constexpr int kYuv[3] = {PLANAR_Y, PLANAR_U, PLANAR_V}, kRgb[3] = {PLANAR_G, PLANAR_B, PLANAR_R};
  return vi.IsPlanarRGB() ? kRgb[i] : kYuv[i];
}

// all frames identical, byte for byte; returns a description of the first difference or ""
std::string compare(PClip a, PClip b) {
  IScriptEnvironment* env = Avs::get().env();
  const VideoInfo &va = a->GetVideoInfo(), &vb = b->GetVideoInfo();
  if (va.width != vb.width || va.height != vb.height || va.pixel_type != vb.pixel_type || va.num_frames != vb.num_frames)
    return "clip properties differ: " + pixel_type(va) + " " + std::to_string(va.width) + "x" + std::to_string(va.height) +
           " vs " + pixel_type(vb) + " " + std::to_string(vb.width) + "x" + std::to_string(vb.height);
  for (int n = 0; n < va.num_frames; ++n) {
    PVideoFrame fa = a->GetFrame(n, env), fb = b->GetFrame(n, env);
    for (int i = 0; i < 3; ++i) {
      const int pl = planes(va, i);
      for (int y = 0; y < fa->GetHeight(pl); ++y)
        if (std::memcmp(fa->GetReadPtr(pl) + y * fa->GetPitch(pl), fb->GetReadPtr(pl) + y * fb->GetPitch(pl),
                        fa->GetRowSize(pl)))
          return "frame " + std::to_string(n) + " plane " + std::to_string(i) + " row " + std::to_string(y) + " differs";
    }
  }
  return "";
}

// smallest and largest sample over all frames and planes
std::pair<int, int> value_range(PClip c) {
  IScriptEnvironment* env = Avs::get().env();
  const VideoInfo& vi = c->GetVideoInfo();
  int lo = 1 << 30, hi = -1;
  for (int n = 0; n < vi.num_frames; ++n) {
    PVideoFrame f = c->GetFrame(n, env);
    for (int i = 0; i < 3; ++i) {
      const int pl = planes(vi, i);
      for (int y = 0; y < f->GetHeight(pl); ++y) {
        const BYTE* row = f->GetReadPtr(pl) + y * f->GetPitch(pl);
        for (int x = 0; x < f->GetRowSize(pl) / vi.ComponentSize(); ++x) {
          const int v = vi.ComponentSize() == 1 ? row[x] : reinterpret_cast<const uint16_t*>(row)[x];
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
    }
  }
  return {lo, hi};
}

}  // namespace

TEST_CASE("automatic conversion: same output as the explicit conversion; convert=false names the format", "[avs]") {
  Avs& avs = Avs::get();
  struct Case {
    const char* format;  // as named in the error message
    const char* make;    // test input from src8 / src16
    const char* expect;  // explicit conversion CudaConvolution applies
    const char* output;  // CudaConvolution output pixel type
  };
  const Case cases[] = {
      {"YUV420P8", "src8.ConvertToYUV420()", "ConvertToYUV422()", "YUV422P8"},
      {"YUV420P10", "src16.ConvertBits(10).ConvertToYUV420()", "ConvertToYUV422()", "YUV422P10"},
      {"YUV411P8", "src8.ConvertToYV411()", "ConvertToYUV422()", "YUV422P8"},
      {"Y8", "src8.ConvertToY()", "ConvertToYUV422()", "YUV422P8"},
      {"YUY2", "src8.ConvertToYUY2()", "ConvertToYUV422()", "YUV422P8"},
      {"YUVA444P10", "src16.ConvertBits(10).AddAlphaPlane()", "RemoveAlphaPlane()", "YUV444P10"},
      {"YUV422P14", "src16.ConvertToYUV422().ConvertBits(14)", "ConvertBits(16)", "YUV422P16"},
      {"YUV422PS", "src16.ConvertToYUV422().ConvertBits(32)", "ConvertBits(16)", "YUV422P16"},
      {"RGB24", "src8.ConvertToRGB24()", "ConvertToPlanarRGB()", "RGBP8"},
      {"RGB32", "src8.ConvertToRGB32()", "ConvertToPlanarRGB()", "RGBP8"},
      {"RGB48", "src16.ConvertToRGB48()", "ConvertToPlanarRGB()", "RGBP16"},
      {"RGB64", "src16.ConvertToRGB64()", "ConvertToPlanarRGB()", "RGBP16"},
      {"RGBAP10", "src16.ConvertBits(10).ConvertToPlanarRGBA()", "RemoveAlphaPlane()", "RGBP10"},
      {"RGBP14", "src16.ConvertToPlanarRGB().ConvertBits(14)", "ConvertBits(16)", "RGBP16"},
  };
  for (const Case& c : cases) {
    INFO(c.format);
    const std::string conv = std::string(", width=256, height=144") + avs.engine() + ")";
    PClip a = avs.clip(std::string(c.make) + ".CudaConvolution(" + conv.substr(2));
    PClip b = avs.clip(std::string(c.make) + "." + c.expect + ".CudaConvolution(" + conv.substr(2));
    CHECK(pixel_type(a->GetVideoInfo()) == c.output);
    CHECK(compare(a, b) == "");
    const std::string err = avs.error_of(std::string(c.make) + ".CudaConvolution(convert=false" + conv);
    CHECK(err.find(std::string("input format ") + c.format + " is not supported") != std::string::npos);
  }
}

TEST_CASE("supported formats pass without conversion", "[avs]") {
  Avs& avs = Avs::get();
  for (const char* make : {"src8.ConvertToYUV422()", "src16.ConvertBits(10).ConvertToYUV422()", "src16.ConvertBits(12)",
                           "src16.ConvertToPlanarRGB()", "src8.ConvertToPlanarRGB()"}) {
    INFO(make);
    const std::string args = std::string("width=256, height=144") + avs.engine();
    CHECK(compare(avs.clip(std::string(make) + ".CudaConvolution(" + args + ")"),
                  avs.clip(std::string(make) + ".CudaConvolution(" + args + ", convert=false)")) == "");
  }
}

TEST_CASE("8-bit input = 10-bit route (x 4 in, rounded / 4 out), on every engine", "[avs]") {
  Avs& avs = Avs::get();
  for (const char* engine : {"", ", cpu=true"}) {
    if (!*engine && !avs.gpu()) continue;
    for (const char* layout : {"ConvertToYUV422()", "ConvertToYUV444()", "ConvertToPlanarRGB()"}) {
      INFO(layout << (*engine ? " CPU" : " GPU"));
      const std::string in = std::string("src8.") + layout, args = std::string("width=256, height=144") + engine;
      PClip native = avs.clip(in + ".CudaConvolution(" + args + ")");
      // exact shift both ways, independent of the clip's range properties
      PClip shifted = avs.clip(in + ".Expr(\"x 4 *\", format=\"" + (std::string(layout).find("RGB") != std::string::npos
                                   ? "RGBP10" : std::string(layout).find("444") != std::string::npos ? "YUV444P10" : "YUV422P10") +
                               "\").CudaConvolution(" + args + ").Expr(\"x 2 + 4 / floor 255 min\", format=\"" +
                               pixel_type(native->GetVideoInfo()) + "\")");
      CHECK(compare(native, shifted) == "");
      if (std::string(layout).find("RGB") == std::string::npos) {  // limited-range YUV: ConvertBits shifts too
        PClip cb = avs.clip(in + ".ConvertBits(10).CudaConvolution(" + args + ").ConvertBits(8, dither=-1)");
        CHECK(compare(native, cb) == "");
      }
    }
  }
}

TEST_CASE("clip_min / clip_max: limits in the input bit depth, same as clamping afterwards", "[avs]") {
  Avs& avs = Avs::get();
  const std::string args = std::string("width=256, height=144") + avs.engine();
  struct Case {
    const char* in;
    int lo, hi;
  };
  for (const Case& c : {Case{"src16.ConvertBits(10).ConvertToYUV422()", 64, 940}, Case{"src8.ConvertToYUV422()", 16, 235}}) {
    INFO(c.in);
    const std::string lim = ", clip_min=" + std::to_string(c.lo) + ", clip_max=" + std::to_string(c.hi);
    PClip clipped = avs.clip(std::string(c.in) + ".CudaConvolution(" + args + lim + ")");
    PClip after = avs.clip(std::string(c.in) + ".CudaConvolution(" + args + ").Expr(\"x " + std::to_string(c.lo) + " max " +
                           std::to_string(c.hi) + " min\")");
    CHECK(compare(clipped, after) == "");
    const auto [lo, hi] = value_range(clipped);
    CHECK(lo >= c.lo);
    CHECK(hi <= c.hi);
    // the unclipped output really exceeds the limits, so the test means something
    const auto [ulo, uhi] = value_range(avs.clip(std::string(c.in) + ".CudaConvolution(" + args + ")"));
    CHECK((ulo < c.lo && uhi > c.hi));
  }
  CHECK(avs.error_of("src8.ConvertToYUV422().CudaConvolution(clip_max=256" + std::string(avs.engine()) + ")")
            .find("clip_max must be within 0..255 for 8-bit input") != std::string::npos);
  CHECK(avs.error_of("src8.ConvertToYUV422().CudaConvolution(clip_min=200, clip_max=100" + std::string(avs.engine()) + ")")
            .find("clip_min (200) is above clip_max (100)") != std::string::npos);
}

TEST_CASE("output clip: size, interlaced TFF / BFF, frame rate and count halved", "[avs]") {
  Avs& avs = Avs::get();
  IScriptEnvironment* env = avs.env();
  for (const char* fo : {"tff", "bff"}) {
    INFO(fo);
    PClip c = avs.clip(std::string("src8.ConvertToYUV422().AssumeFPS(50).CudaConvolution(width=256, height=144, field_order=\"") +
                       fo + "\"" + avs.engine() + ")");
    const VideoInfo& vi = c->GetVideoInfo();
    CHECK(vi.width == 256);
    CHECK(vi.height == 144);
    CHECK(vi.num_frames == 2);
    CHECK(vi.fps_numerator * 1.0 / vi.fps_denominator == 25.0);
    CHECK(vi.IsTFF() == (std::string(fo) == "tff"));
    CHECK(vi.IsBFF() == (std::string(fo) == "bff"));
    PVideoFrame f = c->GetFrame(0, env);
    CHECK(env->propGetInt(env->getFramePropsRO(f), "_FieldBased", 0, nullptr) == (std::string(fo) == "tff" ? 2 : 1));
  }
}

TEST_CASE("parameter errors are reported with the reason", "[avs]") {
  Avs& avs = Avs::get();
  const std::string in = "src8.ConvertToYUV422().CudaConvolution(";
  CHECK(avs.error_of(in + "width=300)").find("width 512 -> 300 is not an integer ratio") != std::string::npos);
  CHECK(avs.error_of(in + "width=1024)").find("upscaling is not supported") != std::string::npos);
  CHECK(avs.error_of(in + "field_order=\"top\")").find("field_order must be") != std::string::npos);
  CHECK(avs.error_of(in + "preset=\"no_such_preset\")").find("neither a readable preset file") != std::string::npos);
  // GPU token frames are optional: the CPU engine cannot return them
  CHECK(avs.error_of(in + "cpu=true, token_out=true)").find("token_out=true needs the GPU engine") != std::string::npos);
}

// GPU token frames of the companion GPU plugins. Needs CUDACONVOLUTION_TOKEN_TEST_DIR = a folder with cudafilters.dll and
// mvtools_cuda.dll (skipped otherwise). CudaConvolution.dll stays in the build folder: it must use the library instance
// cudafilters loaded, not look for one next to itself.
TEST_CASE("GPU token frames: identical to ordinary frames, every frame checked", "[avs][tokens]") {
  const char* dir = std::getenv("CUDACONVOLUTION_TOKEN_TEST_DIR");
  if (!dir) SKIP("CUDACONVOLUTION_TOKEN_TEST_DIR not set");
  Avs& avs = Avs::get();
  if (!avs.gpu()) SKIP("no usable GPU");
  std::string d = dir;
  std::replace(d.begin(), d.end(), '\\', '/');
  avs.eval("LoadPlugin(\"" + d + "/cudafilters.dll\")");
  for (const char* in : {"src8.ConvertToYUV422()", "src8.ConvertToYUV422().ConvertBits(10)",
                         "src16.ConvertToYUV422()", "src8.ConvertToYUV420().ConvertBits(10)"}) {
    INFO(in);
    const std::string s = std::string(in);
    CHECK(compare(avs.clip(s + ".CudaUpload().CudaConvolution()"), avs.clip(s + ".CudaConvolution()")) == "");
    // a splice: frame 0 ordinary, later frames tokens (the conversion in front of CudaConvolution must still see pictures)
    CHECK(compare(avs.clip("(" + s + ".Trim(0, -1) ++ " + s + ".CudaUpload()).CudaConvolution()"),
                  avs.clip("(" + s + ".Trim(0, -1) ++ " + s + ").CudaConvolution()")) == "");
  }
  CHECK(compare(avs.clip("src8.ConvertToYUV422().ConvertBits(10).CudaUpload().CudaConvolution(token_out=true).CudaDownload()"),
                avs.clip("src8.ConvertToYUV422().ConvertBits(10).CudaConvolution()")) == "");
  CHECK(std::string(avs.eval("CudaConvolution_Version()").AsString()).find("GPU token frames: available") != std::string::npos);

  // A second, different copy of the library in the process is a second registry: CudaConvolution must refuse to read tokens
  // (token numbers of the two collide) instead of returning another frame. Loaded here directly from a temp folder.
  wchar_t tmp[MAX_PATH];
  GetTempPathW(MAX_PATH, tmp);
  const std::wstring copy_dir = std::wstring(tmp) + L"cudaconvolution_token_test_copy";
  CreateDirectoryW(copy_dir.c_str(), nullptr);
  const std::wstring src_lib = std::wstring(d.begin(), d.end()) + L"/mvtools_cuda.dll", copy = copy_dir + L"\\mvtools_cuda.dll";
  // a real copy: Windows maps a hard link of an already loaded file to the loaded module
  if (GetFileAttributesW(copy.c_str()) == INVALID_FILE_ATTRIBUTES) REQUIRE(CopyFileW(src_lib.c_str(), copy.c_str(), FALSE));
  REQUIRE(LoadLibraryW(copy.c_str()) != nullptr);
  const std::string err = avs.error_of("src8.ConvertToYUV422().ConvertBits(10).CudaUpload().CudaConvolution()");
  INFO(err);
  CHECK(err.find("different copies of mvtools_cuda.dll") != std::string::npos);
}

TEST_CASE("CudaConvolution_SelfTest() through the DLL: PASS on every engine of this machine", "[avs]") {
  Avs& avs = Avs::get();
  const std::string report = avs.eval("CudaConvolution_SelfTest()").AsString();
  INFO(report);
  CHECK(report.find("RESULT: PASS") != std::string::npos);
  if (!avs.gpu()) WARN("no usable GPU on this machine: GPU engine not tested through the DLL");
}
