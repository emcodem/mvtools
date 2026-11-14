// MovingPattern: synthetic moving test clips for AviSynth+ (any size, frame rate, scan and planar YUV / Y format).
//
// Every output sample is evaluated at its own position and instant: in an interlaced clip each field is computed at
// its own time (two instants per frame), so the clip has real interlacing. Sizes and speeds are given in units of the
// picture height, so a pattern looks the same at every size. The content repeats after `period` frames; `frames` sets
// the length of the clip.
//
// Patterns (pattern=):
//   "motion"  (default) the processing stress scene: a 1/f texture panning at sub-pixel speed, an inset zooming into
//             a second texture, a bar with 2-line horizontal stripes (interlace twitter) and a disc with a block
//             texture moving across it (the disc occludes the bar), luma grain, and a hard cut to a second scene
//             with other textures and directions at half the period
//   "pan"     only the panning background of "motion" (with its cut)
//   "still"   the background texture of the first scene, not moving, no grain
//
// Storage (storage=): "memory" generates the whole period when the clip is created (all threads) and returns the
// stored frames; "disk" generates it into a file in the OS temp folder (removed when the clip is destroyed, and by
// the OS if the process ends without that) and reads frames back; "none" renders each frame when it is requested.
#include <avisynth.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <strings.h>
#include <stdlib.h>
#include <unistd.h>
#endif

namespace
{
  // ------------------------------------------------------------------------------------------------ helpers
  inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c)
  {
    uint32_t h = a * 0x8da6b343u ^ b * 0xd8163841u ^ c * 0xcb1ab31fu;
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    h *= 0x297a2d39u;
    h ^= h >> 15;
    return h;
  }
  inline float unit(uint32_t h) { return (float)(h >> 8) * (1.0f / 16777216.0f); }   // [0, 1)
  inline float smooth(float t) { return t * t * (3.0f - 2.0f * t); }

  // seamless texture with a 1/f amplitude spectrum: n x n samples, values around 0 (about -0.5 .. 0.5), periodic in
  // both directions
  struct Tile
  {
    int n = 0;
    std::vector<float> v;
    void make(int size, uint32_t seed)
    {
      n = size;
      v.assign((size_t)n * n, 0.0f);
      // octaves of value noise: lattice of c x c cells over the tile, c = 4, 8, ... (up to about 2 px cells).
      // A 1/f amplitude spectrum (natural pictures) has about the same energy in every octave; the octave
      // amplitude falls slightly (c^-0.25) so the finest octaves do not turn into plain noise
      std::vector<int> cells;
      for (int c = 4; c <= n / 2; c *= 2)
        cells.push_back(c);
      float norm = 0.0f;
      for (int c : cells)
        norm += std::pow((float)c, -0.25f);
      for (size_t k = 0; k < cells.size(); k++)
      {
        const int c = cells[k];
        const float amp = std::pow((float)c, -0.25f) / norm;
        std::vector<float> lat((size_t)c * c);
        for (int j = 0; j < c; j++)
          for (int i = 0; i < c; i++)
            lat[(size_t)j * c + i] = unit(hash3((uint32_t)i, (uint32_t)j, seed * 64u + (uint32_t)k)) - 0.5f;
        const float s = (float)c / (float)n;
        for (int y = 0; y < n; y++)
        {
          const float fy = y * s;
          const int y0 = (int)fy % c, y1 = (y0 + 1) % c;
          const float ty = smooth(fy - std::floor(fy));
          float *row = &v[(size_t)y * n];
          for (int x = 0; x < n; x++)
          {
            const float fx = x * s;
            const int x0 = (int)fx % c, x1 = (x0 + 1) % c;
            const float tx = smooth(fx - std::floor(fx));
            const float a = lat[(size_t)y0 * c + x0], b = lat[(size_t)y0 * c + x1];
            const float d = lat[(size_t)y1 * c + x0], e = lat[(size_t)y1 * c + x1];
            row[x] += amp * ((a + (b - a) * tx) + ((d + (e - d) * tx) - (a + (b - a) * tx)) * ty);
          }
        }
      }
      // stretch to about +-0.5 (the octave sum is narrower than its range)
      double sq = 0.0;
      for (float f : v)
        sq += (double)f * f;
      const float rms = (float)std::sqrt(sq / (double)v.size());
      const float g = rms > 0.0f ? 0.2f / rms : 1.0f;
      for (float &f : v)
        f = std::max(-0.5f, std::min(0.5f, f * g));
    }
    // bilinear sample at (u, v) in tile units (1 = one tile), wrapping
    float at(float u, float w) const
    {
      float fx = u * n, fy = w * n;
      fx -= std::floor(fx / n) * n;
      fy -= std::floor(fy / n) * n;
      const int x0 = std::min((int)fx, n - 1), y0 = std::min((int)fy, n - 1);
      const int x1 = x0 + 1 == n ? 0 : x0 + 1, y1 = y0 + 1 == n ? 0 : y0 + 1;
      const float tx = fx - x0, ty = fy - y0;
      const float a = v[(size_t)y0 * n + x0], b = v[(size_t)y0 * n + x1];
      const float c = v[(size_t)y1 * n + x0], d = v[(size_t)y1 * n + x1];
      const float top = a + (b - a) * tx, bot = c + (d - c) * tx;
      return top + (bot - top) * ty;
    }
  };

  enum class Pattern { Motion, Pan, Still };
  enum class Storage { Memory, Disk, None };

  // one scene of the motion pattern; positions in picture heights, speeds in picture heights per second
  struct Scene
  {
    float pan_x, pan_y;                       // background velocity
    float zoom0, zoom1;                       // inset zoom over the scene
    float bar_x, bar_y, bar_vx, bar_vy;       // bar top-left and velocity
    float disc_x, disc_y, disc_vx, disc_vy;   // disc centre and velocity
    float bar_u, bar_v;                       // bar chroma (-0.5 .. 0.5)
  };

  // ------------------------------------------------------------------------------------------------ the filter
  class MovingPattern : public IClip
  {
    VideoInfo vi_;
    Pattern pattern_;
    Storage storage_;
    int period_;
    int scan_;            // 0 progressive, 1 bff, 2 tff (as _FieldBased)
    int bits_, sub_w_, sub_h_, planes_;
    bool full_;
    int matrix_, primaries_, transfer_, chroma_loc_, sar_n_, sar_d_;
    uint32_t seed_;
    int threads_;
    Tile bg_[2], inset_[2], cu_[2], cv_[2];
    Scene scene_[2];
    std::vector<PVideoFrame> frames_;        // storage memory
    size_t frame_bytes_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
#else
    int file_ = -1;
#endif

    // value of plane p (0 luma, 1 U, 2 V) at frame position (x, y) in luma pixels, instant time t (seconds from the
    // period start); luma about 0 .. 1, chroma about -0.5 .. 0.5
    float sample(int p, float x, float y, float t) const
    {
      const float H = (float)vi_.height, W = (float)vi_.width;
      const float half = (float)period_ / (2.0f * fps()) ;
      const int s = (pattern_ == Pattern::Still || t < half) ? 0 : 1;
      const float ts = pattern_ == Pattern::Still ? 0.0f : (s == 0 ? t : t - half);
      const Scene &sc = scene_[s];
      const float u = x / H, w = y / H;      // position in picture heights
      // background
      const float bu = u + sc.pan_x * ts, bw = w + sc.pan_y * ts;
      float val = p == 0 ? 0.5f + 0.75f * bg_[s].at(bu, bw) : 0.3f * (p == 1 ? cu_[s] : cv_[s]).at(bu, bw);
      if (pattern_ != Pattern::Motion)
        return val;
      // zoom inset: 0.44 x 0.33 H at (W / H - 0.6, 0.1), zooming about its centre
      const float ix0 = W / H - 0.6f, iy0 = 0.1f, iw = 0.44f, ih = 0.33f;
      if (u >= ix0 && u < ix0 + iw && w >= iy0 && w < iy0 + ih)
      {
        const float z = sc.zoom0 + (sc.zoom1 - sc.zoom0) * (ts / half);
        const float cx = ix0 + iw * 0.5f, cy = iy0 + ih * 0.5f;
        const float zu = cx + (u - cx) / z, zw = cy + (w - cy) / z;
        const bool edge = u < ix0 + 0.004f || u >= ix0 + iw - 0.004f || w < iy0 + 0.004f || w >= iy0 + ih - 0.004f;
        val = p == 0 ? (edge ? 0.08f : 0.5f + 0.8f * inset_[s].at(zu, zw)) : (edge ? 0.0f : 0.25f * cv_[1 - s].at(zw, zu));
      }
      // bar: 0.44 x 0.185 H, 2-line horizontal stripes (in frame rows) with columns of H / 34
      const float bx = sc.bar_x + sc.bar_vx * ts, by = sc.bar_y + sc.bar_vy * ts;
      const bool in_bar = u >= bx && u < bx + 0.44f && w >= by && w < by + 0.185f;
      // disc: radius 0.148 H, blocks of H / 135 moving with it, in front of the bar
      const float dx = (u - (sc.disc_x + sc.disc_vx * ts)) * H, dy = (w - (sc.disc_y + sc.disc_vy * ts)) * H;
      const float r = 0.148f * H;
      if (dx * dx + dy * dy < r * r)
      {
        const float blk = H / 135.0f;
        const int bi = (int)std::floor(dx / blk), bj = (int)std::floor(dy / blk);
        const uint32_t h = hash3((uint32_t)(bi + 4096), (uint32_t)(bj + 4096), seed_ * 7u + (uint32_t)s);
        return p == 0 ? 0.15f + 0.7f * unit(h) : (p == 1 ? 0.3f * (unit(h >> 3) - 0.5f) : -0.1f);
      }
      if (in_bar)
      {
        if (p != 0)
          return p == 1 ? sc.bar_u : sc.bar_v;
        const int rows = (int)std::floor(y - by * H);
        const int cols = (int)std::floor((u - bx) * 34.0f);
        return ((rows & 3) < 2 ? 0.86f : 0.1f) + ((cols & 1) ? 0.0f : 0.08f);
      }
      return val;
    }

    float fps() const { return (float)vi_.fps_numerator / (float)vi_.fps_denominator; }

    // render frame n of the period into the planes of a frame
    void render(int n, uint8_t *const dst[3], const int pitch[3]) const
    {
      const int fields = scan_ ? 2 : 1;
      const float rate = fps() * fields;   // instants per second
      const int maxv = (1 << bits_) - 1;
      for (int p = 0; p < planes_; p++)
      {
        const int pw = p ? vi_.width >> sub_w_ : vi_.width, ph = p ? vi_.height >> sub_h_ : vi_.height;
        const int sw = p ? 1 << sub_w_ : 1, sh = p ? 1 << sub_h_ : 1;
        // chroma siting: centre (1, 3, 5) horizontally half way between luma samples, else on the even ones
        const float xoff = p && sub_w_ && (chroma_loc_ == 1 || chroma_loc_ == 3 || chroma_loc_ == 5) ? 0.5f : 0.0f;
        for (int row = 0; row < ph; row++)
        {
          // field and frame row of the sample (4:2:0 interlaced: chroma rows alternate between the fields)
          int field = 0;
          float y;
          if (!scan_)
            y = sh == 1 ? (float)row : row * 2.0f + 0.5f;
          else if (sh == 1)
          {
            field = row & 1;
            y = (float)row;
          }
          else
          {
            field = row & 1;
            y = (float)((row >> 1) * 4 + field) + 1.0f;   // between the field's luma rows 2k and 2k + 1
          }
          // tff: the top field (even rows) comes first; bff: the bottom field
          const int order = scan_ == 2 ? field : 1 - field;
          const float t = ((float)n * fields + (scan_ ? (float)order : 0.0f)) / rate;
          const uint32_t inst = (uint32_t)(n * fields + (scan_ ? order : 0));
          uint8_t *d8 = dst[p] + (size_t)row * pitch[p];
          uint16_t *d16 = reinterpret_cast<uint16_t *>(d8);
          for (int col = 0; col < pw; col++)
          {
            const float x = (float)(col * sw) + xoff;
            float v = sample(p, x, y, t);
            float code;
            if (p == 0)
            {
              const float lo = full_ ? 0.0f : (float)(16 << (bits_ - 8)), hi = full_ ? (float)maxv : (float)(235 << (bits_ - 8));
              code = lo + std::max(0.0f, std::min(1.0f, v)) * (hi - lo);
              if (pattern_ != Pattern::Still)   // grain: +-1 % of the range, per sample and instant
                code += (unit(hash3((uint32_t)col, (uint32_t)row, inst * 31u + seed_)) - 0.5f) * 0.02f * (hi - lo);
            }
            else
            {
              const float mid = (float)(1 << (bits_ - 1)), span = full_ ? (float)maxv : (float)(224 << (bits_ - 8));
              code = mid + std::max(-0.5f, std::min(0.5f, v)) * span;
            }
            const int c = std::max(0, std::min(maxv, (int)std::lround(code)));
            if (bits_ == 8)
              d8[col] = (uint8_t)c;
            else
              d16[col] = (uint16_t)c;
          }
        }
      }
    }

    void set_props(PVideoFrame &f, IScriptEnvironment *env) const
    {
      AVSMap *m = env->getFramePropsRW(f);
      env->propSetInt(m, "_FieldBased", scan_, 0);
      env->propSetInt(m, "_Matrix", matrix_, 0);
      env->propSetInt(m, "_Primaries", primaries_, 0);
      env->propSetInt(m, "_Transfer", transfer_, 0);
      env->propSetInt(m, "_ColorRange", full_ ? 0 : 1, 0);
      if (planes_ == 3 && (sub_w_ || sub_h_))
        env->propSetInt(m, "_ChromaLocation", chroma_loc_, 0);
      env->propSetInt(m, "_SARNum", sar_n_, 0);
      env->propSetInt(m, "_SARDen", sar_d_, 0);
      env->propSetInt(m, "_DurationNum", vi_.fps_denominator, 0);
      env->propSetInt(m, "_DurationDen", vi_.fps_numerator, 0);
    }

    PVideoFrame new_frame(IScriptEnvironment *env) const
    {
      PVideoFrame f = env->NewVideoFrame(vi_);
      set_props(f, env);
      return f;
    }

    void planes_of(PVideoFrame &f, uint8_t *dst[3], int pitch[3]) const
    {
      static const int ids[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
      for (int p = 0; p < 3; p++)
      {
        dst[p] = p < planes_ ? f->GetWritePtr(ids[p]) : nullptr;
        pitch[p] = p < planes_ ? f->GetPitch(ids[p]) : 0;
      }
    }

    // byte layout of one frame in the disk file: the planes one after the other, rows packed
    size_t plane_row_bytes(int p) const { return (size_t)(p ? vi_.width >> sub_w_ : vi_.width) * (bits_ == 8 ? 1 : 2); }
    int plane_rows(int p) const { return p ? vi_.height >> sub_h_ : vi_.height; }

    template <class F> void parallel(int count, F fn) const
    {
      const int nt = std::max(1, std::min(count, threads_));
      std::atomic<int> next(0);
      std::vector<std::thread> pool;
      for (int i = 0; i < nt; i++)
        pool.emplace_back([&]() {
          for (int k = next++; k < count; k = next++)
            fn(k);
        });
      for (auto &t : pool)
        t.join();
    }

    void open_temp(IScriptEnvironment *env)
    {
#ifdef _WIN32
      wchar_t dir[MAX_PATH + 1], name[MAX_PATH + 1];
      if (!GetTempPathW(MAX_PATH, dir) || !GetTempFileNameW(dir, L"mvp", 0, name))
        env->ThrowError("MovingPattern: no temp file (storage=\"disk\")");
      file_ = CreateFileW(name, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                          FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
      if (file_ == INVALID_HANDLE_VALUE)
        env->ThrowError("MovingPattern: cannot create the temp file (storage=\"disk\")");
#else
      const char *tmp = getenv("TMPDIR");
      std::string path = std::string(tmp && *tmp ? tmp : "/tmp") + "/movingpattern_XXXXXX";
      std::vector<char> buf(path.begin(), path.end());
      buf.push_back(0);
      file_ = mkstemp(buf.data());
      if (file_ < 0)
        env->ThrowError("MovingPattern: cannot create the temp file (storage=\"disk\")");
      unlink(buf.data());   // removed when the descriptor is closed
#endif
    }

    bool file_io(bool write, void *data, size_t bytes, uint64_t offset) const
    {
#ifdef _WIN32
      uint8_t *ptr = static_cast<uint8_t *>(data);
      while (bytes)
      {
        const DWORD chunk = (DWORD)std::min<size_t>(bytes, 1u << 30);
        OVERLAPPED ov = {};
        ov.Offset = (DWORD)(offset & 0xffffffffu);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        DWORD done = 0;
        const BOOL ok = write ? WriteFile(file_, ptr, chunk, &done, &ov) : ReadFile(file_, ptr, chunk, &done, &ov);
        if (!ok || done != chunk)
          return false;
        ptr += chunk;
        bytes -= chunk;
        offset += chunk;
      }
      return true;
#else
      uint8_t *ptr = static_cast<uint8_t *>(data);
      while (bytes)
      {
        const ssize_t done = write ? pwrite(file_, ptr, bytes, (off_t)offset) : pread(file_, ptr, bytes, (off_t)offset);
        if (done <= 0)
          return false;
        ptr += done;
        bytes -= (size_t)done;
        offset += (uint64_t)done;
      }
      return true;
#endif
    }

  public:
    MovingPattern(const AVSValue &args, IScriptEnvironment *env)
    {
      const int width = args[0].AsInt(1920), height = args[1].AsInt(1080);
      const std::string pt = args[2].AsString("YUV422P10");
      const int fn = args[3].AsInt(25), fd = args[4].AsInt(1);
      const std::string scan = args[5].AsString("tff");
      const std::string pat = args[8].AsString("motion");
      const std::string st = args[9].AsString("memory");
      if (fn <= 0 || fd <= 0)
        env->ThrowError("MovingPattern: fps_num and fps_den must be positive");
      if (pat == "motion") pattern_ = Pattern::Motion;
      else if (pat == "pan") pattern_ = Pattern::Pan;
      else if (pat == "still") pattern_ = Pattern::Still;
      else env->ThrowError("MovingPattern: pattern must be \"motion\", \"pan\" or \"still\"");
      if (st == "memory") storage_ = Storage::Memory;
      else if (st == "disk") storage_ = Storage::Disk;
      else if (st == "none") storage_ = Storage::None;
      else env->ThrowError("MovingPattern: storage must be \"memory\", \"disk\" or \"none\"");
      if (scan == "p") scan_ = 0;
      else if (scan == "bff") scan_ = 1;
      else if (scan == "tff") scan_ = 2;
      else env->ThrowError("MovingPattern: scan must be \"tff\", \"bff\" or \"p\"");

      vi_ = {};
      vi_.width = width;
      vi_.height = height;
      vi_.pixel_type = 0;
      static const struct { const char *name; int type; } types[] = {
        { "YUV420P8", VideoInfo::CS_YV12 }, { "YV12", VideoInfo::CS_YV12 }, { "YUV422P8", VideoInfo::CS_YV16 },
        { "YV16", VideoInfo::CS_YV16 }, { "YUV444P8", VideoInfo::CS_YV24 }, { "YV24", VideoInfo::CS_YV24 },
        { "YUV420P10", VideoInfo::CS_YUV420P10 }, { "YUV422P10", VideoInfo::CS_YUV422P10 },
        { "YUV444P10", VideoInfo::CS_YUV444P10 }, { "YUV420P12", VideoInfo::CS_YUV420P12 },
        { "YUV422P12", VideoInfo::CS_YUV422P12 }, { "YUV444P12", VideoInfo::CS_YUV444P12 },
        { "YUV420P16", VideoInfo::CS_YUV420P16 }, { "YUV422P16", VideoInfo::CS_YUV422P16 },
        { "YUV444P16", VideoInfo::CS_YUV444P16 }, { "Y8", VideoInfo::CS_Y8 }, { "Y10", VideoInfo::CS_Y10 },
        { "Y12", VideoInfo::CS_Y12 }, { "Y16", VideoInfo::CS_Y16 },
      };
      for (const auto &t : types)
        if (_stricmp(pt.c_str(), t.name) == 0)
          vi_.pixel_type = t.type;
      if (!vi_.pixel_type)
        env->ThrowError("MovingPattern: pixel_type must be planar YUV 4:2:0 / 4:2:2 / 4:4:4 or Y, 8 to 16 bit "
                        "integer (e.g. \"YUV422P10\")");
      bits_ = vi_.BitsPerComponent();
      planes_ = vi_.IsY() ? 1 : 3;
      sub_w_ = planes_ == 3 ? vi_.GetPlaneWidthSubsampling(PLANAR_U) : 0;
      sub_h_ = planes_ == 3 ? vi_.GetPlaneHeightSubsampling(PLANAR_U) : 0;
      if (width < 16 || height < 16 || width % (1 << sub_w_) || height % (scan_ ? 2 << sub_h_ : 1 << sub_h_))
        env->ThrowError("MovingPattern: width / height too small or not a multiple of the chroma subsampling "
                        "(interlaced: of twice the vertical subsampling)");
      vi_.SetFPS(fn, fd);
      if (scan_)
        vi_.SetFieldBased(false), vi_.image_type = scan_ == 2 ? VideoInfo::IT_TFF : VideoInfo::IT_BFF;
      period_ = args[7].AsInt(std::max(2, (int)std::lround(4.0 * fn / fd)));
      if (period_ < 2)
        env->ThrowError("MovingPattern: period must be at least 2 frames");
      vi_.num_frames = args[6].AsInt(period_);
      if (vi_.num_frames < 1)
        env->ThrowError("MovingPattern: frames must be at least 1");
      matrix_ = args[10].AsInt(1);
      primaries_ = args[11].AsInt(1);
      transfer_ = args[12].AsInt(1);
      full_ = args[13].AsInt(1) == 0;
      chroma_loc_ = args[14].AsInt(sub_h_ ? 0 : 0);
      sar_n_ = args[15].AsInt(1);
      sar_d_ = args[16].AsInt(1);
      seed_ = (uint32_t)args[17].AsInt(1);
      threads_ = args[18].AsInt(0);
      if (threads_ <= 0)
        threads_ = std::max(1u, std::thread::hardware_concurrency());
      if (sar_n_ <= 0 || sar_d_ <= 0)
        env->ThrowError("MovingPattern: sar_num and sar_den must be positive");

      // scenes (positions / speeds in picture heights, per second)
      const float a = (float)width / (float)height;
      scene_[0] = { 0.9f, 0.3f, 1.0f, 1.8f, -0.3f, 0.72f, 0.42f, -0.05f, a - 0.1f, 0.25f, -0.55f, 0.12f, 0.35f, -0.25f };
      scene_[1] = { -0.55f, 0.45f, 1.7f, 1.0f, a - 0.2f, 0.3f, -0.6f, 0.03f, 0.2f, 0.75f, 0.32f, -0.14f, -0.3f, 0.3f };
      // textures: one tile per picture height, at most 2048 samples (coarser for very large pictures)
      const int tn = std::max(64, std::min(2048, height));
      Tile *tiles[8] = { &bg_[0], &bg_[1], &inset_[0], &inset_[1], &cu_[0], &cu_[1], &cv_[0], &cv_[1] };
      parallel(8, [&](int k) { tiles[k]->make(tn, seed_ * 16u + (uint32_t)k); });

      if (storage_ == Storage::Memory)
      {
        frames_.resize(period_);
        for (int k = 0; k < period_; k++)
          frames_[k] = new_frame(env);
        parallel(period_, [&](int k) {
          uint8_t *dst[3];
          int pitch[3];
          planes_of(frames_[k], dst, pitch);
          render(k, dst, pitch);
        });
      }
      else if (storage_ == Storage::Disk)
      {
        open_temp(env);
        for (int p = 0; p < planes_; p++)
          frame_bytes_ += plane_row_bytes(p) * plane_rows(p);
        std::atomic<bool> fail(false);
        parallel(period_, [&](int k) {
          std::vector<uint8_t> buf(frame_bytes_);
          uint8_t *dst[3] = {};
          int pitch[3] = {};
          size_t off = 0;
          for (int p = 0; p < planes_; p++)
          {
            dst[p] = buf.data() + off;
            pitch[p] = (int)plane_row_bytes(p);
            off += plane_row_bytes(p) * plane_rows(p);
          }
          render(k, dst, pitch);
          if (!file_io(true, buf.data(), frame_bytes_, (uint64_t)k * frame_bytes_))
            fail = true;
        });
        if (fail)
          env->ThrowError("MovingPattern: writing the temp file failed (disk full?)");
      }
    }

    ~MovingPattern()
    {
#ifdef _WIN32
      if (file_ != INVALID_HANDLE_VALUE)
        CloseHandle(file_);
#else
      if (file_ >= 0)
        close(file_);
#endif
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      const int k = std::max(0, n) % period_;
      if (storage_ == Storage::Memory)
        return frames_[k];
      PVideoFrame f = new_frame(env);
      uint8_t *dst[3];
      int pitch[3];
      planes_of(f, dst, pitch);
      if (storage_ == Storage::None)
      {
        render(k, dst, pitch);
        return f;
      }
      std::vector<uint8_t> buf(frame_bytes_);
      if (!file_io(false, buf.data(), frame_bytes_, (uint64_t)k * frame_bytes_))
        env->ThrowError("MovingPattern: reading the temp file failed");
      size_t off = 0;
      for (int p = 0; p < planes_; p++)
      {
        const size_t rb = plane_row_bytes(p);
        for (int r = 0; r < plane_rows(p); r++, off += rb)
          std::memcpy(dst[p] + (size_t)r * pitch[p], buf.data() + off, rb);
      }
      return f;
    }

    bool __stdcall GetParity(int n) override { return scan_ == 2; }
    void __stdcall GetAudio(void *, int64_t, int64_t, IScriptEnvironment *) override {}
    const VideoInfo &__stdcall GetVideoInfo() override { return vi_; }
    int __stdcall SetCacheHints(int hints, int) override { return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0; }
  };

  AVSValue __cdecl Create(AVSValue args, void *, IScriptEnvironment *env) { return new MovingPattern(args, env); }
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
  AVS_linkage = vectors;
  try { env->CheckVersion(8); }
  catch (const AvisynthError &) { env->ThrowError("MovingPattern: needs AviSynth+ 3.6 or newer (interface V8)"); }
  env->AddFunction("MovingPattern",
    "[width]i[height]i[pixel_type]s[fps_num]i[fps_den]i[scan]s[frames]i[period]i[pattern]s[storage]s"
    "[matrix]i[primaries]i[transfer]i[range]i[chroma_location]i[sar_num]i[sar_den]i[seed]i[threads]i",
    Create, 0);
  return "MovingPattern: synthetic moving test clips";
}
