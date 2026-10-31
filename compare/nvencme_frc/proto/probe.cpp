// NVENC ME-only probe: synthetic shifts (search range, vector accuracy), MV layout statistics, speed per pair with
// 1..4 threads, session cap.   probe.exe [w h]
#include "nvme_core.h"

#include <atomic>
#include <chrono>
#include <map>
#include <thread>

using namespace nvme;

// smooth random texture (sum of box-filtered noise), deterministic
static std::vector<uint8_t> texture(int w, int h, unsigned seed)
{
  std::vector<int> n((size_t)w * h);
  for (auto &v : n)
  {
    seed = seed * 1664525u + 1013904223u;
    v = (seed >> 24) & 255;
  }
  std::vector<uint8_t> t((size_t)w * h);
  const int r = 2;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
    {
      int s = 0, c = 0;
      for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
        {
          const int xx = std::min(std::max(x + dx, 0), w - 1), yy = std::min(std::max(y + dy, 0), h - 1);
          s += n[(size_t)yy * w + xx];
          c++;
        }
      t[(size_t)y * w + x] = (uint8_t)(s / c);
    }
  return t;
}

// cur(x, y) = ref(x + dx, y + dy): the expected vector (current -> reference) is (dx, dy)
static std::vector<uint8_t> shifted(const std::vector<uint8_t> &big, int bw, int w, int h, int ox, int oy)
{
  std::vector<uint8_t> o((size_t)w * h);
  for (int y = 0; y < h; y++)
    std::memcpy(&o[(size_t)y * w], &big[(size_t)(y + oy) * bw + ox], w);
  return o;
}

int main(int argc, char **argv)
{
  const int w = argc > 2 ? std::atoi(argv[1]) : 3840, h = argc > 2 ? std::atoi(argv[2]) : 2160;
  Driver &d = Driver::get();
  if (!d.ok)
  {
    std::printf("NVENC not available: %s\n", d.err.c_str());
    return 1;
  }
  const int M = 160; // margin of the big texture
  const int bw = w + 2 * M, bh = h + 2 * M;
  std::printf("building %dx%d texture...\n", bw, bh);
  const std::vector<uint8_t> big = texture(bw, bh, 12345);
  const std::vector<uint8_t> ref = shifted(big, bw, w, h, M, M);

  struct Shift { int dx, dy; };
  const Shift shifts[] = { { 5, -3 }, { 0, 0 }, { 17, 9 }, { 40, -24 }, { 64, 32 }, { 120, -60 }, { -150, 0 }, { 0, 100 } };

  for (int hevc = std::getenv("PROBE_HEVC_ONLY") ? 1 : 0; hevc < (std::getenv("PROBE_SKIP") ? 0 : 2); hevc++)
    for (int preset : { 1, 4, 7 })
      if (!std::getenv("PROBE_P4_ONLY") || preset == 4)
    {
      Params p;
      p.w = w;
      p.h = h;
      p.hevc = hevc != 0;
      p.preset = preset;
      Session s(p);
      if (!s.open())
      {
        std::printf("%s P%d: open failed: %s\n", hevc ? "HEVC" : "H264", preset, s.err.c_str());
        continue;
      }
      std::printf("\n== %s P%d ==\n", hevc ? "HEVC" : "H264", preset);
      std::vector<Mv> out;
      for (const Shift &sh : shifts)
      {
        const std::vector<uint8_t> cur = shifted(big, bw, w, h, M + sh.dx, M + sh.dy);
        Stats st;
        if (!s.run(cur.data(), ref.data(), w, out, &st))
        {
          std::printf("run failed: %s\n", s.err.c_str());
          break;
        }
        // interior cells (8 px away from the picture border plus the shift): exact at quarter pel = 4*shift
        long long exact = 0, total = 0, near1 = 0;
        std::map<std::pair<int, int>, int> hist;
        for (int gy = 0; gy < s.gh(); gy++)
          for (int gx = 0; gx < s.gw(); gx++)
          {
            const int x = gx * 8, y = gy * 8;
            if (x < 16 + std::abs(sh.dx) || y < 16 + std::abs(sh.dy) || x + 24 + std::abs(sh.dx) > w || y + 24 + std::abs(sh.dy) > h)
              continue;
            const Mv m = out[(size_t)gy * s.gw() + gx];
            total++;
            exact += (m.x == 4 * sh.dx && m.y == 4 * sh.dy);
            near1 += (std::abs(m.x - 4 * sh.dx) <= 4 && std::abs(m.y - 4 * sh.dy) <= 4);
            hist[{ m.x, m.y }]++;
          }
        std::vector<std::pair<int, std::pair<int, int>>> top;
        for (auto &kv : hist)
          top.push_back({ kv.second, kv.first });
        std::sort(top.rbegin(), top.rend());
        std::printf("shift %4d,%4d: exact %5.1f%%  within 1 px %5.1f%%  top:", sh.dx, sh.dy, 100.0 * exact / std::max(1LL, total),
          100.0 * near1 / std::max(1LL, total));
        for (size_t i = 0; i < std::min<size_t>(3, top.size()); i++)
          std::printf(" (%g,%g)x%d", top[i].second.first / 4.0, top[i].second.second / 4.0, top[i].first);
        std::printf("\n");
        if (sh.dx == 5)
        {
          if (!hevc)
            std::printf("  intra MBs %lld, partitions 16x16 %lld 8x8 %lld 16x8 %lld 8x16 %lld; 16x16 MBs with mv[0]==mv[1]==mv[3]: %lld\n", st.intra, st.part[0], st.part[1],
              st.part[2], st.part[3], st.same4_16x16);
          else
          {
            std::printf("  CUs 8 %lld 16 %lld 32 %lld 64 %lld, units %lld of %d, last %lld, MV bytes/CU %zu\n", st.cu[0], st.cu[1], st.cu[2],
              st.cu[3], st.hevc_units, s.gw() * s.gh(), st.last, sizeof(NV_ENC_HEVC_MV_DATA));
            std::printf("  partition modes:");
            for (int i = 0; i < 8; i++)
              std::printf(" %lld", st.pm[i]);
            std::printf("  cu types: %lld %lld %lld %lld\n", st.cutype[0], st.cutype[1], st.cutype[2], st.cutype[3]);
            for (auto &l : st.dump)
              std::printf("    %s\n", l.c_str());
          }
        }
      }
      // speed: 1..4 threads, each with its own session, 40 pairs each
      for (int th = 1; th <= 4; th++)
      {
        std::vector<std::unique_ptr<Session>> ss;
        for (int i = 0; i < th; i++)
        {
          ss.push_back(std::make_unique<Session>(p));
          if (!ss.back()->open())
          {
            std::printf("  %d threads: session %d failed: %s\n", th, i + 1, ss.back()->err.c_str());
            ss.clear();
            break;
          }
        }
        if (ss.empty())
          break;
        const std::vector<uint8_t> cur = shifted(big, bw, w, h, M + 5, M - 3);
        const int N = 40;
        std::atomic<int> bad{ 0 };
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int i = 0; i < th; i++)
          ts.emplace_back([&, i] {
            std::vector<Mv> o;
            for (int k = 0; k < N; k++)
              if (!ss[i]->run(cur.data(), ref.data(), w, o))
                bad++;
          });
        for (auto &t : ts)
          t.join();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %d thread(s): %.2f ms per pair (wall / pairs, incl. upload)%s\n", th, ms / (N * th), bad ? " ERRORS" : "");
      }
    }

  // history dependence: the same pair after different previous pairs, same session
  {
    Params p;
    p.w = w;
    p.h = h;
    Session s(p);
    if (s.open())
    {
      std::vector<uint8_t> big2 = texture(bw, bh, 777);
      const std::vector<uint8_t> a = shifted(big2, bw, w, h, M + 7, M + 2), ar = shifted(big, bw, w, h, M + 3, M + 1); // unrelated content: vectors depend on the predictors
      const std::vector<uint8_t> b = shifted(big, bw, w, h, M + 30, M - 20);
      std::vector<Mv> o1, o2, o3, tmp;
      if (!s.run(a.data(), ar.data(), w, o1))
        std::printf("history run failed: %s\n", s.err.c_str());
      s.run(b.data(), ref.data(), w, tmp);
      s.run(a.data(), ar.data(), w, o2);
      s.run(a.data(), ar.data(), w, o3);
      auto diff = [](const std::vector<Mv> &x, const std::vector<Mv> &y) {
        size_t d = 0;
        for (size_t i = 0; i < x.size(); i++)
          d += x[i].x != y[i].x || x[i].y != y[i].y;
        return d;
      };
      std::printf("\nhistory: same pair after another pair differs in %zu of %zu cells; repeated directly: %zu\n", diff(o1, o2), o1.size(),
        diff(o2, o3));
    }
  }

  // ME-only capability per codec, straight from the driver
  {
    Driver &dd = Driver::get();
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op = {};
    op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    op.device = dd.ctx;
    op.apiVersion = NVENCAPI_VERSION;
    void *enc = nullptr;
    if (dd.api.nvEncOpenEncodeSessionEx(&op, &enc) == NV_ENC_SUCCESS)
    {
      uint32_t n = 0;
      dd.api.nvEncGetEncodeGUIDCount(enc, &n);
      std::vector<GUID> g(n);
      dd.api.nvEncGetEncodeGUIDs(enc, g.data(), n, &n);
      const struct { const GUID *g; const char *name; } codecs[] = { { &NV_ENC_CODEC_H264_GUID, "H.264" }, { &NV_ENC_CODEC_HEVC_GUID, "HEVC" },
        { &NV_ENC_CODEC_AV1_GUID, "AV1" } };
      for (const auto &c : codecs)
      {
        bool listed = false;
        for (const GUID &x : g)
          listed = listed || std::memcmp(&x, c.g, sizeof(GUID)) == 0;
        NV_ENC_CAPS_PARAM cp = {};
        cp.version = NV_ENC_CAPS_PARAM_VER;
        cp.capsToQuery = NV_ENC_CAPS_SUPPORT_MEONLY_MODE;
        int v = -1;
        const NVENCSTATUS st = dd.api.nvEncGetEncodeCaps(enc, *c.g, &cp, &v);
        std::printf("%s: encoder listed %d, ME-only cap %d (status %d)\n", c.name, listed, v, (int)st);
      }
      dd.api.nvEncDestroyEncoder(enc);
    }
    if (std::getenv("PROBE_CAPS_ONLY"))
      return 0;
  }

  // search range through the hinted pyramid (fresh sessions, deterministic)
  for (int levels = 1; levels <= 3; levels++)
  {
    Params p;
    p.preset = 4;
    std::printf("\npyramid, %d level(s), H.264 P4:\n", levels);
    for (const Shift &sh : shifts)
    {
      const std::vector<uint8_t> cur = shifted(big, bw, w, h, M + sh.dx, M + sh.dy);
      std::vector<Mv> out;
      int gw = 0, gh = 0;
      std::string err;
      const auto t0 = std::chrono::steady_clock::now();
      if (!pyramid_me(cur, ref, w, h, p, levels, out, gw, gh, err))
      {
        std::printf("  failed: %s\n", err.c_str());
        break;
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      long long exact = 0, total = 0;
      for (int gy = 0; gy < gh; gy++)
        for (int gx = 0; gx < gw; gx++)
        {
          const int x = gx * 8, y = gy * 8;
          if (x < 16 + std::abs(sh.dx) || y < 16 + std::abs(sh.dy) || x + 24 + std::abs(sh.dx) > w || y + 24 + std::abs(sh.dy) > h)
            continue;
          const Mv m = out[(size_t)gy * gw + gx];
          total++;
          exact += (m.x == 4 * sh.dx && m.y == 4 * sh.dy);
        }
      std::printf("  shift %4d,%4d: exact %5.1f%%  (%.1f ms)\n", sh.dx, sh.dy, 100.0 * exact / std::max(1LL, total), ms);
    }
  }

  // does the ME use chroma? flat luma, texture only in U / V, shift (5, -3)
  {
    Params p;
    p.w = w;
    p.h = h;
    Session s(p);
    if (s.open())
    {
      const int cw = w / 2, chh = h / 2;
      auto uvplane = [&](int dx, int dy) {
        std::vector<uint8_t> uv((size_t)w * ((h + 1) / 2));
        for (int y = 0; y < chh; y++)
          for (int x = 0; x < cw; x++)
          {
            const uint8_t v = big[(size_t)(y + M / 2 + dy) * bw + x + M / 2 + dx];
            uv[(size_t)y * w + 2 * x] = v;
            uv[(size_t)y * w + 2 * x + 1] = (uint8_t)(255 - v);
          }
        return uv;
      };
      const std::vector<uint8_t> flat((size_t)w * h, 128), uvc = uvplane(5, -3), uvr = uvplane(0, 0);
      std::vector<Mv> o;
      if (s.run(flat.data(), flat.data(), w, o, nullptr, uvc.data(), uvr.data()))
      {
        size_t hit = 0, zero = 0;
        for (const Mv &m : o)
        {
          hit += m.x == 40 && m.y == -24; // (5, -3) chroma samples = (10, -6) luma px = (40, -24) quarter pel
          zero += m.x == 0 && m.y == 0;
        }
        std::printf("\nchroma-only texture, chroma shift (5,-3): %zu of %zu cells exact, %zu zero\n", hit, o.size(), zero);
      }
      else
        std::printf("chroma test failed: %s\n", s.err.c_str());
    }
  }

  // cost of a fresh session per call at this size
  {
    Params p;
    p.w = w;
    p.h = h;
    const std::vector<uint8_t> cur = shifted(big, bw, w, h, M + 5, M - 3);
    for (int fresh = 0; fresh < 2; fresh++)
    {
      Session s(p);
      s.fresh = fresh != 0;
      s.open();
      std::vector<Mv> o;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 20; i++)
        if (!s.run(cur.data(), ref.data(), w, o))
          std::printf("run failed: %s\n", s.err.c_str());
      std::printf("%s session per call: %.2f ms per pair (1 thread)\n", fresh ? "fresh" : "kept ", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20);
    }
  }

  // session cap: open H.264 P4 sessions until one fails (max 40)
  {
    Params p;
    p.w = 1920;
    p.h = 1080;
    std::vector<std::unique_ptr<Session>> ss;
    std::string e;
    for (int i = 0; i < 40; i++)
    {
      auto s = std::make_unique<Session>(p);
      if (!s->open())
      {
        e = s->err;
        break;
      }
      ss.push_back(std::move(s));
    }
    std::printf("\nsession cap: %zu sessions opened%s%s\n", ss.size(), e.empty() ? "" : ", then: ", e.c_str());
  }
  return 0;
}
