// D3D12 video motion estimator probe (DTL's DX12_ME path) with the tests of the NVENC probe: uniform shifts (search
// range), a chroma-only texture (does the ME use chroma?), history dependence, speed.   dx12probe.exe [w h]
#include "dx12me.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>

using namespace dx12me;

static std::vector<uint8_t> texture(int w, int h, unsigned seed)
{
  std::vector<int> n((size_t)w * h);
  for (auto &v : n)
  {
    seed = seed * 1664525u + 1013904223u;
    v = (seed >> 24) & 255;
  }
  std::vector<uint8_t> t((size_t)w * h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
    {
      int s = 0, c = 0;
      for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
        {
          s += n[(size_t)std::min(std::max(y + dy, 0), h - 1) * w + std::min(std::max(x + dx, 0), w - 1)];
          c++;
        }
      t[(size_t)y * w + x] = (uint8_t)(s / c);
    }
  return t;
}

static std::vector<uint8_t> cut(const std::vector<uint8_t> &big, int bw, int w, int h, int ox, int oy)
{
  std::vector<uint8_t> o((size_t)w * h);
  for (int y = 0; y < h; y++)
    std::memcpy(&o[(size_t)y * w], &big[(size_t)(y + oy) * bw + ox], w);
  return o;
}

int main(int argc, char **argv)
{
  const int w = argc > 2 ? std::atoi(argv[1]) : 1920, h = argc > 2 ? std::atoi(argv[2]) : 1080;
  const int M = 160, bw = w + 2 * M, bh = h + 2 * M;
  const std::vector<uint8_t> big = texture(bw, bh, 12345), ref = cut(big, bw, w, h, M, M);
  struct Shift { int dx, dy; };
  const Shift shifts[] = { { 5, -3 }, { 0, 0 }, { 17, 9 }, { 40, -24 }, { 64, 32 }, { 120, -60 }, { -150, 0 }, { 0, 100 } };

  for (int bs : { 8, 16 })
  {
    Estimator e;
    if (!e.open(w, h, bs))
    {
      std::printf("%dx%d blocks: open failed: %s (%s)\n", bs, bs, e.err.c_str(), e.info.c_str());
      continue;
    }
    std::printf("\n== D3D12 ME, %dx%d blocks, %dx%d ==\n%s\n", bs, bs, w, h, e.info.c_str());
    std::vector<Mv> out;
    for (const Shift &sh : shifts)
    {
      const std::vector<uint8_t> cur = cut(big, bw, w, h, M + sh.dx, M + sh.dy);
      if (!e.run(cur.data(), ref.data(), nullptr, nullptr, out))
      {
        std::printf("run failed: %s\n", e.err.c_str());
        return 1;
      }
      long long exact = 0, total = 0;
      std::map<std::pair<int, int>, int> hist;
      for (int gy = 0; gy < e.gh; gy++)
        for (int gx = 0; gx < e.gw; gx++)
        {
          const int x = gx * bs, y = gy * bs;
          if (x < 16 + std::abs(sh.dx) || y < 16 + std::abs(sh.dy) || x + bs + 16 + std::abs(sh.dx) > w || y + bs + 16 + std::abs(sh.dy) > h)
            continue;
          const Mv m = out[(size_t)gy * e.gw + gx];
          total++;
          exact += m.x == 4 * sh.dx && m.y == 4 * sh.dy;
          hist[{ m.x, m.y }]++;
        }
      std::vector<std::pair<int, std::pair<int, int>>> top;
      for (auto &kv : hist)
        top.push_back({ kv.second, kv.first });
      std::sort(top.rbegin(), top.rend());
      std::printf("shift %4d,%4d: exact %5.1f%%  top:", sh.dx, sh.dy, 100.0 * exact / std::max(1LL, total));
      for (size_t i = 0; i < std::min<size_t>(3, top.size()); i++)
        std::printf(" (%g,%g)x%d", top[i].second.first / 4.0, top[i].second.second / 4.0, top[i].first);
      std::printf("\n");
    }

    // chroma-only texture: flat luma, texture in U / V shifted by (5, -3) chroma samples = (10, -6) luma px
    {
      auto uvplane = [&](int dx, int dy) {
        std::vector<uint8_t> uv((size_t)w * ((h + 1) / 2));
        for (int y = 0; y < h / 2; y++)
          for (int x = 0; x < w / 2; x++)
          {
            const uint8_t v = big[(size_t)(y + M / 2 + dy) * bw + x + M / 2 + dx];
            uv[(size_t)y * w + 2 * x] = v;
            uv[(size_t)y * w + 2 * x + 1] = (uint8_t)(255 - v);
          }
        return uv;
      };
      const std::vector<uint8_t> flat((size_t)w * h, 128), uvc = uvplane(5, -3), uvr = uvplane(0, 0);
      if (e.run(flat.data(), flat.data(), uvc.data(), uvr.data(), out))
      {
        size_t hit = 0, zero = 0;
        for (const Mv &m : out)
        {
          hit += m.x == 40 && m.y == -24;
          zero += m.x == 0 && m.y == 0;
        }
        std::printf("chroma-only texture, expected (10,-6) px: %zu of %zu blocks exact, %zu zero\n", hit, out.size(), zero);
      }
      else
        std::printf("chroma test failed: %s\n", e.err.c_str());
    }

    // history: same pair after another pair, and repeated
    {
      const std::vector<uint8_t> big2 = texture(bw, bh, 777);
      const std::vector<uint8_t> a = cut(big2, bw, w, h, M + 7, M + 2), ar = cut(big, bw, w, h, M + 3, M + 1);
      const std::vector<uint8_t> b = cut(big, bw, w, h, M + 30, M - 20);
      std::vector<Mv> o1, o2, o3, tmp;
      e.run(a.data(), ar.data(), nullptr, nullptr, o1);
      e.run(b.data(), ref.data(), nullptr, nullptr, tmp);
      e.run(a.data(), ar.data(), nullptr, nullptr, o2);
      e.run(a.data(), ar.data(), nullptr, nullptr, o3);
      size_t d12 = 0, d23 = 0;
      for (size_t i = 0; i < o1.size(); i++)
      {
        d12 += o1[i].x != o2[i].x || o1[i].y != o2[i].y;
        d23 += o2[i].x != o3[i].x || o2[i].y != o3[i].y;
      }
      std::printf("history: same pair after another pair differs in %zu of %zu blocks; repeated directly: %zu\n", d12, o1.size(), d23);
    }

    // speed (1 thread, includes upload, the three queue round trips and the readback)
    {
      const std::vector<uint8_t> cur = cut(big, bw, w, h, M + 5, M - 3);
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 40; i++)
        e.run(cur.data(), ref.data(), nullptr, nullptr, out);
      std::printf("%.2f ms per pair (1 thread, incl. upload / readback)\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 40);
    }
  }
  return 0;
}
