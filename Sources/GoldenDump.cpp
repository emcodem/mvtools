// Golden-data dump for the GPU port, see GoldenDump.h.

#include "GoldenDump.h"

#ifdef MV_GOLDEN_DUMP

#include "MVFrame.h"
#include "MVGroupOfFrames.h"
#include "MVPlane.h"

#include <windows.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

namespace golden
{
  namespace
  {
    struct Ctx
    {
      std::string dir;
      FILE *log = nullptr;
      bool raw_planes = false;
    };

    thread_local Ctx *cur = nullptr;

    const char *env_dir()
    {
      static const char *d = std::getenv("MVDUMP_DIR");
      return (d != nullptr && d[0] != 0) ? d : nullptr;
    }

    const std::set<int> &frame_set()
    {
      static std::set<int> s;
      static bool init = false;
      if (!init)
      {
        init = true;
        const char *f = std::getenv("MVDUMP_FRAMES");
        while (f != nullptr && *f != 0)
        {
          char *e = nullptr;
          const long v = std::strtol(f, &e, 10);
          if (e == f)
          {
            break;
          }
          s.insert(int(v));
          f = (*e == ',') ? e + 1 : e;
        }
      }
      return s;
    }

    uint64_t fnv1a(uint64_t h, const uint8_t *p, size_t n)
    {
      for (size_t i = 0; i < n; ++i)
      {
        h = (h ^ p[i]) * 1099511628211ull;
      }
      return h;
    }
  }

  bool enabled()
  {
    return env_dir() != nullptr;
  }

  bool want_frame(int nsrc)
  {
    return enabled() && frame_set().count(nsrc) != 0;
  }

  bool active()
  {
    return cur != nullptr;
  }

  bool begin(const char *filter, int nsrc, bool backward)
  {
    if (cur != nullptr || !want_frame(nsrc))
    {
      return false;
    }
    Ctx *c = new Ctx();
    char name[128];
    std::snprintf(name, sizeof(name), "%s_n%d_%c", filter, nsrc, backward ? 'b' : 'f');
    c->dir = std::string(env_dir()) + "\\" + name;
    CreateDirectoryA(c->dir.c_str(), nullptr);
    c->log = std::fopen((c->dir + "\\log.txt").c_str(), "w");
    const char *p = std::getenv("MVDUMP_PLANES");
    c->raw_planes = (p != nullptr && p[0] == '1');
    cur = c;
    return c->log != nullptr;
  }

  void end()
  {
    if (cur == nullptr)
    {
      return;
    }
    if (cur->log != nullptr)
    {
      std::fclose(cur->log);
    }
    delete cur;
    cur = nullptr;
  }

  void logf(const char *fmt, ...)
  {
    if (cur == nullptr || cur->log == nullptr)
    {
      return;
    }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(cur->log, fmt, ap);
    va_end(ap);
    std::fputc('\n', cur->log);
  }

  void write_bin(const char *name, const void *data, size_t bytes)
  {
    if (cur == nullptr)
    {
      return;
    }
    FILE *f = std::fopen((cur->dir + "\\" + name).c_str(), "wb");
    if (f != nullptr)
    {
      std::fwrite(data, 1, bytes, f);
      std::fclose(f);
    }
    logf("file %s bytes %zu fnv1a64 %016llx", name, bytes,
      (unsigned long long)fnv1a(14695981039346656037ull, static_cast<const uint8_t *>(data), bytes));
  }

  void dump_gof(const char *prefix, MVGroupOfFrames &gof, int nlevels, bool chroma)
  {
    if (cur == nullptr)
    {
      return;
    }
    static const char plane_name[3] = { 'Y', 'U', 'V' };
    static const MVPlaneSet plane_set[3] = { YPLANE, UPLANE, VPLANE };

    for (int level = 0; level < nlevels; ++level)
    {
      MVFrame *frame = gof.GetFrame(level);
      for (int pl = 0; pl < (chroma ? 3 : 1); ++pl)
      {
        MVPlane *plane = frame->GetPlane(plane_set[pl]);
        const int pel = plane->DbgPel();
        const int row_bytes = plane->GetExtendedWidth() * plane->DbgPixelSize();
        const int rows = plane->GetExtendedHeight();
        logf("plane %s L%d %c width %d height %d hpad %d vpad %d pel %d pixelsize %d pitch %d",
          prefix, level, plane_name[pl], plane->GetWidth(), plane->GetHeight(), plane->GetHPadding(),
          plane->GetVPadding(), pel, plane->DbgPixelSize(), plane->GetPitch());

        for (int sub = 0; sub < pel * pel; ++sub)
        {
          // hash over the extended (padded) area only, rows without the pitch gap
          const uint8_t *base = plane->DbgSubPlane(sub);
          uint64_t h = 14695981039346656037ull;
          for (int y = 0; y < rows; ++y)
          {
            h = fnv1a(h, base + size_t(y) * plane->GetPitch(), row_bytes);
          }
          logf("hash %s L%d %c sub%d %016llx", prefix, level, plane_name[pl], sub, (unsigned long long)h);

          if (cur->raw_planes)
          {
            char name[96];
            std::snprintf(name, sizeof(name), "%s_L%d_%c_%d.bin", prefix, level, plane_name[pl], sub);
            FILE *f = std::fopen((cur->dir + "\\" + name).c_str(), "wb");
            if (f != nullptr)
            {
              for (int y = 0; y < rows; ++y)
              {
                std::fwrite(base + size_t(y) * plane->GetPitch(), 1, row_bytes, f);
              }
              std::fclose(f);
            }
          }
        }
      }
    }
  }
}

#endif // MV_GOLDEN_DUMP
