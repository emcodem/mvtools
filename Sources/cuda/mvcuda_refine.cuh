// Device ports of the PlaneOfBlocks search routines (Refine and everything it calls), written once over an abstract
// candidate checker so that the MAnalyse and the MRecalculate kernels run exactly the CPU control flow.
//
// The CPU routines test candidate vectors one after the other; most of them test groups of candidates around a *fixed*
// centre (the ring of an expanding search, the hexagon, the cross and grid of UMH ...). For such a group the result of
// the sequential tests equals "evaluate all, then fold in the CPU's order": a candidate is accepted when its full cost is
// strictly lower than the minimum so far (the CPU's early exits only skip work, the partial sums only grow). So every
// group is one *batch*, which the checker may evaluate in parallel. Groups whose centre depends on the previous result
// (the walk of OneTimeSearch, the half hexagons of Hex2Search) are separate, dependent batches.
//
// A checker C provides
//   int bx, by                                   the best vector so far (workarea.bestMV.x / y)
//   bool ok(int x, int y)                        IsVectorOK
//   void run_batch(int mode, int n, const int *xs, const int *ys, const int *vals, int &dir)
//        tests the n candidates in order with the semantics of CheckMV (mode M_CHECK), CheckMV2 (M_CHECK2: bx / by / minimum /
//        sad updated and dir = vals[i] on acceptance) or CheckMVdir (M_CHECKDIR: only minimum / sad and dir = vals[i]).
// All threads of the calling group run this code with identical values; the checker does the parallel work inside
// run_batch, so nothing here may diverge between threads. SearchType values are those of SearchType.h.

#pragma once

namespace mvc_refine
{
  enum
  {
    S_ONETIME = 1, S_NSTEP = 2, S_LOGARITHMIC = 4, S_EXHAUSTIVE = 8, S_HEX2 = 16, S_UMH = 32, S_HORIZONTAL = 64, S_VERTICAL = 128
  };
  enum { M_CHECK = 0, M_CHECK2 = 1, M_CHECKDIR = 2 };

  // largest number of candidates of one batch (UMH with search range 24: cross 48 + 6 grid rings of 16)
  enum { MAX_BATCH = 160 };

  // search types this code implements
  inline bool supported_type(int t)
  {
    return t == S_ONETIME || t == S_NSTEP || t == S_LOGARITHMIC || t == S_EXHAUSTIVE || t == S_HEX2 || t == S_UMH
      || t == S_HORIZONTAL || t == S_VERTICAL;
  }

  // largest search parameter accepted for a type (keeps the work per block bounded)
  inline int max_param(int t)
  {
    return t == S_EXHAUSTIVE ? 3 : 24;
  }

  // ExpandingSearch(r, s, mvx, mvy) candidates in CPU order, appended at n; returns the new count
  __device__ __forceinline__ int add_expanding(int *xs, int *ys, int n, int r, int s, int mvx, int mvy)
  {
    for (int i = -r + s; i < r; i += s)
    {
      xs[n] = mvx + i; ys[n++] = mvy - r;
      xs[n] = mvx + i; ys[n++] = mvy + r;
    }
    for (int j = -r + s; j < r; j += s)
    {
      xs[n] = mvx - r; ys[n++] = mvy + j;
      xs[n] = mvx + r; ys[n++] = mvy + j;
    }
    xs[n] = mvx - r; ys[n++] = mvy - r;
    xs[n] = mvx - r; ys[n++] = mvy + r;
    xs[n] = mvx + r; ys[n++] = mvy - r;
    xs[n] = mvx + r; ys[n++] = mvy + r;
    return n;
  }

  // one candidate, as its own (dependent) batch
  template <class C>
  __device__ __forceinline__ void check_one(C &c, int mode, int x, int y, int &dir, int val)
  {
    int xs[1] = { x }, ys[1] = { y }, vs[1] = { val };
    c.run_batch(mode, 1, xs, ys, vs, dir);
  }

  template <class C>
  __device__ void expanding(C &c, int r, int s, int mvx, int mvy)
  {
    int xs[4 * 3 * 4 + 8], ys[4 * 3 * 4 + 8], dummy = 0; // radius <= 3 (exhaustive) or 1 (square refine)
    const int n = add_expanding(xs, ys, 0, r, s, mvx, mvy);
    c.run_batch(M_CHECK, n, xs, ys, nullptr, dummy);
  }

  template <class C>
  __device__ void one_time(C &c, int length)
  {
    int direction = 0;
    int dx = c.bx, dy = c.by;
    {
      const int xs[2] = { dx - length, dx + length }, ys[2] = { dy, dy }, vs[2] = { 2, 1 };
      c.run_batch(M_CHECK2, 2, xs, ys, vs, direction);
    }
    if (direction == 1)
    {
      while (direction)
      {
        direction = 0;
        dx += length;
        check_one(c, M_CHECK2, dx + length, dy, direction, 1);
      }
    }
    else if (direction == 2)
    {
      while (direction)
      {
        direction = 0;
        dx -= length;
        check_one(c, M_CHECK2, dx - length, dy, direction, 1);
      }
    }
    {
      const int xs[2] = { dx, dx }, ys[2] = { dy - length, dy + length }, vs[2] = { 2, 1 };
      c.run_batch(M_CHECK2, 2, xs, ys, vs, direction);
    }
    if (direction == 1)
    {
      while (direction)
      {
        direction = 0;
        dy += length;
        check_one(c, M_CHECK2, dx, dy + length, direction, 1);
      }
    }
    else if (direction == 2)
    {
      while (direction)
      {
        direction = 0;
        dy -= length;
        check_one(c, M_CHECK2, dx, dy - length, direction, 1);
      }
    }
  }

  template <class C>
  __device__ void nstep(C &c, int stp)
  {
    int dummy = 0;
    for (int length = stp; length > 0; length--)
    {
      const int dx = c.bx, dy = c.by;
      const int xs[8] = { dx + length, dx + length, dx + length, dx, dx, dx - length, dx - length, dx - length };
      const int ys[8] = { dy + length, dy, dy - length, dy - length, dy + length, dy + length, dy, dy - length };
      c.run_batch(M_CHECK, 8, xs, ys, nullptr, dummy);
    }
  }

  // DiamondSearch: the candidates of each step are relative to the centre taken at the start of the step
  template <class C>
  __device__ void diamond(C &c, int length)
  {
    int direction = 15;
    while (direction > 0)
    {
      int dx = c.bx, dy = c.by;
      int last = direction;
      direction = 0;
      {
        int xs[4], ys[4], vs[4], n = 0;
        if (last & 1) { xs[n] = dx + length; ys[n] = dy; vs[n++] = 1; }
        if (last & 2) { xs[n] = dx - length; ys[n] = dy; vs[n++] = 2; }
        if (last & 4) { xs[n] = dx; ys[n] = dy + length; vs[n++] = 4; }
        if (last & 8) { xs[n] = dx; ys[n] = dy - length; vs[n++] = 8; }
        if (n)
          c.run_batch(M_CHECK2, n, xs, ys, vs, direction);
      }
      if (direction)
      {
        last = direction;
        dx = c.bx;
        dy = c.by;
        if (last & 3)
        {
          const int xs[2] = { dx, dx }, ys[2] = { dy + length, dy - length }, vs[2] = { 4, 8 };
          c.run_batch(M_CHECK2, 2, xs, ys, vs, direction);
        }
        else
        {
          const int xs[2] = { dx + length, dx - length }, ys[2] = { dy, dy }, vs[2] = { 1, 2 };
          c.run_batch(M_CHECK2, 2, xs, ys, vs, direction);
        }
      }
      else
      {
        int xs[4], ys[4], vs[4], n = 0;
#define DC(ax, ay, v) do { xs[n] = dx + (ax) * length; ys[n] = dy + (ay) * length; vs[n++] = (v); } while (0)
        switch (last)
        {
        case 1: DC(1, 1, 1 + 4); DC(1, -1, 1 + 8); break;
        case 2: DC(-1, 1, 2 + 4); DC(-1, -1, 2 + 8); break;
        case 4: DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); break;
        case 8: DC(1, -1, 1 + 8); DC(-1, -1, 2 + 8); break;
        case 1 + 4: DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); DC(1, -1, 1 + 8); break;
        case 2 + 4: DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); DC(-1, -1, 2 + 8); break;
        case 1 + 8: DC(1, 1, 1 + 4); DC(-1, -1, 2 + 8); DC(1, -1, 1 + 8); break;
        case 2 + 8: DC(-1, -1, 2 + 8); DC(-1, 1, 2 + 4); DC(1, -1, 1 + 8); break;
        default: DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); DC(1, -1, 1 + 8); DC(-1, -1, 2 + 8); break;
        }
#undef DC
        c.run_batch(M_CHECK2, n, xs, ys, vs, direction);
      }
    }
  }

  template <class C>
  __device__ void hex2(C &c, int range)
  {
    // radius 2 hexagon, repeated entries avoid a modulo (x264)
    const int hex2_t[8][2] = { {-1,-2}, {-2,0}, {-1,2}, {1,2}, {2,0}, {1,-2}, {-1,-2}, {-2,0} };
    const int mod6m1[8] = { 5,0,1,2,3,4,5,0 };
    int dir = -2;
    int bmx = c.bx, bmy = c.by;
    if (range > 1)
    {
      {
        const int xs[6] = { bmx - 2, bmx - 1, bmx + 1, bmx + 2, bmx + 1, bmx - 1 };
        const int ys[6] = { bmy, bmy + 2, bmy + 2, bmy, bmy - 2, bmy - 2 };
        const int vs[6] = { 0, 1, 2, 3, 4, 5 };
        c.run_batch(M_CHECKDIR, 6, xs, ys, vs, dir);
      }
      if (dir != -2)
      {
        bmx += hex2_t[dir + 1][0];
        bmy += hex2_t[dir + 1][1];
        for (int i = 1; i < range / 2 && c.ok(bmx, bmy); i++)
        {
          const int odir = mod6m1[dir + 1];
          dir = -2;
          const int xs[3] = { bmx + hex2_t[odir + 0][0], bmx + hex2_t[odir + 1][0], bmx + hex2_t[odir + 2][0] };
          const int ys[3] = { bmy + hex2_t[odir + 0][1], bmy + hex2_t[odir + 1][1], bmy + hex2_t[odir + 2][1] };
          const int vs[3] = { odir - 1, odir, odir + 1 };
          c.run_batch(M_CHECKDIR, 3, xs, ys, vs, dir);
          if (dir == -2)
            break;
          bmx += hex2_t[dir + 1][0];
          bmy += hex2_t[dir + 1][1];
        }
      }
      c.bx = bmx;
      c.by = bmy;
    }
    expanding(c, 1, 1, bmx, bmy);
  }

  template <class C>
  __device__ void umh(C &c, int range, int omx, int omy)
  {
    // CrossSearch(1, range, range), then the hex grid rings: all candidates are relative to the fixed centre (omx, omy)
    int xs[MAX_BATCH], ys[MAX_BATCH], dummy = 0, n = 0;
    for (int i = 1; i < range; i += 2)
    {
      xs[n] = omx - i; ys[n++] = omy;
      xs[n] = omx + i; ys[n++] = omy;
    }
    for (int j = 1; j < range; j += 2)
    {
      xs[n] = omx; ys[n++] = omy + j;
      xs[n] = omx; ys[n++] = omy - j;
    }
    const int hex4[16][2] = { {-4, 2}, {-4, 1}, {-4, 0}, {-4,-1}, {-4,-2}, { 4,-2}, { 4,-1}, { 4, 0}, { 4, 1}, { 4, 2},
                              { 2, 3}, { 0, 4}, {-2, 3}, {-2,-3}, { 0,-4}, { 2,-3} };
    int i = 1;
    do
    {
      for (int j = 0; j < 16; j++)
      {
        xs[n] = omx + hex4[j][0] * i;
        ys[n++] = omy + hex4[j][1] * i;
      }
    } while (++i <= range / 4);
    c.run_batch(M_CHECK, n, xs, ys, nullptr, dummy);
    hex2(c, range);
  }

  // PlaneOfBlocks::Refine (tryMany off)
  template <class C>
  __device__ void refine(C &c, int type, int param)
  {
    int dummy = 0;
    switch (type)
    {
    case S_ONETIME:
      for (int i = param; i > 0; i /= 2)
        one_time(c, i);
      break;
    case S_NSTEP:
      nstep(c, param);
      break;
    case S_LOGARITHMIC:
      for (int i = param; i > 0; i /= 2)
        diamond(c, i);
      break;
    case S_EXHAUSTIVE:
    {
      // all radii are centred on the vector at the start, so the whole search is one batch
      const int mvx = c.bx, mvy = c.by;
      int xs[48], ys[48], n = 0;
      for (int i = 1; i <= param; i++)
        n = add_expanding(xs, ys, n, i, 1, mvx, mvy);
      c.run_batch(M_CHECK, n, xs, ys, nullptr, dummy);
    }
    break;
    case S_HEX2:
      hex2(c, param);
      break;
    case S_UMH:
      umh(c, param, c.bx, c.by);
      break;
    case S_HORIZONTAL:
    {
      const int mvx = c.bx, mvy = c.by;
      int xs[2 * 24], ys[2 * 24], n = 0;
      for (int i = 1; i <= param; i++)
      {
        xs[n] = mvx - i; ys[n++] = mvy;
        xs[n] = mvx + i; ys[n++] = mvy;
      }
      c.run_batch(M_CHECK, n, xs, ys, nullptr, dummy);
    }
    break;
    case S_VERTICAL:
    {
      const int mvx = c.bx, mvy = c.by;
      int xs[2 * 24], ys[2 * 24], n = 0;
      for (int i = 1; i <= param; i++)
      {
        xs[n] = mvx; ys[n++] = mvy - i;
        xs[n] = mvx; ys[n++] = mvy + i;
      }
      c.run_batch(M_CHECK, n, xs, ys, nullptr, dummy);
    }
    break;
    }
  }
}
