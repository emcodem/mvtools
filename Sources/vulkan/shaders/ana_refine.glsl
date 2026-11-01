// The search routines of mvcuda_refine.cuh (OneTimeSearch, NStepSearch, DiamondSearch, ExpandingSearch, Hex2Search,
// UMHSearch, the horizontal / vertical searches) and the wide "bad vector" search of k_wave (wave_bad_search), rewritten
// as one resumable program: next_batch() writes the next candidate list and returns its fold mode, given the fold result
// of the previous batch (ck_*, rs_dir); refine_run() loops over it. So the candidate evaluation exists once per call site
// of refine_run (as one function per call site of every routine it was inlined dozens of times, and pipeline creation
// took a minute). The candidates, their order, the batches, the modes and the direction values are exactly those of
// mvcuda_refine.cuh (a batch there = a batch here, except that fixed-centre lists may be cut differently, which gives
// the same sequential fold). Included by ana_common.glsl. The bad-vector search needs BAD_RANGE (badrange) defined by
// the kernel (0 elsewhere).

#ifndef BAD_RANGE
#define BAD_RANGE 0
#endif

#define T_BAD 1000 // the wide search for a bad vector (PseudoEPZSearch)

// fixed-centre lists, generated in parallel and folded in batches of ANA_MAXB
#define LK_EXH 0  // ExpandingSearch rings 1 .. rs_lr around (rs_lox, rs_loy), step 1 (the exhaustive refine)
#define LK_UMH 1  // CrossSearch(1, r, r) + the hex grid rings of UMHSearch(r) around (rs_lox, rs_loy)
#define LK_RING 2 // ExpandingSearch(rs_lr, rs_ls, rs_lox, rs_loy)

// program counter values
#define P_DONE 0
#define P_LIST 1
#define P_OT_START 10
#define P_OT_A1 11
#define P_OT_XP 12
#define P_OT_XM 13
#define P_OT_B 14
#define P_OT_B1 15
#define P_OT_YP 16
#define P_OT_YM 17
#define P_OT_NEXT 18
#define P_NS 20
#define P_LG_START 30
#define P_LG_LOOP 31
#define P_LG_AFTER1 32
#define P_HX_INIT 40
#define P_HX_A 41
#define P_HX_A1 42
#define P_HX_LOOP 43
#define P_HX_L1 44
#define P_HX_SET 45
#define P_HX_END 46
#define P_UMH 50
#define P_HV 60
#define P_BD_INIT 70
#define P_BD_EXL 71
#define P_BD_EXL2 72
#define P_BD_SMALL0 73
#define P_BD_SMALL 74
#define P_BD_SMALL2 75

const int HEX2X[8] = int[8](-1, -2, -1, 1, 2, 1, -1, -2);
const int HEX2Y[8] = int[8](-2, 0, 2, 2, 0, -2, -2, 0);
const int MOD6M1[8] = int[8](5, 0, 1, 2, 3, 4, 5, 0);
const int HEX4X[16] = int[16](-4, -4, -4, -4, -4, 4, 4, 4, 4, 4, 2, 0, -2, -2, 0, 2);
const int HEX4Y[16] = int[16](2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 3, 4, 3, -3, -4, -3);

int rs_type, rs_param, rs_pc, rs_dir;
int rs_i, rs_len, rs_dx, rs_dy, rs_last, rs_bmx, rs_bmy, rs_cnt, rs_found, rs_smx, rs_smy, rs_hrange, rs_hnext;
int rs_lk, rs_lr, rs_ls, rs_lox, rs_loy, rs_b0, rs_total, rs_lnext;

// candidates per side of ExpandingSearch(r, s): i = -r + s, -r + 2s, ... < r
int ring_side(int r, int s) { return -r + s >= r ? 0 : (2 * r - 1 - s) / s + 1; }

// candidate k of the current fixed-centre list
ivec2 list_cand(int k)
{
  if (rs_lk == LK_EXH)
  {
    int rr = 1;
    while (4 * rr * (rr + 1) <= k)
      rr++;
    return ivec2(rs_lox, rs_loy) + expanding_point(rr, k - 4 * rr * (rr - 1));
  }
  if (rs_lk == LK_UMH)
  {
    const int nx = rs_lr / 2; // the odd i < range
    if (k < 2 * nx)
    {
      const int i = 2 * (k >> 1) + 1;
      return ivec2(rs_lox + ((k & 1) != 0 ? i : -i), rs_loy);
    }
    if (k < 4 * nx)
    {
      const int k2 = k - 2 * nx, j = 2 * (k2 >> 1) + 1;
      return ivec2(rs_lox, rs_loy + ((k2 & 1) != 0 ? -j : j));
    }
    const int k3 = k - 4 * nx, ring = k3 / 16 + 1, e = k3 - (ring - 1) * 16;
    return ivec2(rs_lox + HEX4X[e] * ring, rs_loy + HEX4Y[e] * ring);
  }
  // LK_RING: top / bottom sides (interleaved), left / right sides (interleaved), the corners
  const int r = rs_lr, s = rs_ls, m = ring_side(r, s);
  if (k < 2 * m)
    return ivec2(rs_lox - r + s + (k >> 1) * s, rs_loy + ((k & 1) != 0 ? r : -r));
  if (k < 4 * m)
  {
    const int k2 = k - 2 * m;
    return ivec2(rs_lox + ((k2 & 1) != 0 ? r : -r), rs_loy - r + s + (k2 >> 1) * s);
  }
  const int k3 = k - 4 * m;
  return ivec2(rs_lox + ((k3 & 2) != 0 ? r : -r), rs_loy + ((k3 & 1) != 0 ? r : -r));
}

// starts a fixed-centre list; after its last batch the program continues at next
void start_list(int kind, int r, int s, int ox, int oy, int next)
{
  rs_lk = kind;
  rs_lr = r;
  rs_ls = s;
  rs_lox = ox;
  rs_loy = oy;
  rs_b0 = 0;
  if (kind == LK_EXH)
    rs_total = 4 * r * (r + 1);
  else if (kind == LK_UMH)
    rs_total = 4 * (r / 2) + 16 * max(1, r / 4);
  else
    rs_total = 4 * ring_side(r, s) + 4;
  rs_lnext = next;
  rs_pc = P_LIST;
}

void refine_init(int type, int param)
{
  rs_type = type;
  rs_param = param;
  rs_dir = 0;
  rs_pc = P_DONE;
  if (type == S_ONETIME)
  {
    rs_i = param;
    rs_pc = P_OT_START;
  }
  else if (type == S_NSTEP)
  {
    rs_len = param;
    rs_pc = P_NS;
  }
  else if (type == S_LOGARITHMIC)
  {
    rs_i = param;
    rs_pc = P_LG_START;
  }
  else if (type == S_EXHAUSTIVE)
    start_list(LK_EXH, param, 1, ck_bx, ck_by, P_DONE);
  else if (type == S_HEX2)
  {
    rs_hrange = param;
    rs_hnext = P_DONE;
    rs_pc = P_HX_INIT;
  }
  else if (type == S_UMH)
    rs_pc = P_UMH;
  else if (type == S_HORIZONTAL || type == S_VERTICAL)
    rs_pc = P_HV;
  else if (type == T_BAD)
    rs_pc = P_BD_INIT;
}

// Writes the next batch (n candidates, fold mode) and returns true, or false when the search is over. Uniform.
bool next_batch(out int mode, out int n)
{
  mode = M_CHECK;
  n = 0;
  const int tid = ana_tid();
  for (int guard = 0; guard < 64; guard++) // state changes between two batches: a few
  {
    const int pc = rs_pc;
    if (pc == P_DONE)
      return false;
    if (pc == P_LIST)
    {
      if (rs_b0 >= rs_total)
      {
        rs_pc = rs_lnext;
        continue;
      }
      n = min(ANA_MAXB, rs_total - rs_b0);
      for (int k = tid; k < n; k += ANA_WG)
      {
        const ivec2 c = list_cand(rs_b0 + k);
        s_xs[e_bo + k] = c.x;
        s_ys[e_bo + k] = c.y;
      }
      rs_b0 += n;
      return true;
    }

    // ---- OneTimeSearch(length) for length = param, param / 2, ... 1
    if (pc == P_OT_START)
    {
      if (rs_i <= 0)
      {
        rs_pc = P_DONE;
        continue;
      }
      rs_len = rs_i;
      rs_dir = 0;
      rs_dx = ck_bx;
      rs_dy = ck_by;
      cand(0, rs_dx - rs_len, rs_dy, 2);
      cand(1, rs_dx + rs_len, rs_dy, 1);
      mode = M_CHECK2;
      n = 2;
      rs_pc = P_OT_A1;
      return true;
    }
    if (pc == P_OT_A1)
    {
      rs_pc = rs_dir == 1 ? P_OT_XP : (rs_dir == 2 ? P_OT_XM : P_OT_B);
      continue;
    }
    if (pc == P_OT_XP || pc == P_OT_XM)
    {
      if (rs_dir == 0)
      {
        rs_pc = P_OT_B;
        continue;
      }
      const int l = pc == P_OT_XP ? rs_len : -rs_len;
      rs_dir = 0;
      rs_dx += l;
      cand(0, rs_dx + l, rs_dy, 1);
      mode = M_CHECK2;
      n = 1;
      return true;
    }
    if (pc == P_OT_B)
    {
      cand(0, rs_dx, rs_dy - rs_len, 2);
      cand(1, rs_dx, rs_dy + rs_len, 1);
      mode = M_CHECK2;
      n = 2;
      rs_pc = P_OT_B1;
      return true;
    }
    if (pc == P_OT_B1)
    {
      rs_pc = rs_dir == 1 ? P_OT_YP : (rs_dir == 2 ? P_OT_YM : P_OT_NEXT);
      continue;
    }
    if (pc == P_OT_YP || pc == P_OT_YM)
    {
      if (rs_dir == 0)
      {
        rs_pc = P_OT_NEXT;
        continue;
      }
      const int l = pc == P_OT_YP ? rs_len : -rs_len;
      rs_dir = 0;
      rs_dy += l;
      cand(0, rs_dx, rs_dy + l, 1);
      mode = M_CHECK2;
      n = 1;
      return true;
    }
    if (pc == P_OT_NEXT)
    {
      rs_i /= 2;
      rs_pc = P_OT_START;
      continue;
    }

    // ---- NStepSearch
    if (pc == P_NS)
    {
      if (rs_len <= 0)
      {
        rs_pc = P_DONE;
        continue;
      }
      const int dx = ck_bx, dy = ck_by, l = rs_len;
      cand(0, dx + l, dy + l, 0);
      cand(1, dx + l, dy, 0);
      cand(2, dx + l, dy - l, 0);
      cand(3, dx, dy - l, 0);
      cand(4, dx, dy + l, 0);
      cand(5, dx - l, dy + l, 0);
      cand(6, dx - l, dy, 0);
      cand(7, dx - l, dy - l, 0);
      rs_len--;
      n = 8;
      return true;
    }

    // ---- DiamondSearch(length) for length = param, param / 2, ... 1
    if (pc == P_LG_START)
    {
      if (rs_i <= 0)
      {
        rs_pc = P_DONE;
        continue;
      }
      rs_len = rs_i;
      rs_dir = 15;
      rs_pc = P_LG_LOOP;
      continue;
    }
    if (pc == P_LG_LOOP)
    {
      if (rs_dir <= 0)
      {
        rs_i /= 2;
        rs_pc = P_LG_START;
        continue;
      }
      rs_dx = ck_bx;
      rs_dy = ck_by;
      rs_last = rs_dir;
      rs_dir = 0;
      const int last = rs_last, dx = rs_dx, dy = rs_dy, l = rs_len;
      if ((last & 1) != 0) cand(n++, dx + l, dy, 1);
      if ((last & 2) != 0) cand(n++, dx - l, dy, 2);
      if ((last & 4) != 0) cand(n++, dx, dy + l, 4);
      if ((last & 8) != 0) cand(n++, dx, dy - l, 8);
      rs_pc = P_LG_AFTER1;
      if (n != 0)
      {
        mode = M_CHECK2;
        return true;
      }
      continue;
    }
    if (pc == P_LG_AFTER1)
    {
      const int l = rs_len;
      if (rs_dir != 0)
      {
        rs_last = rs_dir;
        const int dx = ck_bx, dy = ck_by;
        if ((rs_last & 3) != 0)
        {
          cand(0, dx, dy + l, 4);
          cand(1, dx, dy - l, 8);
        }
        else
        {
          cand(0, dx + l, dy, 1);
          cand(1, dx - l, dy, 2);
        }
        n = 2;
      }
      else
      {
        const int dx = rs_dx, dy = rs_dy, last = rs_last;
#define DC(ax, ay, v) cand(n++, dx + (ax) * l, dy + (ay) * l, (v))
        if (last == 1) { DC(1, 1, 1 + 4); DC(1, -1, 1 + 8); }
        else if (last == 2) { DC(-1, 1, 2 + 4); DC(-1, -1, 2 + 8); }
        else if (last == 4) { DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); }
        else if (last == 8) { DC(1, -1, 1 + 8); DC(-1, -1, 2 + 8); }
        else if (last == 5) { DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); DC(1, -1, 1 + 8); }
        else if (last == 6) { DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); DC(-1, -1, 2 + 8); }
        else if (last == 9) { DC(1, 1, 1 + 4); DC(-1, -1, 2 + 8); DC(1, -1, 1 + 8); }
        else if (last == 10) { DC(-1, -1, 2 + 8); DC(-1, 1, 2 + 4); DC(1, -1, 1 + 8); }
        else { DC(1, 1, 1 + 4); DC(-1, 1, 2 + 4); DC(1, -1, 1 + 8); DC(-1, -1, 2 + 8); }
#undef DC
      }
      mode = M_CHECK2;
      rs_pc = P_LG_LOOP;
      return true;
    }

    // ---- Hex2Search(rs_hrange), then rs_hnext
    if (pc == P_HX_INIT)
    {
      rs_dir = -2;
      rs_bmx = ck_bx;
      rs_bmy = ck_by;
      rs_pc = rs_hrange > 1 ? P_HX_A : P_HX_END;
      continue;
    }
    if (pc == P_HX_A)
    {
      const int bmx = rs_bmx, bmy = rs_bmy;
      cand(0, bmx - 2, bmy, 0);
      cand(1, bmx - 1, bmy + 2, 1);
      cand(2, bmx + 1, bmy + 2, 2);
      cand(3, bmx + 2, bmy, 3);
      cand(4, bmx + 1, bmy - 2, 4);
      cand(5, bmx - 1, bmy - 2, 5);
      mode = M_CHECKDIR;
      n = 6;
      rs_pc = P_HX_A1;
      return true;
    }
    if (pc == P_HX_A1)
    {
      if (rs_dir != -2)
      {
        rs_bmx += HEX2X[rs_dir + 1];
        rs_bmy += HEX2Y[rs_dir + 1];
        rs_cnt = 1;
        rs_pc = P_HX_LOOP;
      }
      else
        rs_pc = P_HX_SET;
      continue;
    }
    if (pc == P_HX_LOOP)
    {
      if (rs_cnt < rs_hrange / 2 && ck_ok(rs_bmx, rs_bmy))
      {
        const int odir = MOD6M1[rs_dir + 1], bmx = rs_bmx, bmy = rs_bmy;
        rs_dir = -2;
        cand(0, bmx + HEX2X[odir], bmy + HEX2Y[odir], odir - 1);
        cand(1, bmx + HEX2X[odir + 1], bmy + HEX2Y[odir + 1], odir);
        cand(2, bmx + HEX2X[odir + 2], bmy + HEX2Y[odir + 2], odir + 1);
        mode = M_CHECKDIR;
        n = 3;
        rs_pc = P_HX_L1;
        return true;
      }
      rs_pc = P_HX_SET;
      continue;
    }
    if (pc == P_HX_L1)
    {
      if (rs_dir == -2)
        rs_pc = P_HX_SET;
      else
      {
        rs_bmx += HEX2X[rs_dir + 1];
        rs_bmy += HEX2Y[rs_dir + 1];
        rs_cnt++;
        rs_pc = P_HX_LOOP;
      }
      continue;
    }
    if (pc == P_HX_SET)
    {
      ck_bx = rs_bmx;
      ck_by = rs_bmy;
      rs_pc = P_HX_END;
      continue;
    }
    if (pc == P_HX_END)
    {
      start_list(LK_RING, 1, 1, rs_bmx, rs_bmy, rs_hnext);
      continue;
    }

    // ---- UMHSearch(param, bx, by): the cross and grid list around the start, then Hex2Search(param)
    if (pc == P_UMH)
    {
      rs_hrange = rs_param;
      rs_hnext = P_DONE;
      start_list(LK_UMH, rs_param, 0, ck_bx, ck_by, P_HX_INIT);
      continue;
    }

    // ---- the horizontal / vertical searches: one batch
    if (pc == P_HV)
    {
      const int mvx = ck_bx, mvy = ck_by;
      for (int i = 1; i <= rs_param; i++)
      {
        if (rs_type == S_HORIZONTAL)
        {
          cand(n++, mvx - i, mvy, 0);
          cand(n++, mvx + i, mvy, 0);
        }
        else
        {
          cand(n++, mvx, mvy - i, 0);
          cand(n++, mvx, mvy + i, 0);
        }
      }
      rs_pc = P_DONE;
      return true;
    }

    // ---- the wide search for a bad vector (k_wave's wave_bad_search): UMHSearch(badrange * pel) around (0, 0) for
    // badrange > 0, ExpandingSearch rings around (0, 0) until a good vector is found for badrange < 0, then the small
    // sub-pel rings around the result
    if (pc == P_BD_INIT)
    {
      rs_found = ck_bsad;
      const int range = BAD_RANGE * E_PEL;
      if (BAD_RANGE > 0)
      {
        rs_hrange = range;
        rs_hnext = P_BD_SMALL0;
        start_list(LK_UMH, range, 0, 0, 0, P_HX_INIT);
      }
      else if (BAD_RANGE < 0)
      {
        rs_i = 1;
        rs_pc = P_BD_EXL;
      }
      else
        rs_pc = P_BD_SMALL0;
      continue;
    }
    if (pc == P_BD_EXL)
    {
      if (rs_i < -BAD_RANGE * E_PEL)
        start_list(LK_RING, rs_i, E_PEL, 0, 0, P_BD_EXL2);
      else
        rs_pc = P_BD_SMALL0;
      continue;
    }
    if (pc == P_BD_EXL2)
    {
      if (ck_bsad < rs_found / 4)
        rs_pc = P_BD_SMALL0; // a rather good vector is found
      else
      {
        rs_i += E_PEL;
        rs_pc = P_BD_EXL;
      }
      continue;
    }
    if (pc == P_BD_SMALL0)
    {
      rs_smx = ck_bx;
      rs_smy = ck_by;
      rs_i = 1;
      rs_pc = P_BD_SMALL;
      continue;
    }
    if (pc == P_BD_SMALL)
    {
      if (rs_i < E_PEL)
        start_list(LK_RING, rs_i, 1, rs_smx, rs_smy, P_BD_SMALL2);
      else
        rs_pc = P_DONE;
      continue;
    }
    if (pc == P_BD_SMALL2)
    {
      rs_i++;
      rs_pc = P_BD_SMALL;
      continue;
    }
    rs_pc = P_DONE;
  }
  return false;
}

// PlaneOfBlocks::Refine (tryMany off) of the given type, or T_BAD: the whole search with one evaluation site
void refine_run(int type, int param)
{
  refine_init(type, param);
  int mode, n;
  while (next_batch(mode, n))
    run_batch(mode, n, rs_dir);
}
