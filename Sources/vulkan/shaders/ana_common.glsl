// Shared code of the MAnalyse / MRecalculate search shaders (ana_search, ana_wave, recalc): port of
// Sources/cuda/mvcuda_refine.cuh (PlaneOfBlocks::Refine and the search routines over a candidate checker) and of the
// candidate evaluation / fold of mvcuda_analyse.cu (AnaChecker::run_batch), plus calc_nlambda in exact integer arithmetic.
//
// Execution model: one workgroup of ANA_WG invocations searches one block at a time. Every invocation runs the search
// control flow with identical scalar state (ck_*), so every branch and loop around a barrier() is uniform; the candidate
// lists live in shared memory (written by invocation 0 or in parallel), the SADs of a batch are spread over the whole
// workgroup (eval_batch), and every invocation then folds the batch in the CPU's order (fold_batch). No subgroup
// operations: the partial sums go through shared memory.
//
// The including shader defines before the #include:
//   ANA_WG                    local size (power of two)
//   E_REF(p, s)               device address of the reference sub-plane s of plane p (current level)
//   E_SRC(p)                  device address of the source plane p (sub-plane 0)
//   E_EW(p)                   samples per row of plane p
//   E_PEL, E_LP               pel and log2(pel) of the level
//   E_BSX, E_BSY, E_CW, E_CH  luma block size, chroma block size (E_CW = E_CH = 0: luma only)
//   E_LXUV, E_LYUV            log2 of the chroma subsampling
//   E_CSCALE, E_BITS, E_PNEW  chroma SAD scale, bits per sample, penaltyNew
// and sets the block position e_x0 / e_y0 (luma), e_x1 / e_y1 (chroma) before evaluating candidates.

#define S_ONETIME 1
#define S_NSTEP 2
#define S_LOGARITHMIC 4
#define S_EXHAUSTIVE 8
#define S_HEX2 16
#define S_UMH 32
#define S_HORIZONTAL 64
#define S_VERTICAL 128

#define M_CHECK 0
#define M_CHECK2 1
#define M_CHECKDIR 2

#define ANA_MAXB 256 // candidates of one batch in shared memory (mvcuda_refine.cuh MAX_BATCH is 160; larger exhaustive
                     // and bad-search batches only save barriers, the sequential fold is the same)

// candidate lists, double-buffered: the list of the next batch may be written while other invocations still fold
// the previous one (e_bo: offset of the current buffer, identical in all invocations)
shared int s_xs[2 * ANA_MAXB], s_ys[2 * ANA_MAXB], s_vals[2 * ANA_MAXB], s_sl[2 * ANA_MAXB], s_sc[2 * ANA_MAXB];
int e_bo = 0;

// block position of the current block (padded planes, full samples)
int e_x0, e_y0, e_x1, e_y1;
// the checker state (workarea.bestMV, nMinCost, the search range, the predictor, nLambda)
int ck_bx, ck_by, ck_min, ck_bsad;
int ck_dxmin, ck_dxmax, ck_dymin, ck_dymax, ck_px, ck_py, ck_nlam;

int ana_tid() { return int(gl_LocalInvocationIndex); }

#include "ana_nlambda.glsl"

// ---- small helpers -------------------------------------------------------------------------------------------------
int scale_chroma(int sad, int scale)
{
  if (scale == 0)
    return sad;
  return scale > 0 ? sad >> scale : sad << (-scale);
}

int median3(int a, int b, int c)
{
  if (a < b)
  {
    if (b < c) return b;
    else if (a < c) return c;
    else return a;
  }
  else
  {
    if (a < c) return a;
    else if (b < c) return c;
    else return b;
  }
}

ivec3 clip_mv(int x, int y, int sad)
{
  return ivec3(x < ck_dxmin ? ck_dxmin : (x >= ck_dxmax ? ck_dxmax - 1 : x),
               y < ck_dymin ? ck_dymin : (y >= ck_dymax ? ck_dymax - 1 : y), sad);
}

bool ck_ok(int cx, int cy) { return cx >= ck_dxmin && cy >= ck_dymin && cx < ck_dxmax && cy < ck_dymax; }

// vector cost of candidate (cx, cy) against the predictor (the first part of CheckMV's cost chain)
int ck_lambda_cost(int cx, int cy)
{
  const int dxp = ck_px - cx, dyp = ck_py - cy;
  const int dist = int(uint(dxp * dxp) + uint(dyp * dyp));
  return int(uint(ck_nlam) * uint(dist)) >> (16 - E_BITS);
}

int pen(int pnew, int sad) { return int((int64_t(pnew) * int64_t(sad)) >> 8); }

// candidate i of the current list (invocation 0 writes; the barrier at the start of run_batch publishes it)
void cand(int i, int x, int y, int v)
{
  if (ana_tid() == 0)
  {
    s_xs[e_bo + i] = x;
    s_ys[e_bo + i] = y;
    s_vals[e_bo + i] = v;
  }
}

// ---- candidate evaluation --------------------------------------------------------------------------------------------
// Specialization constants of the including kernel: the luma rows and the rows of one chroma block when they are known
// at pipeline creation (0: run-time loops). With known counts the row loops unroll and every reference row of a column
// is loaded before the sums (the loads go out together: the evaluation is latency bound).
layout(constant_id = 0) const int SC_BSY = 0;
layout(constant_id = 1) const int SC_CHH = 0;

#define ANA_SRC_MAX 3072 // source block in shared memory (32x32 luma + two 32x32 chroma blocks)
shared int s_src[ANA_SRC_MAX];
bool e_src_sh;  // the source block of the current block is in s_src (luma, then U, then V)
int e_gw;       // invocations per candidate: a power of two >= the block width (column = invocation), <= ANA_WG

// lanes per candidate for the block geometry
int pick_gw(int bsx)
{
  int gw = 4;
  while (gw < bsx && gw < ANA_WG)
    gw *= 2;
  return gw;
}

// copies the source block at (e_x0, e_y0) / (e_x1, e_y1) into s_src when it fits (all invocations; the caller's next
// barrier publishes it) and sets e_src_sh / e_gw
void load_src_block()
{
  const int bsx = E_BSX, bsy = E_BSY, cw = E_CW, ch = E_CH;
  e_gw = pick_gw(bsx);
  e_src_sh = bsx * bsy + 2 * cw * ch <= ANA_SRC_MAX;
  if (!e_src_sh)
    return;
  const int tid = ana_tid();
  const uint64_t sy = E_SRC(0);
  const int ew = E_EW(0);
  for (int i = tid; i < bsx * bsy; i += ANA_WG)
  {
    const int r = i / bsx, c = i - r * bsx;
    s_src[i] = ldpix(sy, uint((e_y0 + r) * ew + e_x0 + c));
  }
  if (cw > 0)
  {
    const uint64_t su = E_SRC(1), sv = E_SRC(2);
    const int ewu = E_EW(1), ewv = E_EW(2), o = bsx * bsy, nc = cw * ch;
    for (int i = tid; i < nc; i += ANA_WG)
    {
      const int r = i / cw, c = i - r * cw;
      s_src[o + i] = ldpix(su, uint((e_y1 + r) * ewu + e_x1 + c));
      s_src[o + nc + i] = ldpix(sv, uint((e_y1 + r) * ewv + e_x1 + c));
    }
  }
}

// source sample (row, col) of plane p (0 = luma) of the current block
int src_px(int p, int row, int col)
{
  if (e_src_sh)
  {
    const int o = p == 0 ? 0 : E_BSX * E_BSY + (p - 1) * E_CW * E_CH;
    return s_src[o + row * (p == 0 ? E_BSX : E_CW) + col];
  }
  return p == 0 ? ldpix(E_SRC(0), uint((e_y0 + row) * E_EW(0) + e_x0 + col))
                : ldpix(E_SRC(p), uint((e_y1 + row) * E_EW(p) + e_x1 + col));
}

// SAD of one column of the luma block (col_sad_l) / of the chroma block of plane p (col_sad_c): reference column at
// sample index r0 (pitch ew) of the plane at rb
int col_sad_l(int col, uint64_t rb, int r0, int ew)
{
  int s = 0;
  if (SC_BSY > 0 && e_src_sh)
  {
    const int w = E_BSX;
    [[unroll]] for (int row = 0; row < SC_BSY; row++)
      s += abs(s_src[row * w + col] - ldpix(rb, uint(r0 + row * ew)));
  }
  else
  {
    const int rows = E_BSY;
    for (int row = 0; row < rows; row++)
      s += abs(src_px(0, row, col) - ldpix(rb, uint(r0 + row * ew)));
  }
  return s;
}

int col_sad_c(int p, int col, uint64_t rb, int r0, int ew)
{
  int s = 0;
  if (SC_CHH > 0 && e_src_sh)
  {
    const int o = E_BSX * E_BSY + (p - 1) * E_CW * E_CH, w = E_CW;
    [[unroll]] for (int row = 0; row < SC_CHH; row++)
      s += abs(s_src[o + row * w + col] - ldpix(rb, uint(r0 + row * ew)));
  }
  else
  {
    const int rows = E_CH;
    for (int row = 0; row < rows; row++)
      s += abs(src_px(p, row, col) - ldpix(rb, uint(r0 + row * ew)));
  }
  return s;
}

// SADs (luma, raw chroma) of the candidates 0 .. n-1 of the list into s_sl / s_sc (cleared by clear_sums before the
// list was published): e_gw invocations per candidate, invocation = column (luma; chroma U in [0, cw) and V in [cw, 2 cw)
// when 2 cw <= e_gw, else both planes per column), the column sums added with shared atomics. range_test: candidates
// outside the search range are skipped (their entries are never read). Called by every invocation (n uniform); ends with
// the results published by a barrier.
void eval_batch(int n, bool range_test)
{
  const int tid = ana_tid();
  const int GW = e_gw, cpr = ANA_WG / GW;
  const int bsx = E_BSX, bsy = E_BSY, cw = E_CW, ch = E_CH;
  const int pel = E_PEL, lp = E_LP, pm = pel - 1;
  const bool pair = 2 * cw <= GW;
  for (int base = 0; base < n; base += cpr)
  {
    const int c = base + tid / GW, col = tid & (GW - 1);
    int suml = 0, sumc = 0;
    if (c < n)
    {
      const int cx = s_xs[e_bo + c], cy = s_ys[e_bo + c];
      if (!range_test || ck_ok(cx, cy))
      {
        const int xp = e_x0 * pel + cx, yp = e_y0 * pel + cy;
        const int sub = (xp & pm) | ((yp & pm) << lp);
        const int rx = xp >> lp, ry = yp >> lp;
        const uint64_t rb = E_REF(0, sub);
        const int ew = E_EW(0);
        for (int cc = col; cc < bsx; cc += GW)
          suml += col_sad_l(cc, rb, ry * ew + rx + cc, ew);
        if (cw > 0)
        {
          const int xpc = e_x1 * pel + (cx >> E_LXUV), ypc = e_y1 * pel + (cy >> E_LYUV);
          const int subc = (xpc & pm) | ((ypc & pm) << lp);
          const int rxc = xpc >> lp, ryc = ypc >> lp;
          if (pair)
          {
            if (col < 2 * cw)
            {
              const int p = col < cw ? 1 : 2, cc = col < cw ? col : col - cw;
              const int ewc = E_EW(p);
              sumc += col_sad_c(p, cc, E_REF(p, subc), ryc * ewc + rxc + cc, ewc);
            }
          }
          else
          {
            for (int cc = col; cc < cw; cc += GW)
              for (int p = 1; p < 3; p++)
              {
                const int ewc = E_EW(p);
                sumc += col_sad_c(p, cc, E_REF(p, subc), ryc * ewc + rxc + cc, ewc);
              }
          }
        }
      }
    }
    // column sums straight into the candidate's totals (cleared before the list was published)
    if (c < n)
    {
      if (suml != 0)
        atomicAdd(s_sl[e_bo + c], suml);
      if (sumc != 0)
        atomicAdd(s_sc[e_bo + c], sumc);
    }
  }
  memoryBarrierShared();
  barrier();
}

// the scaled chroma SAD of candidate c of the evaluated batch (s_sc holds the raw sum)
int cand_sc(int c) { return scale_chroma(s_sc[e_bo + c], E_CSCALE); }

// clears the sums of candidates 0 .. n-1 (before the barrier that publishes their list)
void clear_sums(int n)
{
  for (int k = ana_tid(); k < n; k += ANA_WG)
  {
    s_sl[e_bo + k] = 0;
    s_sc[e_bo + k] = 0;
  }
}

// The CPU's sequence of CheckMV (M_CHECK) / CheckMV2 (M_CHECK2) / CheckMVdir (M_CHECKDIR) over the evaluated batch, in
// order: a candidate is accepted when its cost chain (vector cost, luma + penaltyNew, chroma + penaltyNew) stays strictly
// below the running minimum. Every invocation runs it with identical values.
void fold_batch(int mode, int n, inout int dir)
{
  for (int c = 0; c < n; c++)
  {
    const int cx = s_xs[e_bo + c], cy = s_ys[e_bo + c];
    if (!ck_ok(cx, cy))
      continue;
    int cost = ck_lambda_cost(cx, cy);
    if (cost >= ck_min)
      continue;
    const int sl = s_sl[e_bo + c], sc = cand_sc(c);
    cost += sl + pen(E_PNEW, sl);
    if (cost >= ck_min)
      continue;
    cost += sc + pen(E_PNEW, sc);
    if (cost >= ck_min)
      continue;
    if (mode != M_CHECKDIR)
    {
      ck_bx = cx;
      ck_by = cy;
    }
    ck_min = cost;
    ck_bsad = sl + sc;
    if (mode != M_CHECK)
      dir = s_vals[e_bo + c];
  }
}

// one batch of n candidates (list already written): evaluate, fold
void run_batch(int mode, int n, inout int dir)
{
  clear_sums(n);
  memoryBarrierShared();
  barrier(); // the list
  eval_batch(n, true);
  fold_batch(mode, n, dir);
  e_bo = ANA_MAXB - e_bo; // the next list goes to the other buffer (no barrier needed before writing it)
}

// ---- PlaneOfBlocks search routines (mvcuda_refine.cuh) ---------------------------------------------------------------
// candidate k (0 .. 8r - 1) of ExpandingSearch(r, 1, 0, 0) in CPU order
ivec2 expanding_point(int r, int k)
{
  const int side = 2 * (2 * r - 1);
  if (k < side)
    return ivec2(-r + 1 + (k >> 1), (k & 1) != 0 ? r : -r);
  if (k < 2 * side)
  {
    k -= side;
    return ivec2((k & 1) != 0 ? r : -r, -r + 1 + (k >> 1));
  }
  k -= 2 * side;
  return ivec2((k & 2) != 0 ? r : -r, (k & 1) != 0 ? r : -r);
}

#include "ana_refine.glsl"
