// Level arguments of the MAnalyse search shaders (ana_search, ana_wave; vk_analyse.cpp LevelArgs, same layout) and the
// search of one block (PlaneOfBlocks::PseudoEPZSearch up to the bad-vector handling), shared by both kernels.

struct Args
{
  uint64_t ref[3][16]; // sub-pel planes of the reference (pel * pel at level 0, MVPlane order), the rest 0
  uint64_t src[3];
  uint64_t vec;        // (vx, vy, sad) triples of this level (interpolated predictors on entry, results on exit)
  uint64_t glob;       // doubled global MV estimate of this level (x, y)
  uint64_t state;      // ints: [0] bad flag, [1] abort flag, [2] row ticket, [3 ..] row progress, then bad blocks per row
  uint64_t nlam;       // nLambda per block (ana_interp), not the smallest level
  int ew[3], hpad[3], vpad[3];
  int n_blk_x, n_blk_y, pel, log_pel, log_scale, smallest;
  int bs_x, bs_y, ov_x, ov_y, cw, ch, log_x_uv, log_y_uv;
  int ext_w, ext_h, lambda_level, lsad, pnew, pzero, pglobal, bits, chroma_scale, radius;
  int meander, n_slices, search_type, try_many, bad_sad, bad_range, spin_max, pad_;
  int chunk_rows, chunk, pad2_, pad3_; // ana_search: this dispatch searches rows [beg + chunk * chunk_rows, + chunk_rows) of every slice
};
ARGS(Args)

#define ST_BAD 0
#define ST_ABORT 1
#define ST_TICKET 2
#define ST_PROG 3

#define E_REF(p, s) A.ref[p][s]
#define E_SRC(p) A.src[p]
#define E_EW(p) A.ew[p]
#define E_PEL A.pel
#define E_LP A.log_pel
#define E_BSX A.bs_x
#define E_BSY A.bs_y
#define E_CW A.cw
#define E_CH A.ch
#define E_LXUV A.log_x_uv
#define E_LYUV A.log_y_uv
#define E_CSCALE A.chroma_scale
#define E_BITS A.bits
#define E_PNEW A.pnew

#include "ana_common.glsl"

// neighbours of the block, gathered by invocation 0 (s_nb) and read by everybody after a barrier:
// [0..2] own (interpolated) vector, [3..5] up, [6..8] p3 (bottom-right / up-right), [9..11] left,
// [12] use up, [13] use p3, [14] use left
shared int s_nb[16];

// Position, search range and source block of block (blkx, blky): sets e_*, ck_d*, copies the source block to s_src
// (published by the next barrier). Called by every invocation before the neighbours are gathered, so that the source
// loads are in flight while invocation 0 waits for the row above.
void block_setup(int blkx, int blky)
{
  const int step_x = A.bs_x - A.ov_x, step_y = A.bs_y - A.ov_y;
  const int h_scaled = A.hpad[0] >> A.log_scale, v_scaled = A.vpad[0] >> A.log_scale;
  e_x0 = A.hpad[0] + blkx * step_x;
  e_y0 = A.vpad[0] + blky * step_y;
  e_x1 = A.hpad[1] + blkx * (step_x >> A.log_x_uv);
  e_y1 = A.vpad[1] + blky * (step_y >> A.log_y_uv);
  ck_dxmax = A.pel * (A.ext_w - e_x0 - A.bs_x - A.hpad[0] + h_scaled);
  ck_dymax = A.pel * (A.ext_h - e_y0 - A.bs_y - A.vpad[0] + v_scaled);
  ck_dxmin = -A.pel * (e_x0 - A.hpad[0] + h_scaled);
  ck_dymin = -A.pel * (e_y0 - A.vpad[0] + v_scaled);
  load_src_block();
}

// The search of block (blkx, blky) after block_setup, with the neighbours in s_nb (published): result in ck_bx / ck_by /
// ck_bsad / ck_min, the predictor in ck_px / ck_py, nLambda in ck_nlam. Called by every invocation.
void search_block(int blkx, int blky, bool is_top)
{
  // predictors (FetchPredictors)
  const ivec3 own_c = clip_mv(s_nb[0], s_nb[1], s_nb[2]);
  const ivec3 zero_shifted = clip_mv(0, 0, 0);
  ivec3 p1 = zero_shifted, p2 = zero_shifted, p3 = zero_shifted, p0;
  if (s_nb[14] != 0)
    p1 = clip_mv(s_nb[9], s_nb[10], s_nb[11]);
  if (s_nb[12] != 0)
    p2 = clip_mv(s_nb[3], s_nb[4], s_nb[5]);
  if (s_nb[13] != 0)
    p3 = clip_mv(s_nb[6], s_nb[7], s_nb[8]);
  if (!is_top)
  {
    p0.x = median3(p1.x, p2.x, p3.x);
    p0.y = median3(p1.y, p2.y, p3.y);
    p0.z = max(p1.z, max(p2.z, p3.z));
  }
  else
    p0 = p1;
  const ivec3 pred = A.smallest != 0 ? p0 : own_c;
  ck_px = pred.x;
  ck_py = pred.y;
  // not the smallest level: the predictor is the block's own interpolated vector (clipping changes x / y only), so
  // nLambda was computed with the interpolation (off the serial path of the search)
  ck_nlam = A.smallest != 0 ? calc_nlambda(A.lambda_level, A.lsad, pred.z, is_top) : PI32(A.nlam).v[blky * A.n_blk_x + blkx];
  const ivec3 glob = clip_mv(PI32(A.glob).v[0] * A.pel, PI32(A.glob).v[1] * A.pel, 0);

  // phase A: zero, global, predictor, predictors[0..3] (the first three unconditionally, as the CPU)
  clear_sums(7);
  if (ana_tid() == 0)
  {
    s_xs[e_bo + 0] = 0; s_ys[e_bo + 0] = 0;
    s_xs[e_bo + 1] = glob.x; s_ys[e_bo + 1] = glob.y;
    s_xs[e_bo + 2] = pred.x; s_ys[e_bo + 2] = pred.y;
    s_xs[e_bo + 3] = p0.x; s_ys[e_bo + 3] = p0.y;
    s_xs[e_bo + 4] = p1.x; s_ys[e_bo + 4] = p1.y;
    s_xs[e_bo + 5] = p2.x; s_ys[e_bo + 5] = p2.y;
    s_xs[e_bo + 6] = p3.x; s_ys[e_bo + 6] = p3.y;
  }
  memoryBarrierShared();
  barrier();
  eval_batch(7, false);
  int sl7[7], sc7[7];
  for (int i = 0; i < 7; i++)
  {
    sl7[i] = s_sl[e_bo + i];
    sc7[i] = cand_sc(i);
  }
  barrier(); // everybody has the phase A sums before a refine batch overwrites them

  {
    const int sad = sl7[0] + sc7[0];
    ck_bx = 0;
    ck_by = 0;
    ck_bsad = sad;
    ck_min = sad + pen(A.pzero, sad);
    const int sadg = sl7[1] + sc7[1];
    const int costg = sadg + pen(A.pglobal, sadg);
    if (costg < ck_min)
    {
      ck_bx = glob.x;
      ck_by = glob.y;
      ck_bsad = sadg;
      ck_min = costg;
    }
    const int sadp = sl7[2] + sc7[2];
    if (sadp < ck_min)
    {
      ck_bx = pred.x;
      ck_by = pred.y;
      ck_bsad = sadp;
      ck_min = sadp;
    }
    const ivec3 pv[4] = ivec3[4](p0, p1, p2, p3);
    for (int i = 0; i < 4; i++) // CheckMV0
    {
      const int cx = pv[i].x, cy = pv[i].y;
      if (!ck_ok(cx, cy))
        continue;
      int cost = ck_lambda_cost(cx, cy);
      if (cost >= ck_min)
        continue;
      cost += sl7[3 + i];
      if (cost >= ck_min)
        continue;
      cost += sc7[3 + i];
      if (cost >= ck_min)
        continue;
      ck_bx = cx;
      ck_by = cy;
      ck_bsad = sl7[3 + i] + sc7[3 + i];
      ck_min = cost;
    }
  }

  // PseudoEPZSearch with tryMany: a Refine around zero, global, the predictor and each of predictors[0..3], every one
  // from its own start state, then the lowest saved cost (the first one on ties). Without it one Refine from the phase A
  // result. (One refine_run call site: the evaluation is instantiated once.)
  const int verybig = (3 * A.bs_x * A.bs_y) << A.bits; // PlaneOfBlocks::verybigSAD
  const ivec3 pv[4] = ivec3[4](p0, p1, p2, p3);
  const int nm = A.try_many != 0 ? 7 : 1;
  int mx[7], my[7], ms[7], mc[7];
  for (int m = 0; m < nm; m++)
  {
    if (A.try_many != 0)
    {
      if (m == 0)
      {
        const int sad = sl7[0] + sc7[0];
        ck_bx = 0;
        ck_by = 0;
        ck_bsad = sad;
        ck_min = sad + pen(A.pzero, sad);
      }
      else if (m == 1)
      {
        // "cost < nMinCost || tryMany": the global vector is taken whatever its cost
        const int sad = sl7[1] + sc7[1];
        ck_bx = glob.x;
        ck_by = glob.y;
        ck_bsad = sad;
        ck_min = sad + pen(A.pglobal, sad);
      }
      else if (m == 2)
      {
        const int sad = sl7[2] + sc7[2];
        ck_bx = pred.x;
        ck_by = pred.y;
        ck_bsad = sad;
        ck_min = sad;
      }
      else
      {
        // nMinCost = verybigSAD + 1, then CheckMV0; bestMV stays the previous Refine's result when the predictor is out
        // of range or not cheaper
        const int i = m - 3, cx = pv[i].x, cy = pv[i].y;
        ck_min = verybig + 1;
        if (ck_ok(cx, cy))
        {
          int cost = ck_lambda_cost(cx, cy);
          if (cost < ck_min)
          {
            cost += sl7[3 + i];
            if (cost < ck_min)
            {
              cost += sc7[3 + i];
              if (cost < ck_min)
              {
                ck_bx = cx;
                ck_by = cy;
                ck_bsad = sl7[3 + i] + sc7[3 + i];
                ck_min = cost;
              }
            }
          }
        }
      }
    }
    refine_run(A.search_type, A.radius);
    mx[m] = ck_bx;
    my[m] = ck_by;
    ms[m] = ck_bsad;
    mc[m] = ck_min;
  }
  if (A.try_many != 0)
  {
    int sel_cost = verybig + 1;
    for (int m = 0; m < 7; m++)
      if (mc[m] < sel_cost)
      {
        ck_bx = mx[m];
        ck_by = my[m];
        ck_bsad = ms[m];
        sel_cost = mc[m];
      }
    ck_min = sel_cost;
  }
}
