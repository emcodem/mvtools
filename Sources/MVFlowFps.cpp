// Pixels flow motion interpolation function to change FPS
// Copyright(c)2005 A.G.Balakhnin aka Fizick

// See legal notice in Copying.txt for more information

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .

#include <cstdio>
#include <cstring>
#include "ClipFnc.h"
#include "commonfunctions.h"
#include "MaskFun.h"
#include "MVFinest.h"
#include "MVFlowFps.h"
#include "MVSuper.h"
#include "profile.h"
#include "GoldenDump.h"
#include <vector>

#ifdef MV_GOLDEN_DUMP
// golden-data dump of one vector field: (vx, vy, sad) int triples per block
static void golden_dump_vectors(const char *name, MVClip &clip, int blk_x, int blk_y)
{
  std::vector<int> v(size_t(blk_x) * blk_y * 3);
  for (int i = 0; i < blk_x * blk_y; i++)
  {
    const VECTOR m = clip.GetBlock(0, i).GetMV();
    v[3 * i] = m.x;
    v[3 * i + 1] = m.y;
    v[3 * i + 2] = (int)m.sad;
  }
  golden::write_bin(name, v.data(), v.size() * sizeof(int));
}
#endif
#include "SuperParams64Bits.h"
#include "info.h"
#include "GpuBackend.h"


MVFlowFps::MVFlowFps(PClip _child, PClip super, PClip _mvbw, PClip _mvfw, unsigned int _num, unsigned int _den, int _maskmode, double _ml,
  bool _blend, sad_t nSCD1, int nSCD2, bool _isse, bool _planar, int _optDebug, bool mt_flag, bool gpu_flag, IScriptEnvironment* env) :
  GenericVideoFilter(_child),
  MVFilter(_mvfw, "MFlowFps", env, 1, 0),
  mvClipB(_mvbw, nSCD1, nSCD2, env, 1, 0),
  mvClipF(_mvfw, nSCD1, nSCD2, env, 1, 0),
  optDebug(_optDebug),
  _mt_flag(mt_flag),
  _gpu_ok(false)
{
  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  static int id = 0; _instance_id = id++;
  reentrancy_check = false;
  // _RPT1(0, "MVFlowFps.Create id=%d\n", _instance_id);

  if (!vi.IsYUV() && !vi.IsYUVA() && !vi.IsPlanarRGB() && !vi.IsPlanarRGBA())
    env->ThrowError("MFlowFps: Clip must be Y, YUV or YUY2 or planar RGB/RGBA");

  numeratorOld = vi.fps_numerator;
  denominatorOld = vi.fps_denominator;

  if (_num != 0 && _den != 0)
  {
    numerator = _num;
    denominator = _den;
  }
  else if (numeratorOld < (1 << 30))
  {
    numerator = (numeratorOld << 1); // double fps by default
    denominator = denominatorOld;
  }
  else // very big numerator
  {
    numerator = numeratorOld;
    denominator = (denominatorOld >> 1);// double fps by default
  }

  //  safe for big numbers since v2.1
  fa = int64_t(denominator)*int64_t(numeratorOld);
  fb = int64_t(numerator)*int64_t(denominatorOld);
  int64_t fgcd = gcd(fa, fb); // general common divisor
  fa /= fgcd;
  fb /= fgcd;

  vi.SetFPS(numerator, denominator);

  vi.num_frames = (int)(1 + int64_t(vi.num_frames - 1) * fb / fa);

  maskmode = _maskmode; // speed mode
  ml = _ml;
//   nIdx = _nIdx;
  cpuFlags = _isse ? env->GetCPUFlags() : 0;
  planar = _planar;
  blend = _blend;

  CheckSimilarity(mvClipB, "mvbw", env);
  CheckSimilarity(mvClipF, "mvfw", env);

  if (nWidth != vi.width || nHeight != vi.height)
    env->ThrowError("MFlowFps: inconsistent source and vector frame size");


  if (!((nWidth + nHPadding * 2) == super->GetVideoInfo().width && (nHeight + nVPadding * 2) <= super->GetVideoInfo().height))
    env->ThrowError("MFlowFps: inconsistent clips frame size!");

    // get parameters of prepared super clip - v2.0
  SuperParams64Bits params;
  memcpy(&params, &super->GetVideoInfo().num_audio_samples, 8);
  int nHeightS = params.nHeight;
  int nSuperHPad = params.nHPad;
  //int nSuperVPad = params.nVPad;
  int nSuperPel = params.nPel;
  //int nSuperModeYUV = params.nModeYUV;
  //int nSuperLevels = params.nLevels;
  int nSuperWidth = super->GetVideoInfo().width; // really super
  //int nSuperHeight = super->GetVideoInfo().height;

  if (!super->GetVideoInfo().IsSameColorspace(child->GetVideoInfo()))
    env->ThrowError("MFlowFPS: input and super clip format is different");

  if (nHeight != nHeightS
    || nWidth != nSuperWidth - nSuperHPad * 2
    || nPel != nSuperPel)
  {
    env->ThrowError("MFlowFps : wrong super frame clip");
  }

  const bool super_lite = MV_SUPER_IS_LITE(params); // MSuper(gpu=true): MVFinest completes it where the CPU path needs the real thing
  if (nPel == 1 && !super_lite)
  {
    finest = super; // v2.0.9.1
  }
  else
  {
    finest = new MVFinest(super, _isse, env, true);
    AVSValue cache_args[1] = { finest };
    finest = env->Invoke("InternalCache", AVSValue(cache_args, 1)).AsClip(); // add cache for speed
  }

  // actual format can differ from the base clip of vectors
  pixelsize_super = super->GetVideoInfo().ComponentSize();
  bits_per_pixel_super = super->GetVideoInfo().BitsPerComponent();
  pixelsize_super_shift = ilog2(pixelsize_super);

  planecount = vi.IsYUY2() ? 3 : std::min(vi.NumComponents(), 3);

  xRatioUVs[0] = xRatioUVs[1] = xRatioUVs[2] = 1;
  yRatioUVs[0] = yRatioUVs[1] = yRatioUVs[2] = 1;

  if (!vi.IsY() && !vi.IsRGB()) {
    xRatioUVs[1] = xRatioUVs[2] = vi.IsYUY2() ? 2 : (1 << vi.GetPlaneWidthSubsampling(PLANAR_U));
    yRatioUVs[1] = yRatioUVs[2] = vi.IsYUY2() ? 1 : (1 << vi.GetPlaneHeightSubsampling(PLANAR_U));
  }

  for (int i = 0; i < planecount; i++) {
    nLogxRatioUVs[i] = ilog2(xRatioUVs[i]);
    nLogyRatioUVs[i] = ilog2(yRatioUVs[i]);
  }

  // may be padded for full frame cover
  /*
  nBlkXP = (nBlkX*(nBlkSizeX - nOverlapX) + nOverlapX < nWidth) ? nBlkX + 1 : nBlkX;
  nBlkYP = (nBlkY*(nBlkSizeY - nOverlapY) + nOverlapY < nHeight) ? nBlkY + 1 : nBlkY;
  */
  // 2.7.27- sometimes +1 is not enough 
  nBlkXP = nBlkX;
  while (nBlkXP*(nBlkSizeX - nOverlapX) + nOverlapX < nWidth)
    nBlkXP++;
  nBlkYP = nBlkY;
  while (nBlkYP*(nBlkSizeY - nOverlapY) + nOverlapY < nHeight)
    nBlkYP++;

  nWidthP = nBlkXP*(nBlkSizeX - nOverlapX) + nOverlapX;
  nHeightP = nBlkYP*(nBlkSizeY - nOverlapY) + nOverlapY;

  nWidthPUV = nWidthP / xRatioUVs[1];
  nHeightPUV = nHeightP / yRatioUVs[1];
  nHeightUV = nHeight / yRatioUVs[1];
  nWidthUV = nWidth / xRatioUVs[1];

  nHPaddingUV = nHPadding / xRatioUVs[1];
  nVPaddingUV = nVPadding / yRatioUVs[1];

  VPitchY = AlignNumber(nWidthP, 16);
  VPitchUV = AlignNumber(nWidthPUV, 16);

  is444 = vi.Is444();
  isGrey = vi.IsY();
  isRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA(); // planar only
  needDistinctChroma = !is444 && !isGrey && !isRGB;

  // ---- GPU rendering: only for the configuration the GPU code reproduces bit-exactly (GPU_PLAN.md)
  if (gpu_flag)
  {
    const bool sharp_ok = (params.param & 0x80) && ((params.param >> 3) & 3) == 2 && !(params.param & 0x20);
    _gpu_ok = GpuBackend::instance().usable() && sharp_ok && (pixelsize_super == 1 || pixelsize_super == 2)
      && vi.IsPlanar() && !vi.IsYUY2() && !isGrey && !isRGB && planecount == 3 && maskmode == 2 && (nPel == 1 || nPel == 2)
      && (xRatioUVs[1] == 1 || xRatioUVs[1] == 2) && (yRatioUVs[1] == 1 || yRatioUVs[1] == 2);
    if (GpuBackend::log_enabled())
      std::fprintf(stderr, "MFlowFps gpu requested: %s\n", _gpu_ok ? "using the GPU" : "not supported here, using the CPU");
    if (_gpu_ok)
    {
      _super_clip = super;
      GpuBackend::instance().add_user();
      _producer = GpuBackend::instance().producer_register(&MVFlowFps::gpu_regen, this);
    }
  }

  // 2*: sizeof(short)
  VXFullYB = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  VYFullYB = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  if (needDistinctChroma) {
    VYFullUVB = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
    VXFullUVB = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
  }

  VXFullYF = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  VYFullYF = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  if (needDistinctChroma) {
    VXFullUVF = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
    VYFullUVF = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
  }

  VXSmallYB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  VYSmallYB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  if (needDistinctChroma) {
    VXSmallUVB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
    VYSmallUVB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  }

  VXSmallYF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  VYSmallYF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  if (needDistinctChroma) {
    VXSmallUVF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
    VYSmallUVF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  }

  VXFullYBB = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  VYFullYBB = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  if (needDistinctChroma) {
    VXFullUVBB = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
    VYFullUVBB = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
  }

  VXFullYFF = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  VYFullYFF = (short*)_aligned_malloc(2 * nHeightP*VPitchY + 128, 128);
  if (needDistinctChroma) {
    VXFullUVFF = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
    VYFullUVFF = (short*)_aligned_malloc(2 * nHeightPUV*VPitchUV + 128, 128);
  }

  VXSmallYBB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  VYSmallYBB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  if (needDistinctChroma) {
    VXSmallUVBB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
    VYSmallUVBB = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  }

  VXSmallYFF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  VYSmallYFF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  if (needDistinctChroma) {
    VXSmallUVFF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
    VYSmallUVFF = (short*)_aligned_malloc(2 * nBlkXP*nBlkYP + 128, 128);
  }

  // PF remark: masks are 8 bits
  MaskSmallB = (unsigned char*)_aligned_malloc(nBlkXP*nBlkYP + 128, 128);
  MaskFullYB = (unsigned char*)_aligned_malloc(nHeightP*VPitchY + 128, 128);
  if(needDistinctChroma)
    MaskFullUVB = (unsigned char*)_aligned_malloc(nHeightPUV*VPitchUV + 128, 128);

  MaskSmallF = (unsigned char*)_aligned_malloc(nBlkXP*nBlkYP + 128, 128);
  MaskFullYF = (unsigned char*)_aligned_malloc(nHeightP*VPitchY + 128, 128);
  if(needDistinctChroma)
    MaskFullUVF = (unsigned char*)_aligned_malloc(nHeightPUV*VPitchUV + 128, 128);

  SADMaskSmallB = (unsigned char*)_aligned_malloc(nBlkXP*nBlkYP + 128, 128);
  SADMaskSmallF = (unsigned char*)_aligned_malloc(nBlkXP*nBlkYP + 128, 128);

  upsizer = new SimpleResize(nWidthP, nHeightP, nBlkXP, nBlkYP, cpuFlags);
  if(needDistinctChroma)
    upsizerUV = new SimpleResize(nWidthPUV, nHeightPUV, nBlkXP, nBlkYP, cpuFlags);

  nleftLast = -1000;
  nrightLast = -1000;

  if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !planar)
  {
    DstPlanes = new YUY2Planes(nWidth, nHeight);
  }

}

MVFlowFps::~MVFlowFps()
{
  if (_producer)
    GpuBackend::instance().producer_unregister(_producer);
  if (_gpu_ok)
    GpuBackend::instance().remove_user();
  if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !planar)
  {
    delete DstPlanes;
  }

  delete upsizer;
  if(needDistinctChroma)
    delete upsizerUV;

  _aligned_free(MaskSmallB);
  _aligned_free(MaskFullYB);
  if(needDistinctChroma)
    _aligned_free(MaskFullUVB);
  _aligned_free(MaskSmallF);
  _aligned_free(MaskFullYF);
  if (needDistinctChroma)
    _aligned_free(MaskFullUVF);

  _aligned_free(VXFullYB);
  _aligned_free(VYFullYB);
  if (needDistinctChroma) {
    _aligned_free(VYFullUVB);
    _aligned_free(VXFullUVB);
  }

  _aligned_free(VXFullYF);
  _aligned_free(VYFullYF);
  if (needDistinctChroma) {
    _aligned_free(VXFullUVF);
    _aligned_free(VYFullUVF);
  }

  _aligned_free(VXSmallYB);
  _aligned_free(VYSmallYB);
  if (needDistinctChroma) {
    _aligned_free(VXSmallUVB);
    _aligned_free(VYSmallUVB);
  }

  _aligned_free(VXSmallYF);
  _aligned_free(VYSmallYF);
  if (needDistinctChroma) {
    _aligned_free(VXSmallUVF);
    _aligned_free(VYSmallUVF);
  }

  _aligned_free(VXFullYBB);
  _aligned_free(VYFullYBB);
  if (needDistinctChroma) {
    _aligned_free(VXFullUVBB);
    _aligned_free(VYFullUVBB);
  }

  _aligned_free(VXSmallYBB);
  _aligned_free(VYSmallYBB);
  if (needDistinctChroma) {
    _aligned_free(VXSmallUVBB);
    _aligned_free(VYSmallUVBB);
  }

  _aligned_free(VXFullYFF);
  _aligned_free(VYFullYFF);
  if (needDistinctChroma) {
    _aligned_free(VXFullUVFF);
    _aligned_free(VYFullUVFF);
  }


  _aligned_free(VXSmallYFF);
  _aligned_free(VYSmallYFF);
  if (needDistinctChroma) {
    _aligned_free(VXSmallUVFF);
    _aligned_free(VYSmallUVFF);
  }

  _aligned_free(SADMaskSmallB);
  _aligned_free(SADMaskSmallF);

}

// Renders the destination rows [y_beg, y_end) (luma rows; chroma scaled)
// of all planes. A row depends only on its own row of the full-frame
// vectors and masks and on the reference frames, which are read-only, so
// the slices need no synchronisation. The FlowInter* functions advance
// their pointers row by row; a slice just starts them at its first row.
template <typename pixel_t>
void MVFlowFps::interpolate_rows(int y_beg, int y_end)
{
  for (int p = 0; p < 3; p++)
  {
    if (p > 0 && isGrey)
      break;
    const bool uv = (p > 0 && needDistinctChroma);
    const int yb = uv ? (y_beg >> nLogyRatioUVs[1]) : y_beg;
    const int ye = uv ? (y_end >> nLogyRatioUVs[1]) : y_end;
    if (ye <= yb)
      continue;
    const int vpitch = uv ? VPitchUV : VPitchY;
    const int voff = yb * vpitch;
    // one destination row is nPel reference rows (ref_pitch << NPELL2 in FlowInter*_NPel)
    const int refoff = (uv ? _cpNOffsetUV : _cpNOffsetY) + yb * _cpNRefPitches[p] * nPel;
    BYTE *pdst = _cpPDst[p] + yb * _cpNDstPitches[p];
    const BYTE *prefB = _cpPRef[p] + refoff;
    const BYTE *prefF = _cpPSrc[p] + refoff;
    short *vxB = (uv ? VXFullUVB : VXFullYB) + voff;
    short *vxF = (uv ? VXFullUVF : VXFullYF) + voff;
    short *vyB = (uv ? VYFullUVB : VYFullYB) + voff;
    short *vyF = (uv ? VYFullUVF : VYFullYF) + voff;
    BYTE *maskB = (uv ? MaskFullUVB : MaskFullYB) + voff;
    BYTE *maskF = (uv ? MaskFullUVF : MaskFullYF) + voff;
    const int width = uv ? nWidthUV : nWidth;
    const int height = ye - yb;

    if (_cpMode == 2)
    {
      FlowInterExtra<pixel_t>(pdst, _cpNDstPitches[p], prefB, prefF, _cpNRefPitches[p],
        vxB, vxF, vyB, vyF, maskB, maskF, vpitch,
        width, height, _cpTime256, nPel,
        (uv ? VXFullUVBB : VXFullYBB) + voff, (uv ? VXFullUVFF : VXFullYFF) + voff,
        (uv ? VYFullUVBB : VYFullYBB) + voff, (uv ? VYFullUVFF : VYFullYFF) + voff);
    }
    else if (_cpMode == 1)
    {
      FlowInter<pixel_t>(pdst, _cpNDstPitches[p], prefB, prefF, _cpNRefPitches[p],
        vxB, vxF, vyB, vyF, maskB, maskF, vpitch,
        width, height, _cpTime256, nPel);
    }
    else
    {
      FlowInterSimple<pixel_t>(pdst, _cpNDstPitches[p], prefB, prefF, _cpNRefPitches[p],
        vxB, vxF, vyB, vyF, maskB, maskF, vpitch,
        width, height, _cpTime256, nPel);
    }
  }
}

void MVFlowFps::upsize_vectors(short *VXFullY, short *VYFullY, short *VXFullUV, short *VYFullUV,
  const short *VXSmallY, const short *VYSmallY, const short *VXSmallUV, const short *VYSmallUV,
  int y_beg, int y_end, void *work, void *workUV)
{
  upsizer->SimpleResizeDo_int16(VXFullY, nWidthP, nHeightP, VPitchY, VXSmallY, nBlkXP, nBlkXP, nPel, true, nWidth, nHeight, y_beg, y_end, work);
  upsizer->SimpleResizeDo_int16(VYFullY, nWidthP, nHeightP, VPitchY, VYSmallY, nBlkXP, nBlkXP, nPel, false, nWidth, nHeight, y_beg, y_end, work);
  if (needDistinctChroma) {
    const int yb = y_beg >> nLogyRatioUVs[1];
    const int ye = y_end >> nLogyRatioUVs[1];
    upsizerUV->SimpleResizeDo_int16(VXFullUV, nWidthPUV, nHeightPUV, VPitchUV, VXSmallUV, nBlkXP, nBlkXP, nPel, true, nWidthUV, nHeightUV, yb, ye, workUV);
    upsizerUV->SimpleResizeDo_int16(VYFullUV, nWidthPUV, nHeightPUV, VPitchUV, VYSmallUV, nBlkXP, nBlkXP, nPel, false, nWidthUV, nHeightUV, yb, ye, workUV);
  }
}

void MVFlowFps::upsize_mask(BYTE *MaskFullY, BYTE *MaskFullUV, const BYTE *MaskSmall,
  int y_beg, int y_end, void *work, void *workUV)
{
  upsizer->SimpleResizeDo_uint8(MaskFullY, nWidthP, nHeightP, VPitchY, MaskSmall, nBlkXP, nBlkXP, y_beg, y_end, work);
  if (needDistinctChroma)
    upsizerUV->SimpleResizeDo_uint8(MaskFullUV, nWidthPUV, nHeightPUV, VPitchUV, MaskSmall, nBlkXP, nBlkXP, y_beg >> nLogyRatioUVs[1], y_end >> nLogyRatioUVs[1], workUV);
}

// Upsizes (bilinear) the small vectors and masks this frame needs to frame
// size, rows [y_beg, y_end) of the padded frame (chroma scaled). Each row
// of the result depends only on the small arrays, so the slices are
// independent; the work buffers are per call because the ones inside
// SimpleResize are not thread-safe.
void MVFlowFps::upsize_rows(int y_beg, int y_end)
{
  void *work = _aligned_malloc(upsizer->work_size(), 128);
  void *workUV = needDistinctChroma ? _aligned_malloc(upsizerUV->work_size(), 128) : 0;

  if (_cpUpsizeB)
    upsize_vectors(VXFullYB, VYFullYB, VXFullUVB, VYFullUVB, VXSmallYB, VYSmallYB, VXSmallUVB, VYSmallUVB, y_beg, y_end, work, workUV);
  upsize_mask(MaskFullYB, MaskFullUVB, MaskSmallB, y_beg, y_end, work, workUV);

  if (_cpUpsizeF)
    upsize_vectors(VXFullYF, VYFullYF, VXFullUVF, VYFullUVF, VXSmallYF, VYSmallYF, VXSmallUVF, VYSmallUVF, y_beg, y_end, work, workUV);
  upsize_mask(MaskFullYF, MaskFullUVF, MaskSmallF, y_beg, y_end, work, workUV);

  if (_cpMode == 2)
  {
    upsize_vectors(VXFullYBB, VYFullYBB, VXFullUVBB, VYFullUVBB, VXSmallYBB, VYSmallYBB, VXSmallUVBB, VYSmallUVBB, y_beg, y_end, work, workUV);
    upsize_vectors(VXFullYFF, VYFullYFF, VXFullUVFF, VYFullUVFF, VXSmallYFF, VYSmallYFF, VXSmallUVFF, VYSmallUVFF, y_beg, y_end, work, workUV);
  }

  _aligned_free(work);
  if (workUV)
    _aligned_free(workUV);
}

// One slice: rows [y_beg, y_end) of the padded frame. The interpolation of
// a row reads only the same row of the upsized arrays, so a slice can
// interpolate its rows right after upsizing them, while they are in cache.
void MVFlowFps::process_slice(Slicer::TaskData &td)
{
  upsize_rows(td._y_beg, td._y_end);

  const int y_end = std::min(td._y_end, nHeight); // no picture in the padding rows
  if (td._y_beg < y_end)
  {
    if (pixelsize_super == 1)
      interpolate_rows<uint8_t>(td._y_beg, y_end);
    else if (pixelsize_super == 2)
      interpolate_rows<uint16_t>(td._y_beg, y_end);
    else
      interpolate_rows<float>(td._y_beg, y_end);
  }
}

//-------------------------------------------------------------------------
// GPU version of the mask=2 path for output frame n. The vector clips are in the state "B at nleft, F at nright" when
// this is called; it reads the four vector fields, renders the frame on the GPU and returns true: dst is a token frame
// when the source clip's frame is one (or regen_out receives the token, for a regeneration), else a normal frame. On any
// problem it restores that state and returns false, and the CPU code renders the frame.
bool MVFlowFps::gpu_flow(int n, int nleft, int nright, int time256, PVideoFrame &dst, MvCudaTokenHeader *regen_out, IScriptEnvironment *env)
{
  GpuBackend &gpu = GpuBackend::instance();
  if (GpuBackend::force_fallback())
    return false;
  GpuScope sc_total("flow: gpu_flow total");
  const int nb = nBlkX * nBlkY;
  std::vector<int> vb(size_t(nb) * 3), vf(size_t(nb) * 3), vbb(size_t(nb) * 3), vff(size_t(nb) * 3);
  auto grab = [&](MVClip &c, std::vector<int> &v) {
    for (int i = 0; i < nb; i++)
    {
      const VECTOR m = c.GetBlock(0, i).GetMV();
      v[3 * i] = m.x;
      v[3 * i + 1] = m.y;
      v[3 * i + 2] = (int)m.sad;
    }
  };
  {
    GpuScope sc("flow: grab B,F vectors");
    grab(mvClipB, vb);
    grab(mvClipF, vf);
  }

  bool ok = false;
  {
    GpuScope sc("flow: FF/BB GetFrame+Update+grab");
    PVideoFrame mvFF = mvClipF.GetFrame(nleft, env);
    mvClipF.Update(mvFF, env); // forward vectors of the left frame
    PVideoFrame mvBB = mvClipB.GetFrame(nright, env);
    mvClipB.Update(mvBB, env); // backward vectors of the right frame
    if (mvClipB.IsUsable() && mvClipF.IsUsable())
    {
      grab(mvClipF, vff);
      grab(mvClipB, vbb);
      ok = true;
    }
  }

  int rc = -1;
  if (ok)
  {
    try
    {
      PVideoFrame sl, sr, src0;
      {
        GpuScope sc("flow: super+child GetFrame (CPU MSuper)");
        sl = _super_clip->GetFrame(nleft, env);
        sr = _super_clip->GetFrame(nright, env);
        src0 = child->GetFrame(nleft, env);
      }
      // the pictures: device frames, or host pictures uploaded once per super frame
      GpuBackend::Input in[2];
      const VideoInfo &svi = _super_clip->GetVideoInfo();
      MvCudaTokenHeader src_tok;
      const bool token_out = regen_out || GpuBackend::frame_token(src0, child->GetVideoInfo(), src_tok);
      const bool in_ok = GpuBackend::super_input(sl, svi, (const void *)_super_clip.operator->(), nleft, in[0])
        && GpuBackend::super_input(sr, svi, (const void *)_super_clip.operator->(), nright, in[1]) && in[0].fmt.planes == 3;

      MvCudaSuperParams sp = {};
      sp.width = nWidth;
      sp.height = nHeight;
      sp.x_ratio_uv = xRatioUVs[1];
      sp.y_ratio_uv = yRatioUVs[1];
      sp.hpad = nHPadding;
      sp.vpad = nVPadding;
      sp.pel = nPel;
      sp.levels = 1;
      sp.bits_per_pixel = bits_per_pixel_super;
      sp.chroma = 1;
      sp.rfilter = 4; // irrelevant for level 0
      sp.sharp = 2;

      MvCudaFlowParams fp = {};
      fp.width = nWidth;
      fp.height = nHeight;
      fp.x_ratio_uv = xRatioUVs[1];
      fp.y_ratio_uv = yRatioUVs[1];
      fp.chroma_distinct = needDistinctChroma ? 1 : 0;
      fp.n_blk_x = nBlkX;
      fp.n_blk_y = nBlkY;
      fp.n_blk_xp = nBlkXP;
      fp.n_blk_yp = nBlkYP;
      fp.blk_step_x = nBlkSizeX - nOverlapX;
      fp.blk_step_y = nBlkSizeY - nOverlapY;
      fp.width_p = nWidthP;
      fp.height_p = nHeightP;
      fp.width_p_uv = nWidthPUV;
      fp.height_p_uv = nHeightPUV;
      fp.time256 = time256;
      fp.occ_norm = 10 / ml / nPel;
      fp.mode = 2;

      GpuScope sc_call("flow: flow_token call");
      MvCudaTokenHeader h;
      if (!in_ok)
        rc = -4;
      else if (token_out)
      {
        rc = gpu.with_inputs(in, 2, env, [&](const unsigned long long *tk) {
          return gpu.flow_token(&sp, &fp, tk[0], tk[1], vb.data(), vf.data(), vbb.data(), vff.data(), _producer, n, &h);
        });
        if (rc == 0 && regen_out)
          *regen_out = h;
        else if (rc == 0)
        {
          dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src0) : env->NewVideoFrame(vi);
          GpuBackend::write_token(dst, h);
        }
      }
      else
      {
        // normal frames in, normal frames out: the picture comes back in the same call
        dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src0) : env->NewVideoFrame(vi);
        const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        void *dp[3];
        int dpitch[3];
        for (int p = 0; p < 3; p++)
        {
          dp[p] = dst->GetWritePtr(planes_y[p]);
          dpitch[p] = dst->GetPitch(planes_y[p]);
        }
        rc = gpu.with_inputs(in, 2, env, [&](const unsigned long long *tk) {
          return gpu.flow_token_host(&sp, &fp, tk[0], tk[1], vb.data(), vf.data(), vbb.data(), vff.data(), dp, dpitch);
        });
      }
    }
    catch (...)
    {
      rc = -2;
    }
  }
  if (GpuBackend::log_enabled())
    std::fprintf(stderr, "MFlowFps gpu frames %d/%d t=%d rc=%d%s\n", nleft, nright, time256, ok ? rc : -9, (ok && rc == 0) ? "" : " -> CPU");
  if (ok && rc == 0)
    return true;
  if (regen_out)
    return false; // GetFrame fetches the vectors again

  // restore the state the CPU path starts from: B at nleft, F at nright
  PVideoFrame mvF = mvClipF.GetFrame(nright, env);
  mvClipF.Update(mvF, env);
  PVideoFrame mvB = mvClipB.GetFrame(nleft, env);
  mvClipB.Update(mvB, env);
  dst = PVideoFrame();
  return false;
}

// Makes the token frame n again (its device frame was evicted before a consumer read it). Same steps as GetFrame.
int MVFlowFps::gpu_regen(void *user, int frame, void *env_v, MvCudaTokenHeader *out)
{
  MVFlowFps *self = static_cast<MVFlowFps *>(user);
  IScriptEnvironment *env = static_cast<IScriptEnvironment *>(env_v);
  std::lock_guard<std::mutex> lk(self->_gpu_mutex);
  const int nleft = (int)(int64_t(frame) * self->fa / self->fb);
  int time256 = int((double(frame) * double(self->fa) / double(self->fb) - nleft) * 256 + 0.5);
  const int off = self->mvClipB.GetDeltaFrame();
  if (off <= 0)
    return -4;
  if (off > 1)
    time256 = time256 / off;
  if (time256 == 0 || time256 == 256)
    return -4; // the source clip's frame, not one of ours
  const int nright = nleft + off;
  PVideoFrame mvF = self->mvClipF.GetFrame(nright, env);
  self->mvClipF.Update(mvF, env);
  PVideoFrame mvB = self->mvClipB.GetFrame(nleft, env);
  self->mvClipB.Update(mvB, env);
  if (!self->mvClipB.IsUsable() || !self->mvClipF.IsUsable())
    return -4;
  PVideoFrame unused;
  return self->gpu_flow(frame, nleft, nright, time256, unused, out, env) ? 0 : -2;
}

//-------------------------------------------------------------------------
PVideoFrame __stdcall MVFlowFps::GetFrame(int n, IScriptEnvironment* env)
{
  // with the GPU, a regeneration (gpu_regen) may come in from another thread: it shares the vector clip state
  std::unique_lock<std::mutex> gpu_lock(_gpu_mutex, std::defer_lock);
  if (_gpu_ok)
    gpu_lock.lock();
  if (reentrancy_check) {
    env->ThrowError("MFlowFps: error in frame %d: Previous GetFrame did not finished properly. There was a crash or filter was set to reentrant multithread mode!", n);
  }
  reentrancy_check = true;
  GpuScope sc_getframe("flow: GetFrame total");

#ifndef _M_X64
  _mm_empty();
#endif
  _RPT2(0, "MFlowFPS GetFrame, frame=%d id=%d\n", n, _instance_id);

  int nleft = (int)(int64_t(n)* fa / fb);
  // intermediate product may be very large! Now I know how to multiply int64
  int time256 = int((double(n)*double(fa) / double(fb) - nleft) * 256 + 0.5);

  PVideoFrame dst;
  BYTE *pDst[3];
  const BYTE *pRef[3], *pSrc[3];
  int nDstPitches[3], nRefPitches[3], nSrcPitches[3];
  unsigned char *pDstYUY2;
  int nDstPitchYUY2;


  int off = mvClipB.GetDeltaFrame(); // integer offset of reference frame
  if (off <= 0)
  {
    env->ThrowError("MFlowFps: cannot use motion vectors with absolute frame references.");
  }
  // usually off must be = 1
  if (off > 1)
    time256 = time256 / off;

  int nright = nleft + off;

// v2.0
  if (time256 == 0) {
    dst = child->GetFrame(nleft, env); // simply left
    if (optDebug != 0) {
      env->MakeWritable(&dst);
      char buf[2048];
      snprintf(buf, sizeof(buf), "FRAME %d time256=%d off=%d, nleft=%d, nright=%d, fa=%d, fb=%d, using left!", n, time256, off, nleft, nright, (int)fa, (int)fb);
      DrawString(dst, vi, 0, 0, buf);
    }
    reentrancy_check = false;
    return dst;
  }
  else if (time256 == 256) {
    dst = child->GetFrame(nright, env); // simply right
    if (optDebug != 0) {
      env->MakeWritable(&dst);
      char buf[2048];
      snprintf(buf, sizeof(buf), "FRAME %d time256=%d off=%d, nleft=%d, nright=%d, fa=%d, fb=%d, using left!", n, time256, off, nleft, nright, (int)fa, (int)fb);
      DrawString(dst, vi, 0, 0, buf);
    }
    reentrancy_check = false;
    return dst;
  }

  _RPT3(0, "Before mvClipF GetFrame frame %d, nright=%d id=%d\n", n, nright, _instance_id);
  {
    GpuScope sc("flow: vector GetFrame F (upstream analysis+recalc)");
    PVideoFrame mvF = mvClipF.GetFrame(nright, env);
    mvClipF.Update(mvF, env);// forward from current to next
    mvF = 0;
  }
  _RPT3(0, "Before mvClipB GetFrame frame %d, nleft=%d id=%d\n", n, nleft, _instance_id);
  {
    GpuScope sc("flow: vector GetFrame B (upstream analysis+recalc)");
    PVideoFrame mvB = mvClipB.GetFrame(nleft, env);
    mvClipB.Update(mvB, env);// backward from next to current
    mvB = 0;
  }

  // Checked here instead of the constructor to allow using multi-vector
  // clips, because the backward flag is not reliable before the vector
  // data are actually read from the frame.
  if (!mvClipB.IsBackward())
    env->ThrowError("MFlowFps: wrong backward vectors");
  if (mvClipF.IsBackward())
    env->ThrowError("MFlowFps: wrong forward vectors");

  if (_gpu_ok && mvClipB.IsUsable() && mvClipF.IsUsable())
  {
    PVideoFrame gdst;
    if (gpu_flow(n, nleft, nright, time256, gdst, nullptr, env))
    {
      reentrancy_check = false;
      return gdst;
    }
  }

  PVideoFrame	src = finest->GetFrame(nleft, env); // move here - v2.0
  PVideoFrame ref = finest->GetFrame(nright, env);//  right frame for  compensation

  dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi); // frame property support

  bool isUsableB = mvClipB.IsUsable();
  bool isUsableF = mvClipF.IsUsable();

  if (isUsableB && isUsableF)
  {
#ifdef MV_GOLDEN_DUMP
    const bool golden_on = golden::begin("MFlowFps", n, false);
    if (golden_on)
    {
      golden::logf("call n %d nleft %d nright %d time256 %d off %d fa %lld fb %lld", n, nleft, nright, time256, off, (long long)fa, (long long)fb);
      golden::logf("geometry width %d height %d blkX %d blkY %d blkXP %d blkYP %d blkSizeX %d blkSizeY %d overlapX %d overlapY %d pel %d", nWidth, nHeight, nBlkX, nBlkY, nBlkXP, nBlkYP, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, nPel);
      golden::logf("geometry2 widthP %d heightP %d widthPUV %d heightPUV %d widthUV %d heightUV %d hpad %d vpad %d vpitchY %d vpitchUV %d xratioUV %d yratioUV %d distinctChroma %d pixelsize %d bits %d", nWidthP, nHeightP, nWidthPUV, nHeightPUV, nWidthUV, nHeightUV, nHPadding, nVPadding, VPitchY, VPitchUV, xRatioUVs[1], yRatioUVs[1], needDistinctChroma ? 1 : 0, pixelsize_super, bits_per_pixel_super);
      golden::logf("params maskmode %d ml %.17g blend %d mt %d", maskmode, ml, blend ? 1 : 0, _mt_flag ? 1 : 0);
      golden_dump_vectors("vec_B.bin", mvClipB, nBlkX, nBlkY);
      golden_dump_vectors("vec_F.bin", mvClipF, nBlkX, nBlkY);
    }
#endif

    PROFILE_START(MOTION_PROFILE_YUY2CONVERT);

    if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2)
    {
            // planar data packed to interleaved format (same as interleved2planar by kassandro) - v2.0.0.5
      pSrc[0] = src->GetReadPtr();
      pSrc[1] = pSrc[0] + src->GetRowSize() / 2;
      pSrc[2] = pSrc[1] + src->GetRowSize() / 4;
      nSrcPitches[0] = src->GetPitch();
      nSrcPitches[1] = nSrcPitches[0];
      nSrcPitches[2] = nSrcPitches[0];
      // planar data packed to interleaved format (same as interleved2planar by kassandro) - v2.0.0.5
      pRef[0] = ref->GetReadPtr();
      pRef[1] = pRef[0] + ref->GetRowSize() / 2;
      pRef[2] = pRef[1] + ref->GetRowSize() / 4;
      nRefPitches[0] = ref->GetPitch();
      nRefPitches[1] = nRefPitches[0];
      nRefPitches[2] = nRefPitches[0];

      if (!planar)
      {
        pDstYUY2 = dst->GetWritePtr();
        nDstPitchYUY2 = dst->GetPitch();
        pDst[0] = DstPlanes->GetPtr();
        pDst[1] = DstPlanes->GetPtrU();
        pDst[2] = DstPlanes->GetPtrV();
        nDstPitches[0] = DstPlanes->GetPitch();
        nDstPitches[1] = DstPlanes->GetPitchUV();
        nDstPitches[2] = DstPlanes->GetPitchUV();
      }
      else
      {
        pDst[0] = dst->GetWritePtr();
        pDst[1] = pDst[0] + dst->GetRowSize() / 2;
        pDst[2] = pDst[1] + dst->GetRowSize() / 4;
        nDstPitches[0] = dst->GetPitch();
        nDstPitches[1] = nDstPitches[0];
        nDstPitches[2] = nDstPitches[0];
      }
      planecount = 3;
    }
    else
    {
      int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
      int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
      int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
      planecount = std::min(vi.NumComponents(), 3);
      for (int p = 0; p < planecount; ++p) {
        const int plane = planes[p];
        pSrc[p] = src->GetReadPtr(plane);
        nSrcPitches[p] = src->GetPitch(plane);

        pRef[p] = ref->GetReadPtr(plane);
        nRefPitches[p] = ref->GetPitch(plane);

        pDst[p] = dst->GetWritePtr(plane);
        nDstPitches[p] = dst->GetPitch(plane);
      }
    }

    PROFILE_STOP(MOTION_PROFILE_YUY2CONVERT);

    int nOffsetY = nRefPitches[0] * nVPadding*nPel + nHPadding*nPel*pixelsize_super;
    int nOffsetUV = nRefPitches[1] * nVPaddingUV*nPel + nHPaddingUV*nPel*pixelsize_super;

    // Block-grid work (small vectors and masks) is done here; upsizing them
    // to frame size is done by the slices, together with the interpolation.
    PROFILE_START(MOTION_PROFILE_MASK);
    const bool upsizeB = (nright != nrightLast);
    if (upsizeB)
    {
      // make  vector vx and vy small masks
      MakeVectorSmallMasks(mvClipB, nBlkX, nBlkY, VXSmallYB, nBlkXP, VYSmallYB, nBlkXP);

      CheckAndPadSmallY(VXSmallYB, VYSmallYB, nBlkXP, nBlkYP, nBlkX, nBlkY);

      if (needDistinctChroma) {
        VectorSmallMaskYToHalfUV(VXSmallYB, nBlkXP, nBlkYP, VXSmallUVB, xRatioUVs[1]);
        VectorSmallMaskYToHalfUV(VYSmallYB, nBlkXP, nBlkYP, VYSmallUVB, yRatioUVs[1]);
      }
    }
   // analyse vectors field to detect occlusion
   // Backward part
    MakeVectorOcclusionMaskTime(mvClipB, nBlkX, nBlkY, ml, 1.0, nPel, MaskSmallB, nBlkXP, (256 - time256), nBlkSizeX - nOverlapX, nBlkSizeY - nOverlapY);

    CheckAndPadMaskSmall(MaskSmallB, nBlkXP, nBlkYP, nBlkX, nBlkY);

    nrightLast = nright;

    // Forward part
    const bool upsizeF = (nleft != nleftLast);
    if (upsizeF)
    {
     // make  vector vx and vy small masks
      MakeVectorSmallMasks(mvClipF, nBlkX, nBlkY, VXSmallYF, nBlkXP, VYSmallYF, nBlkXP);

      CheckAndPadSmallY(VXSmallYF, VYSmallYF, nBlkXP, nBlkYP, nBlkX, nBlkY);

      if (needDistinctChroma) {
        VectorSmallMaskYToHalfUV(VXSmallYF, nBlkXP, nBlkYP, VXSmallUVF, xRatioUVs[1]);
        VectorSmallMaskYToHalfUV(VYSmallYF, nBlkXP, nBlkYP, VYSmallUVF, yRatioUVs[1]);
      }
    }
   // analyse vectors field to detect occlusion
   // Forward part
    MakeVectorOcclusionMaskTime(mvClipF, nBlkX, nBlkY, ml, 1.0, nPel, MaskSmallF, nBlkXP, time256, nBlkSizeX - nOverlapX, nBlkSizeY - nOverlapY);

    CheckAndPadMaskSmall(MaskSmallF, nBlkXP, nBlkYP, nBlkX, nBlkY);

    PROFILE_STOP(MOTION_PROFILE_MASK);

    nleftLast = nleft;

    // Backward and forward is ready

  // Get motion info from more frames for occlusion areas
    PVideoFrame mvFF = mvClipF.GetFrame(nleft, env);
    mvClipF.Update(mvFF, env);// forward from prev to cur
    mvFF = 0;

    PVideoFrame mvBB = mvClipB.GetFrame(nright, env);
    mvClipB.Update(mvBB, env);// backward from next next to next
    mvBB = 0;

#ifdef MV_GOLDEN_DUMP
    if (golden_on)
    {
      golden_dump_vectors("vec_FF.bin", mvClipF, nBlkX, nBlkY); // forward vectors of the left frame
      golden_dump_vectors("vec_BB.bin", mvClipB, nBlkX, nBlkY); // backward vectors of the right frame
    }
#endif
    bool isUsableB = mvClipB.IsUsable();
    bool isUsableF = mvClipF.IsUsable();
    _RPT5(0, "part#2 IsUsableB=%d IsUsableF=%d frame=%d,nleft=%d,nright=%d\n", isUsableB ? 1 : 0, isUsableF ? 1 : 0, n, nleft, nright);

    int interMode; // which FlowInter variant renders this frame
    if (maskmode == 2 && isUsableB && isUsableF) // slow method with extra frames
    {
     // get vector mask from extra frames
      PROFILE_START(MOTION_PROFILE_MASK);
      MakeVectorSmallMasks(mvClipB, nBlkX, nBlkY, VXSmallYBB, nBlkXP, VYSmallYBB, nBlkXP);
      MakeVectorSmallMasks(mvClipF, nBlkX, nBlkY, VXSmallYFF, nBlkXP, VYSmallYFF, nBlkXP);

      CheckAndPadSmallY_BF(VXSmallYBB, VXSmallYFF, VYSmallYBB, VYSmallYFF, nBlkXP, nBlkYP, nBlkX, nBlkY);

      if (needDistinctChroma) {
        VectorSmallMaskYToHalfUV(VXSmallYBB, nBlkXP, nBlkYP, VXSmallUVBB, xRatioUVs[1]);
        VectorSmallMaskYToHalfUV(VYSmallYBB, nBlkXP, nBlkYP, VYSmallUVBB, yRatioUVs[1]);
        VectorSmallMaskYToHalfUV(VXSmallYFF, nBlkXP, nBlkYP, VXSmallUVFF, xRatioUVs[1]);
        VectorSmallMaskYToHalfUV(VYSmallYFF, nBlkXP, nBlkYP, VYSmallUVFF, yRatioUVs[1]);
      }
      PROFILE_STOP(MOTION_PROFILE_MASK);

      interMode = 2;
    }
    else if (maskmode == 1) // old method without extra frames
      interMode = 1;
    else // mode=0, faster simple method
      interMode = 0;

    PROFILE_START(MOTION_PROFILE_FLOWINTER);
    {
      // Per-frame state for the slices. GetFrame is never re-entered on the
      // same instance (MT_MULTI_INSTANCE), so plain members are fine.
      for (int p = 0; p < planecount; p++) {
        _cpPDst[p] = pDst[p];
        _cpPRef[p] = pRef[p];
        _cpPSrc[p] = pSrc[p];
        _cpNDstPitches[p] = nDstPitches[p];
        _cpNRefPitches[p] = nRefPitches[p];
      }
      _cpNOffsetY = nOffsetY;
      _cpNOffsetUV = nOffsetUV;
      _cpTime256 = time256;
      _cpMode = interMode;
      _cpUpsizeB = upsizeB;
      _cpUpsizeF = upsizeF;

      Slicer slicer(_mt_flag);
      slicer.start(nHeightP, *this, &MVFlowFps::process_slice, 16);
      slicer.wait();
    }
    PROFILE_STOP(MOTION_PROFILE_FLOWINTER);

#ifdef MV_GOLDEN_DUMP
    if (golden_on)
    {
      golden::logf("interMode %d usableB %d usableF %d", interMode, isUsableB ? 1 : 0, isUsableF ? 1 : 0);
      // small arrays (pitch nBlkXP) and full arrays (pitch VPitchY / VPitchUV, nHeightP rows); the full arrays are
      // the upsized vectors / masks that the interpolation reads
      golden::write_bin("small_vxB.bin", VXSmallYB, 2 * nBlkXP * nBlkYP);
      golden::write_bin("small_vyB.bin", VYSmallYB, 2 * nBlkXP * nBlkYP);
      golden::write_bin("small_vxF.bin", VXSmallYF, 2 * nBlkXP * nBlkYP);
      golden::write_bin("small_vyF.bin", VYSmallYF, 2 * nBlkXP * nBlkYP);
      golden::write_bin("small_maskB.bin", MaskSmallB, nBlkXP * nBlkYP);
      golden::write_bin("small_maskF.bin", MaskSmallF, nBlkXP * nBlkYP);
      if (interMode == 2)
      {
        golden::write_bin("small_vxBB.bin", VXSmallYBB, 2 * nBlkXP * nBlkYP);
        golden::write_bin("small_vyBB.bin", VYSmallYBB, 2 * nBlkXP * nBlkYP);
        golden::write_bin("small_vxFF.bin", VXSmallYFF, 2 * nBlkXP * nBlkYP);
        golden::write_bin("small_vyFF.bin", VYSmallYFF, 2 * nBlkXP * nBlkYP);
      }
      const size_t fy = size_t(VPitchY) * nHeightP;
      golden::write_bin("full_vxB_Y.bin", VXFullYB, 2 * fy);
      golden::write_bin("full_vyB_Y.bin", VYFullYB, 2 * fy);
      golden::write_bin("full_vxF_Y.bin", VXFullYF, 2 * fy);
      golden::write_bin("full_vyF_Y.bin", VYFullYF, 2 * fy);
      golden::write_bin("full_maskB_Y.bin", MaskFullYB, fy);
      golden::write_bin("full_maskF_Y.bin", MaskFullYF, fy);
      if (interMode == 2)
      {
        golden::write_bin("full_vxBB_Y.bin", VXFullYBB, 2 * fy);
        golden::write_bin("full_vyBB_Y.bin", VYFullYBB, 2 * fy);
        golden::write_bin("full_vxFF_Y.bin", VXFullYFF, 2 * fy);
        golden::write_bin("full_vyFF_Y.bin", VYFullYFF, 2 * fy);
      }
      if (needDistinctChroma)
      {
        const size_t fuv = size_t(VPitchUV) * nHeightPUV;
        golden::write_bin("full_vxB_UV.bin", VXFullUVB, 2 * fuv);
        golden::write_bin("full_vyB_UV.bin", VYFullUVB, 2 * fuv);
        golden::write_bin("full_vxF_UV.bin", VXFullUVF, 2 * fuv);
        golden::write_bin("full_vyF_UV.bin", VYFullUVF, 2 * fuv);
        golden::write_bin("full_maskB_UV.bin", MaskFullUVB, fuv);
        golden::write_bin("full_maskF_UV.bin", MaskFullUVF, fuv);
        if (interMode == 2)
        {
          golden::write_bin("full_vxBB_UV.bin", VXFullUVBB, 2 * fuv);
          golden::write_bin("full_vyBB_UV.bin", VYFullUVBB, 2 * fuv);
          golden::write_bin("full_vxFF_UV.bin", VXFullUVFF, 2 * fuv);
          golden::write_bin("full_vyFF_UV.bin", VYFullUVFF, 2 * fuv);
        }
      }
      // the rendered frame, planes without the pitch gap
      for (int p = 0; p < planecount; p++)
      {
        const bool uv = (p > 0 && needDistinctChroma);
        const int w = uv ? nWidthUV : nWidth, h = uv ? nHeightUV : nHeight;
        std::vector<unsigned char> plane(size_t(w) * h * pixelsize_super);
        for (int y = 0; y < h; y++)
          std::memcpy(&plane[size_t(y) * w * pixelsize_super], pDst[p] + size_t(y) * nDstPitches[p], size_t(w) * pixelsize_super);
        char name[32];
        std::snprintf(name, sizeof(name), "out_plane%d.bin", p);
        golden::write_bin(name, plane.data(), plane.size());
      }
      // the raw source frames the interpolation starts from (the super clip's level 0 is a copy of them)
      {
        PVideoFrame fl = child->GetFrame(nleft, env), fr = child->GetFrame(nright, env);
        const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        for (int side = 0; side < 2; side++)
        {
          PVideoFrame &f = side == 0 ? fl : fr;
          for (int p = 0; p < planecount; p++)
          {
            const bool uv = (p > 0 && needDistinctChroma);
            const int w = uv ? nWidthUV : nWidth, h = uv ? nHeightUV : nHeight;
            const unsigned char *rp = f->GetReadPtr(planes_y[p]);
            const int pitch = f->GetPitch(planes_y[p]);
            std::vector<unsigned char> plane(size_t(w) * h * pixelsize_super);
            for (int y = 0; y < h; y++)
              std::memcpy(&plane[size_t(y) * w * pixelsize_super], rp + size_t(y) * pitch, size_t(w) * pixelsize_super);
            char name[32];
            std::snprintf(name, sizeof(name), "%s_raw_plane%d.bin", side == 0 ? "left" : "right", p);
            golden::write_bin(name, plane.data(), plane.size());
          }
        }
      }
      golden::end();
    }
#endif

    if (optDebug > 0) {
      if (interMode == 2) {
        char buf[100];
        snprintf(buf, sizeof(buf), "FlowInter mode=2");
        DrawString(dst, vi, 0, 6, buf);
      }
      else if (interMode == 1) {
        char buf[2048];
        snprintf(buf, sizeof(buf), "FlowInter mode=1");
        DrawString(dst, vi, 0, 6, buf);
      }
      else {
        int sum_VXFullYB = 0;
        int sum_VXFullYF = 0;
        int sum_VYFullYB = 0;
        int sum_VYFullYF = 0;
        int sum_MaskFullYB = 0;
        int sum_MaskFullYF = 0;
        for (int y = 0; y < nHeight; y++)
          for (int x = 0; x < nWidth; x++) {
            sum_VXFullYB += VXFullYB[y * VPitchY + x];
            sum_VXFullYF += VXFullYF[y * VPitchY + x];
            sum_VYFullYB += VYFullYB[y * VPitchY + x];
            sum_VYFullYF += VYFullYF[y * VPitchY + x];
            sum_MaskFullYB += MaskFullYB[y * VPitchY + x];
            sum_MaskFullYF += MaskFullYF[y * VPitchY + x];
          }

        int sum_MaskSmallB = 0;
        int sum_MaskSmallF = 0;
        for (int y = 0; y < nBlkY; y++)
          for (int x = 0; x < nBlkX; x++) {
            sum_MaskSmallB += MaskSmallB[nBlkXP*y + x];
            sum_MaskSmallF += MaskSmallF[nBlkXP*y + x];
          }

        int sum_MaskSmallBP = 0;
        int sum_MaskSmallFP = 0;
        for (int y = 0; y < nBlkYP; y++)
          for (int x = 0; x < nBlkXP; x++) {
            sum_MaskSmallBP += MaskSmallB[nBlkXP*y + x];
            sum_MaskSmallFP += MaskSmallB[nBlkXP*y + x];
          }
        char buf[2048];
        snprintf(buf, sizeof(buf), "FlowInterSimple mode=0 or mode=2 not usable");
        DrawString(dst, vi, 0, 6, buf);
        snprintf(buf, sizeof(buf), "sum_VXFullYB=%d sum_VXFullYF=%d", sum_VXFullYB, sum_VXFullYF);
        DrawString(dst, vi, 0, 7, buf);
        snprintf(buf, sizeof(buf), "sum_VYFullYB=%d sum_VYFullYF=%d", sum_VYFullYB, sum_VYFullYF);
        DrawString(dst, vi, 0, 8, buf);
        snprintf(buf, sizeof(buf), "sum_MaskFullYB=%d sum_MaskFullYF=%d", sum_MaskFullYB, sum_MaskFullYF);
        DrawString(dst, vi, 0, 9, buf);
        snprintf(buf, sizeof(buf), "sum_MaskSmallBP=%d sum_MaskSmallFP=%d", sum_MaskSmallBP, sum_MaskSmallFP);
        DrawString(dst, vi, 0, 10, buf);
        snprintf(buf, sizeof(buf), "sum_MaskSmallB=%d sum_MaskSmallF=%d", sum_MaskSmallB, sum_MaskSmallF);
        DrawString(dst, vi, 0, 11, buf);

        snprintf(buf, sizeof(buf), "FRAME %d time256=%d IsUsableB=%d IsUsableF=%d", n, time256, isUsableB ? 1 : 0, isUsableF ? 1 : 0);
        DrawString(dst, vi, 0, 0, buf);
        snprintf(buf, sizeof(buf), "B: BlkCount=%d OVx=%d OVy=%d thSCD1=%d thSCD2=%d", mvClipB.GetBlkCount(), mvClipB.GetOverlapX(), mvClipB.GetOverlapY(), mvClipB.GetThSCD1(), mvClipB.GetThSCD2());
        DrawString(dst, vi, 0, 1, buf);
        snprintf(buf, sizeof(buf), "F: BlkCount=%d OVx=%d OVy=%d thSCD1=%d thSCD2=%d", mvClipF.GetBlkCount(), mvClipF.GetOverlapX(), mvClipF.GetOverlapY(), mvClipF.GetThSCD1(), mvClipF.GetThSCD2());
        DrawString(dst, vi, 0, 2, buf);
        snprintf(buf, sizeof(buf), "nBlkX=%d nBlkY=%d nBlkXP=%d nBlkY=%d", nBlkX, nBlkY, nBlkXP, nBlkYP);
        DrawString(dst, vi, 0, 3, buf);
        snprintf(buf, sizeof(buf), "nright=%d nleft=%d", nright, nleft);
        DrawString(dst, vi, 0, 4, buf);
      }
    }

    PROFILE_START(MOTION_PROFILE_YUY2CONVERT);
    if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2 && !planar)
    {
      YUY2FromPlanes(pDstYUY2, nDstPitchYUY2, nWidth, nHeight,
        pDst[0], nDstPitches[0], pDst[1], pDst[2], nDstPitches[1], cpuFlags);
    }
    PROFILE_STOP(MOTION_PROFILE_YUY2CONVERT);
    _RPT2(0, "MVFlowFPS GetFrame END, frame=%d id=%d\n", n, _instance_id);
    reentrancy_check = false;
    return dst;
  }
  else
  {
       // poor estimation

       // prepare pointers
    PVideoFrame src = child->GetFrame(nleft, env); // it is easy to use child here - v2.0

    if (blend) //let's blend src with ref frames like ConvertFPS
    {
      PVideoFrame ref = child->GetFrame(nright, env);
      PROFILE_START(MOTION_PROFILE_FLOWINTER);
      if ((pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2)
      {
        pSrc[0] = src->GetReadPtr(); // we can blend YUY2
        nSrcPitches[0] = src->GetPitch();
        pRef[0] = ref->GetReadPtr();
        nRefPitches[0] = ref->GetPitch();
        pDstYUY2 = dst->GetWritePtr();
        nDstPitchYUY2 = dst->GetPitch();
        Blend<uint8_t>(pDstYUY2, pSrc[0], pRef[0], nHeight, nWidth * 2, nDstPitchYUY2, nSrcPitches[0], nRefPitches[0], time256, cpuFlags);
      }
      else
      {
        int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
        int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
        int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
        planecount = std::min(vi.NumComponents(), 3);
        for (int p = 0; p < planecount; ++p) {
          const int plane = planes[p];
          pSrc[p] = src->GetReadPtr(plane);
          nSrcPitches[p] = src->GetPitch(plane);

          pRef[p] = ref->GetReadPtr(plane);
          nRefPitches[p] = ref->GetPitch(plane);

          pDst[p] = dst->GetWritePtr(plane);
          nDstPitches[p] = dst->GetPitch(plane);
        }
        // blend with time weight
        if (pixelsize_super == 1) {
          Blend<uint8_t>(pDst[0], pSrc[0], pRef[0], nHeight, nWidth, nDstPitches[0], nSrcPitches[0], nRefPitches[0], time256, cpuFlags);
          if (!isGrey) {
            if (needDistinctChroma) {
              Blend<uint8_t>(pDst[1], pSrc[1], pRef[1], nHeightUV, nWidthUV, nDstPitches[1], nSrcPitches[1], nRefPitches[1], time256, cpuFlags);
              Blend<uint8_t>(pDst[2], pSrc[2], pRef[2], nHeightUV, nWidthUV, nDstPitches[2], nSrcPitches[2], nRefPitches[2], time256, cpuFlags);
            }
            else {
              Blend<uint8_t>(pDst[1], pSrc[1], pRef[1], nHeight, nWidth, nDstPitches[1], nSrcPitches[1], nRefPitches[1], time256, cpuFlags);
              Blend<uint8_t>(pDst[2], pSrc[2], pRef[2], nHeight, nWidth, nDstPitches[2], nSrcPitches[2], nRefPitches[2], time256, cpuFlags);
            }
          }
        }
        else if (pixelsize_super == 2) {
          Blend<uint16_t>(pDst[0], pSrc[0], pRef[0], nHeight, nWidth, nDstPitches[0], nSrcPitches[0], nRefPitches[0], time256, cpuFlags);
          if (!isGrey) {
            if (needDistinctChroma) {
              Blend<uint16_t>(pDst[1], pSrc[1], pRef[1], nHeightUV, nWidthUV, nDstPitches[1], nSrcPitches[1], nRefPitches[1], time256, cpuFlags);
              Blend<uint16_t>(pDst[2], pSrc[2], pRef[2], nHeightUV, nWidthUV, nDstPitches[2], nSrcPitches[2], nRefPitches[2], time256, cpuFlags);
            }
            else {
              Blend<uint16_t>(pDst[1], pSrc[1], pRef[1], nHeight, nWidth, nDstPitches[1], nSrcPitches[1], nRefPitches[1], time256, cpuFlags);
              Blend<uint16_t>(pDst[2], pSrc[2], pRef[2], nHeight, nWidth, nDstPitches[2], nSrcPitches[2], nRefPitches[2], time256, cpuFlags);
            }
          }
        }
        else if (pixelsize_super == 4) {
          Blend<float>(pDst[0], pSrc[0], pRef[0], nHeight, nWidth, nDstPitches[0], nSrcPitches[0], nRefPitches[0], time256, cpuFlags);
          if (!isGrey) {
            if (needDistinctChroma) {
              Blend<float>(pDst[1], pSrc[1], pRef[1], nHeightUV, nWidthUV, nDstPitches[1], nSrcPitches[1], nRefPitches[1], time256, cpuFlags);
              Blend<float>(pDst[2], pSrc[2], pRef[2], nHeightUV, nWidthUV, nDstPitches[2], nSrcPitches[2], nRefPitches[2], time256, cpuFlags);
            }
            else {
              Blend<float>(pDst[1], pSrc[1], pRef[1], nHeight, nWidth, nDstPitches[1], nSrcPitches[1], nRefPitches[1], time256, cpuFlags);
              Blend<float>(pDst[2], pSrc[2], pRef[2], nHeight, nWidth, nDstPitches[2], nSrcPitches[2], nRefPitches[2], time256, cpuFlags);
            }
          }
        }
      }
      PROFILE_STOP(MOTION_PROFILE_FLOWINTER);
      if (optDebug > 0) {
        char buf[2048];
        snprintf(buf, sizeof(buf), "BLEND %d time256=%d off=%d, nleft=%d, nright=%d, fa=%d, fb=%d, using left!", n, time256, off, nleft, nright, (int)fa, (int)fb);
        DrawString(dst, vi, 0, 0, buf);
        _RPT2(0, "MVFlowFPS GetFrame END BLEND, frame=%d id=%d\n", n, _instance_id);
      }

      reentrancy_check = false;
      return dst;
    }
    else
    {
      if (optDebug > 0) {
        env->MakeWritable(&src);
        char buf[2048];
        snprintf(buf, sizeof(buf), "POORNOBLEND %d time256=%d IsUsableB=%d IsUsableF=%d", n, time256, isUsableB ? 1 : 0, isUsableF ? 1 : 0);
        DrawString(src, vi, 0, 0, buf);
        snprintf(buf, sizeof(buf), "B: BlkCount=%d OVx=%d OVy=%d thSCD1=%d thSCD2=%d", mvClipB.GetBlkCount(), mvClipB.GetOverlapX(), mvClipB.GetOverlapY(), mvClipB.GetThSCD1(), mvClipB.GetThSCD2());
        DrawString(src, vi, 0, 1, buf);
        snprintf(buf, sizeof(buf), "F: BlkCount=%d OVx=%d OVy=%d thSCD1=%d thSCD2=%d", mvClipF.GetBlkCount(), mvClipF.GetOverlapX(), mvClipF.GetOverlapY(), mvClipF.GetThSCD1(), mvClipF.GetThSCD2());
        DrawString(src, vi, 0, 2, buf);
        snprintf(buf, sizeof(buf), "nright=%d nleft=%d", nright, nleft);
        DrawString(src, vi, 0, 4, buf);
        _RPT2(0, "MVFlowFPS GetFrame END POORNOBLEND, frame=%d id=%d\n", n, _instance_id);
      }
      reentrancy_check = false;
      return src; // like ChangeFPS
    }

  }

}
