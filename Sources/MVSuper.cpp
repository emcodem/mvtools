// MVTools v2

// Copyright(c)2008 A.G.Balakhnin aka Fizick
// Prepare super clip (hierachical levels data) for MVAnalyse

// See legal notice in Copying.txt for more information

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
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

#include	"debugprintf.h"
#include "ClipFnc.h"
#include "MVFrame.h"
#include "MVGroupOfFrames.h"
#include "MVPlane.h"
#include "MVSuper.h"
#include	"profile.h"
#include "SuperParams64Bits.h"
#include "GpuBackend.h"
#include <stdint.h>
#include "CopyCode.h"

#include <cmath>

// move profile init here from MVCore.cpp (it is not quite correct for several mvsuper)
#ifdef MOTION_PROFILE
int ProfTimeTable[MOTION_PROFILE_COUNT];
int ProfResults[MOTION_PROFILE_COUNT * 2];
int64_t ProfCumulatedResults[MOTION_PROFILE_COUNT * 2];
#endif


MVSuper::MVSuper(
  PClip _child, int _hPad, int _vPad, int _pel, int _levels, bool _chroma,
  int _sharp, int _rfilter, PClip _pelclip, bool _isse,
  bool mt_flag, bool yuy2_source, bool gpu_flag, IScriptEnvironment* env
)
  : GenericVideoFilter(_child)
  , pelclip(_pelclip)
  , _mt_flag(mt_flag)
  , _lite(false)
{
  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }


  nWidth = vi.width;

  nHeight = vi.height;

  // 2.7.35 planar RGB support
  if (!ClipFnc::is_yuv_any(vi) && !vi.IsPlanarRGB() && !vi.IsPlanarRGBA()) // YUY2 is also YUV but let's see what is supported
  {
    env->ThrowError("MSuper: Clip must be Y(A), YUV(A), YUY2 or planar RGB(A)");
  }

  nPel = _pel;
  if ((nPel != 1) && (nPel != 2) && (nPel != 4))
  {
    env->ThrowError("MSuper: pel has to be 1 or 2 or 4");
  }

  nHPad = _hPad;
  nVPad = _vPad;
  rfilter = _rfilter;
  sharp = _sharp; // pel2 interpolation type
  cpuFlags = _isse ? env->GetCPUFlags() : 0;

  chroma = _chroma;
  if (vi.IsRGB())
    chroma = true; // Chroma means: all channels, for RGB: compulsory
  if (ClipFnc::is_grey(vi))
    chroma = false; // Grey support, no U V plane usage

  nModeYUV = chroma ? YUVPLANES : YPLANE;

  pixelType = vi.pixel_type;
  if (!ClipFnc::is_grey(vi) && !vi.IsRGB()) {
    yRatioUV = (1 << vi.GetPlaneHeightSubsampling(PLANAR_U));
    xRatioUV = (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)); // for YV12 and YUY2, really do not used and assumed to 2
  }
  else {
    yRatioUV = 1; // n/a
    xRatioUV = 1; // n/a
  }
  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  nLevels = _levels;
  int nLevelsMax = 0;
  while (PlaneHeightLuma(vi.height, nLevelsMax, yRatioUV, nVPad) >= yRatioUV * 2 &&
    PlaneWidthLuma(vi.width, nLevelsMax, xRatioUV, nHPad) >= xRatioUV * 2) // at last two pixels width and height of chroma
  {
    nLevelsMax++;
  }
  if (nLevels <= 0 || nLevels > nLevelsMax) nLevels = nLevelsMax;

  usePelClip = false;
  if (pelclip && (nPel >= 2))
  {
    if (pelclip->GetVideoInfo().width == vi.width*nPel &&
      pelclip->GetVideoInfo().height == vi.height*nPel)
    {
      usePelClip = true;
      isPelClipPadded = false;
    }
    else if (pelclip->GetVideoInfo().width == (vi.width + nHPad * 2)*nPel &&
      pelclip->GetVideoInfo().height == (vi.height + nVPad * 2)*nPel)
    {
      usePelClip = true;
      isPelClipPadded = true;
    }
    else
    {
      env->ThrowError("MSuper: pelclip frame size must be Pel of source!");
    }
  }

  nSuperWidth = nWidth + 2 * nHPad;
  if (nSuperWidth % xRatioUV != 0) // 4:1:1, 4:1:0 with odd hpad: the frame width must be a multiple of 4
    env->ThrowError("MSuper: hpad must be even for 4:1:1 and 4:1:0 (width + 2 * hpad must be a multiple of 4)");
  nSuperHeight = PlaneSuperOffset(false, nHeight, nLevels, nPel, nVPad, nSuperWidth*pixelsize, yRatioUV) / (nSuperWidth*pixelsize);
  // height must be a multiple of the vertical chroma subsampling (4:2:0, 4:4:0: 2, 4:1:0: 4)
  nSuperHeight = (nSuperHeight + yRatioUV - 1) / yRatioUV * yRatioUV;
  vi.width = nSuperWidth;
  vi.height = nSuperHeight;

  // CudaMSuper: the Cuda* filters downstream build the pyramid and the sub-pel planes on the GPU from the plain picture,
  // so the CPU skips them: the frame holds only the picture with its padding rows ("lite"; for a token clip only the
  // token header at the picture position). A CPU filter that needs the full thing completes it itself (CompleteLite),
  // bit-identical.
  if (gpu_flag)
  {
    GpuCheck c("CudaMSuper", "MSuper");
    c.need(!usePelClip, "pelclip is not supported")
      .need(sharp == 2, "only sharp=2")
      .need(rfilter == 4 || rfilter == 2, "only rfilter 2 or 4")
      .need(vi.IsPlanar() && vi.IsYUV() && !vi.IsY(), "only planar YUV formats")
      .need(pixelsize == 1 || pixelsize == 2, "only 8 to 16 bit integer formats")
      .need((xRatioUV == 1 || xRatioUV == 2) && (yRatioUV == 1 || yRatioUV == 2), "only 4:2:0, 4:2:2, 4:4:0 or 4:4:4 chroma")
      .need(nModeYUV == YUVPLANES || nModeYUV == YPLANE, "unsupported plane selection")
      .need(nHPad % xRatioUV == 0 && nVPad % yRatioUV == 0 && nHeight % yRatioUV == 0,
        "hpad, vpad and the height must be multiples of the chroma subsampling");
    _lite = c.decide(env);
    if (_lite)
      vi.height = nHeight + 2 * nVPad;
  }

  SuperParams64Bits params;

  params.nHeight = nHeight;
  params.nHPad = nHPad;
  params.nVPad = nVPad;
  params.nPel = nPel;
  params.nModeYUV = nModeYUV;
  params.nLevels = nLevels;
  // bit 0: YUY2 source (upstream); bits 1-7: our settings for the GPU paths and lite clips (SuperParams64Bits.h, D4)
  params.param = super_param_make(yuy2_source, rfilter, sharp, usePelClip, _lite);


  // pack parameters to fake audio properties
  memcpy(&vi.num_audio_samples, &params, 8); //nHeight + (nHPad<<16) + (nVPad<<24) + ((_int64)(nPel)<<32) + ((_int64)nModeYUV<<40) + ((_int64)nLevels<<48);
  vi.audio_samples_per_second = 0; // kill audio

  // LDS: why not nModeYUV?
//	pSrcGOF = new MVGroupOfFrames(nLevels, nWidth, nHeight, nPel, nHPad, nVPad, nModeYUV, isse, yRatioUV, mt_flag);
  pSrcGOF = new MVGroupOfFrames(nLevels, nWidth, nHeight, nPel, nHPad, nVPad, YUVPLANES, cpuFlags, xRatioUV, yRatioUV, pixelsize, bits_per_pixel, mt_flag);

  pSrcGOF->set_interp(nModeYUV, rfilter, sharp);

  PROFILE_INIT();
}

MVSuper::~MVSuper()
{
  delete pSrcGOF;

  PROFILE_SHOW();
}

PVideoFrame __stdcall MVSuper::GetFrame(int n, IScriptEnvironment* env)
{
  const unsigned char *pSrc[3];
  unsigned char *pDst[3];
  const unsigned char *pSrcPel[3];
  int nSrcPitch[3];
  int nDstPitch[3];
  int nSrcPelPitch[3];
  
  int planecount;
  
    //DebugPrintf("MSuper: Get src frame %d clip %d",n,child);

  PVideoFrame src;
  {
    GpuScope sc("msuper: child GetFrame (decode+RemoveGrain)");
    src = child->GetFrame(n, env);
  }
  if (_lite)
  {
    GpuScope sc_lite("msuper: lite copy");
    PVideoFrame lite = has_at_least_v8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);
    MvCudaTokenHeader tok;
    if (GpuBackend::read_token(src->GetReadPtr(PLANAR_Y), src->GetRowSize(PLANAR_Y), tok))
    {
      // token frame (picture on the device): pass the header on at the picture position; the GPU filters use the device
      // frame, CPU paths download it (ResolveToken)
      memcpy(lite->GetWritePtr(PLANAR_Y) + size_t(nVPad) * lite->GetPitch(PLANAR_Y) + size_t(nHPad) * pixelsize, &tok, sizeof(tok));
      return lite;
    }
    const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    for (int p = 0; p < 3; ++p)
    {
      const int plane = planes_l[p];
      const int hp = p == 0 ? nHPad : nHPad / xRatioUV;
      const int vp = p == 0 ? nVPad : nVPad / yRatioUV;
      BYTE *d = lite->GetWritePtr(plane) + size_t(vp) * lite->GetPitch(plane) + size_t(hp) * pixelsize;
      env->BitBlt(d, lite->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
    }
    return lite;
  }
  GpuScope sc_build("msuper: build pyramid+subpel (CPU)");
  MvCudaTokenHeader tok;
  if (vi.IsPlanar() && GpuBackend::read_token(src->GetReadPtr(PLANAR_Y), src->GetRowSize(PLANAR_Y), tok))
  {
    // token frame from a GPU filter feeding a super the GPU cannot take over: the CPU needs the picture
    PVideoFrame pic = env->NewVideoFrame(child->GetVideoInfo());
    const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    void *d[3] = { 0, 0, 0 };
    int dp[3] = { 0, 0, 0 };
    for (int p = 0; p < (vi.IsY() ? 1 : 3); ++p)
    {
      d[p] = pic->GetWritePtr(planes_l[p]);
      dp[p] = pic->GetPitch(planes_l[p]);
    }
    if (!GpuBackend::instance().download_token(tok, d, dp, env))
      env->ThrowError("MSuper: the GPU frame %d is no longer available on the device", tok.frame);
    src = pic;
  }
  PVideoFrame srcPel;
  if (usePelClip)
    srcPel = pelclip->GetFrame(n, env);

  PVideoFrame dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi); // frame property support

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  planecount = ClipFnc::nbr_planes_processed(vi);
  for (int p = 0; p < planecount; ++p) {
    const int plane = planes[p];
    pSrc[p] = src->GetReadPtr(plane);
    nSrcPitch[p] = src->GetPitch(plane);
    if (usePelClip)
    {
      pSrcPel[p] = srcPel->GetReadPtr(plane);
      nSrcPelPitch[p] = srcPel->GetPitch(plane);
    }
    pDst[p] = dst->GetWritePtr(plane);
    nDstPitch[p] = dst->GetPitch(plane);
  }
  // P.F. 170519 debug: clear super area
  // todo: check how it affects for non-modulo blocksize vs. supersize_padded
  // maybe some garbage is read outside the area by MAnalyse later?
  /*
  MemZoneSet(pDstY, 0, dst->GetRowSize(PLANAR_Y_ALIGNED), vi.height, 0, 0, nDstPitchY);
  if (!ClipFnc::is_grey(vi)) {
    MemZoneSet(pDstU, 0, dst->GetRowSize(PLANAR_U_ALIGNED), vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U), 0, 0, nDstPitchUV);
    MemZoneSet(pDstV, 0, dst->GetRowSize(PLANAR_V_ALIGNED), vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_V), 0, 0, nDstPitchUV);
  }
  */


  PROFILE_START(MOTION_PROFILE_INTERPOLATION);

  pSrcGOF->Update(YUVPLANES, pDst[0], nDstPitch[0], pDst[1], nDstPitch[1], pDst[2], nDstPitch[2]);
  // constant name is Y U and V but for MVFrame this is good for RGB
  const MVPlaneSet mvplanes[3] = { YPLANE, UPLANE, VPLANE };

  for (int p = 0; p < planecount; ++p) {
    const MVPlaneSet plane = mvplanes[p];
    pSrcGOF->SetPlane(pSrc[p], nSrcPitch[p], plane);
  }

  // level 0 is padded first: with odd (chroma) dimensions the reduction to level 1 reads its padding
  pSrcGOF->Pad(nModeYUV);
  pSrcGOF->Reduce(nModeYUV);

  if (usePelClip)
  {
    MVFrame *srcFrames = pSrcGOF->GetFrame(0);

    for (int p = 0; p < planecount; ++p) {
      const MVPlaneSet plane = mvplanes[p];
      MVPlane *srcPlane = srcFrames->GetPlane(plane);

      if (nModeYUV & plane) {
        if (pixelsize == 1)
          srcPlane->RefineExt<uint8_t>(pSrcPel[p], nSrcPelPitch[p], isPelClipPadded);
        else if (pixelsize == 2)
          srcPlane->RefineExt<uint16_t>(pSrcPel[p], nSrcPelPitch[p], isPelClipPadded);
        else
          srcPlane->RefineExt<float>(pSrcPel[p], nSrcPelPitch[p], isPelClipPadded);
      }
    }
  }
  else
  {
    pSrcGOF->Refine(nModeYUV);
  }

  PROFILE_STOP(MOTION_PROFILE_INTERPOLATION);


  /*

  */

  PROFILE_CUMULATE();

  return dst;
}


// Rebuilds, from a lite super frame (picture inside its padding rows), the frame the normal MSuper makes. Same steps and
// same GroupOfFrames code as GetFrame / the constructor, so the result is bit-identical.
PVideoFrame MVSuper::ResolveToken(PVideoFrame lite, const VideoInfo &lite_vi, IScriptEnvironment* env)
{
  SuperParams64Bits params;
  memcpy(&params, &lite_vi.num_audio_samples, 8);
  const int hpad = params.nHPad, vpad = params.nVPad;
  const int pixelsize = lite_vi.ComponentSize();
  const int xr = 1 << lite_vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int yr = 1 << lite_vi.GetPlaneHeightSubsampling(PLANAR_U);
  const int pic_row = (lite_vi.width - 2 * hpad) * pixelsize;
  MvCudaTokenHeader tok;
  if (!GpuBackend::read_token(lite->GetReadPtr(PLANAR_Y) + size_t(vpad) * lite->GetPitch(PLANAR_Y) + size_t(hpad) * pixelsize, pic_row, tok))
    return lite;
  PVideoFrame pic = env->NewVideoFrame(lite_vi);
  const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  void *dst[3];
  int pitch[3];
  for (int p = 0; p < 3; ++p)
  {
    const int hp = p == 0 ? hpad : hpad / xr;
    const int vp = p == 0 ? vpad : vpad / yr;
    pitch[p] = pic->GetPitch(planes_l[p]);
    dst[p] = pic->GetWritePtr(planes_l[p]) + size_t(vp) * pitch[p] + size_t(hp) * pixelsize;
  }
  if (!GpuBackend::instance().download_token(tok, dst, pitch, env))
    env->ThrowError("MSuper: the GPU search clip frame %d is no longer available on the device", tok.frame);
  return pic;
}

PVideoFrame MVSuper::CompleteLite(PVideoFrame lite, const VideoInfo &lite_vi, bool mt_flag, IScriptEnvironment* env)
{
  lite = ResolveToken(lite, lite_vi, env);
  SuperParams64Bits params;
  memcpy(&params, &lite_vi.num_audio_samples, 8);
  const int rfilter = super_param_rfilter(params); // valid: MV_SUPER_IS_LITE checked by the caller
  const int sharp = super_param_sharp(params);
  const int hpad = params.nHPad, vpad = params.nVPad;

  bool v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { v8 = false; }

  const int xr = 1 << lite_vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int yr = 1 << lite_vi.GetPlaneHeightSubsampling(PLANAR_U);
  const int pixelsize = lite_vi.ComponentSize();
  const int bits = lite_vi.BitsPerComponent();
  const int width = lite_vi.width - 2 * hpad;
  const int super_width = lite_vi.width;

  VideoInfo vi = lite_vi;
  int super_height = PlaneSuperOffset(false, params.nHeight, params.nLevels, params.nPel, vpad, super_width * pixelsize, yr) / (super_width * pixelsize);
  if (yr == 2 && (super_height & 1)) super_height++;
  vi.height = super_height;
  PVideoFrame dst = v8 ? env->NewVideoFrameP(vi, &lite) : env->NewVideoFrame(vi);

  MVGroupOfFrames gof(params.nLevels, width, params.nHeight, params.nPel, hpad, vpad, YUVPLANES, env->GetCPUFlags(), xr, yr, pixelsize, bits, mt_flag);
  gof.set_interp((MVPlaneSet)params.nModeYUV, rfilter, sharp);

  const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  BYTE *pDst[3];
  int dpitch[3];
  const BYTE *pSrc[3];
  int spitch[3];
  for (int p = 0; p < 3; ++p)
  {
    pDst[p] = dst->GetWritePtr(planes_l[p]);
    dpitch[p] = dst->GetPitch(planes_l[p]);
    spitch[p] = lite->GetPitch(planes_l[p]);
    const int hp = p == 0 ? hpad : hpad / xr;
    const int vp = p == 0 ? vpad : vpad / yr;
    pSrc[p] = lite->GetReadPtr(planes_l[p]) + size_t(vp) * spitch[p] + size_t(hp) * pixelsize;
  }
  gof.Update(YUVPLANES, pDst[0], dpitch[0], pDst[1], dpitch[1], pDst[2], dpitch[2]);
  const MVPlaneSet planes[3] = { YPLANE, UPLANE, VPLANE };
  for (int p = 0; p < 3; ++p)
    gof.SetPlane(pSrc[p], spitch[p], planes[p]);
  gof.Reduce((MVPlaneSet)params.nModeYUV);
  gof.Pad((MVPlaneSet)params.nModeYUV);
  gof.Refine((MVPlaneSet)params.nModeYUV);
  return dst;
}
