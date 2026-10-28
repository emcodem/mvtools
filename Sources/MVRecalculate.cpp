// MVTools
// Recalculate motion data (based on MVAnalyse)
// Copyright(c)2008 A.G.Balakhnin aka Fizick

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

#include "def.h"
#include "AnaFlags.h"
#include "ClipFnc.h"
#include "cpu.h"
#include "DCTFFTW.h"
#include "DCTINT.h"
#include "MVClip.h"
#include "MVGroupOfFrames.h"
#include "MVRecalculate.h"
#include "profile.h"
#include "SuperParams64Bits.h"
#include "MVSuper.h"
#include "GoldenDump.h"
#include "GpuBackend.h"
#include "MVFrame.h"
#include "MVPlane.h"

#include	<algorithm>

#include <cmath>
#include <cstring>
#include <cstdio>

MVRecalculate::MVRecalculate(
  PClip _super, PClip _vectors, sad_t _thSAD, int _smooth,
  int _blksizex, int _blksizey, int st, int stp, int lambda, bool chroma,
  int _pnew, int _overlapx, int _overlapy, const char* _outfilename,
  int _dctmode, int _divide, int _sadx264, bool _isse, bool _meander,
  int trad, bool mt_flag, int _chromaSADScale, bool gpu_flag, IScriptEnvironment* env
)
  : GenericVideoFilter(_super)
  , _srd_arr()
  , _vectorfields_aptr()
  , _dct_factory_ptr()
  , _dct_pool()
  , _nbr_srd((trad > 0) ? trad * 2 : 1)
  , _mt_flag(mt_flag)
  , _gpu_ok(false)
  , _super_lite(false)
  , _gpu_slices(1)
  , _gpu_effective_chroma_scale(0)
{
  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  _srd_arr.resize(_nbr_srd);

  cpuFlags = _isse ? env->GetCPUFlags() : 0;

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  for (int srd_index = 0; srd_index < _nbr_srd; ++srd_index)
  {
    SrcRefData &	srd = _srd_arr[srd_index];
    const sad_t max_nSCD1 = 8 * 8 * (255 - 0); // normalized to blocksize and bits_per_pixel in MvClip
    srd._clip_sptr = SharedPtr <MVClip>(
      new MVClip(_vectors, max_nSCD1, 255, env, _nbr_srd, srd_index)
      );
  }

  MVAnalysisData &	analysisData = _srd_arr[0]._analysis_data;
  MVAnalysisData &	analysisDataDivided = _srd_arr[0]._analysis_data_divided;

  if (pixelsize == 4)
  {
    env->ThrowError("MRecalculate: Clip with float pixel type is not supported");
  }

  if (!vi.IsYUV() && !vi.IsYUY2()) // YUY2 is also YUV but let's see what is supported
  {
    env->ThrowError("MRecalculate: Clip must be YUV or YUY2");
  }

  vi.num_frames *= _nbr_srd;
  vi.MulDivFPS(_nbr_srd, 1);

  smooth = _smooth;

  SuperParams64Bits	params;
  memcpy(&params, &child->GetVideoInfo().num_audio_samples, 8);
  const int nHeight = params.nHeight;
  const int nSuperHPad = params.nHPad;
  const int nSuperVPad = params.nVPad;
  const int nSuperPel = params.nPel;
  const int nSuperModeYUV = params.nModeYUV;
  const int nSuperLevels = params.nLevels;
  _super_lite = MV_SUPER_IS_LITE(params);

  if (vi.IsY())
    chroma = false; // silent fallback

  nModeYUV = chroma ? YUVPLANES : YPLANE;
  if ((nModeYUV & nSuperModeYUV) != nModeYUV)
  {
    env->ThrowError(
      "MRecalculate: super clip does not contain needed color data"
    );
  }

#if !defined(MV_64BIT)
  MVAnalysisData *	pAnalyseFilter =
    reinterpret_cast <MVAnalysisData *> (_vectors->GetVideoInfo().nchannels);
#else
  // hack! 
  uintptr_t p = (((uintptr_t)(unsigned int)_vectors->GetVideoInfo().nchannels ^ 0x80000000) << 32) | (uintptr_t)(unsigned int)_vectors->GetVideoInfo().sample_type;
  MVAnalysisData *pAnalyseFilter = reinterpret_cast<MVAnalysisData *>(p);
#endif

  analysisData.nWidth = pAnalyseFilter->GetWidth();
  analysisData.nHeight = pAnalyseFilter->GetHeight();
  analysisData.pixelType = pAnalyseFilter->GetPixelType();
  if (!vi.IsY()) {
    analysisData.yRatioUV = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
    analysisData.xRatioUV = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
  }
  else {
    analysisData.yRatioUV = 1; // n/a
    analysisData.xRatioUV = 1; // n/a
  }
  analysisData.pixelsize = pixelsize;
  analysisData.bits_per_pixel = bits_per_pixel;

  if (_chromaSADScale<-2 || _chromaSADScale>2)
    env->ThrowError(
      "MRecalculate: scaleCSAD must be -2..2"
    );

  analysisData.chromaSADScale = _chromaSADScale;

  pSrcGOF = new MVGroupOfFrames(
    nSuperLevels, analysisData.nWidth, analysisData.nHeight,
    nSuperPel, nSuperHPad, nSuperVPad, nSuperModeYUV,
    cpuFlags, analysisData.xRatioUV, analysisData.yRatioUV, analysisData.pixelsize, analysisData.bits_per_pixel, mt_flag
  );
  pRefGOF = new MVGroupOfFrames(
    nSuperLevels, analysisData.nWidth, analysisData.nHeight,
    nSuperPel, nSuperHPad, nSuperVPad, nSuperModeYUV,
    cpuFlags, analysisData.xRatioUV, analysisData.yRatioUV, analysisData.pixelsize, analysisData.bits_per_pixel, mt_flag
  );
  const int nSuperWidth = child->GetVideoInfo().width;

  if (nHeight != analysisData.nHeight
    || nSuperWidth - 2 * nSuperHPad != analysisData.nWidth)
  {
    env->ThrowError("MRecalculate : wrong frame size");
  }
  if (vi.pixel_type != analysisData.pixelType)
  {
    env->ThrowError("MRecalculate: wrong pixel type");
  }

  analysisData.nBlkSizeX = _blksizex;
  analysisData.nBlkSizeY = _blksizey;
  // same blocksize check in MAnalyze and MRecalculate
  // some blocksizes may not work in 4:2:0 (chroma subsampling division), but o.k. in 4:4:4
  const std::vector< std::pair< int, int > > allowed_blksizes =
  {
    { 64, 64 },{ 64,48 },{ 64,32 },{ 64,16 },
    { 48,64 },{ 48,48 }, { 48,24 }, { 48,12 },
    { 32,64 },{ 32,32 },{ 32,24 },{ 32,16 },{ 32,8 },
    { 24,48 },{ 24,32 },{ 24,24 },{ 24,12 },{ 24,6 },
    { 16,64 },{ 16,32 },{ 16,16 },{ 16,12 },{ 16,8 },{ 16,4 },{ 16,2 },
    { 12,48 },{ 12,24 },{ 12,16 },{ 12,12 },{ 12,6 },{ 12,3 },
    { 8,32 },{ 8,16 },{ 8,8 },{ 8,4 },{ 8,2 },{ 8,1 },
    { 6,24 },{ 6,12 },{ 6,6 },{ 6,3 },
    { 4,8 },{ 4,4 },{ 4,2 },
    { 3,6 },{ 3,3 },
    { 2,4 },{ 2,2 }
  };
  bool found = false;
  for (int i = 0; i < (int)allowed_blksizes.size(); i++) {
    if (analysisData.nBlkSizeX == allowed_blksizes[i].first && analysisData.nBlkSizeY == allowed_blksizes[i].second) {
      found = true;
      break;
    }
  }
  if (!found) {
    env->ThrowError(
      "MRecalculate: Invalid block size: %d x %d", analysisData.nBlkSizeX, analysisData.nBlkSizeY);
  }

  analysisData.nPel = nSuperPel;
  analysisData.nDeltaFrame = pAnalyseFilter->GetDeltaFrame();
  analysisData.isBackward = pAnalyseFilter->IsBackward();

  if (_overlapx < 0 || _overlapx > _blksizex / 2
    || _overlapy < 0 || _overlapy > _blksizey / 2)
  {
    env->ThrowError("MRecalculate: overlap must be less or equal than half block size");
  }

  if (_overlapx % analysisData.xRatioUV || _overlapy % analysisData.yRatioUV)
  {
    env->ThrowError("MRecalculate: wrong overlap for the colorspace subsampling");
  }

  if (_divide != 0 && (_blksizex < 8 || _blksizey < 8)) // 2.5.11.22 || instead of &&
  {
    env->ThrowError(
      "MRecalculate: Block sizes must be 8 or more for divide mode"
    );
  }
  if (_divide != 0
    && ((_overlapx % (2 * analysisData.xRatioUV)) || (_overlapy % (2 * analysisData.yRatioUV))) // PF subsampling-aware
    )
  {
    env->ThrowError("MRecalculate: wrong overlap for the colorspace subsampling for divide mode");
  }

  divideExtra = _divide;

  // include itself, but usually equal to 256 :-)
  headerSize = std::max(int(4 + sizeof(analysisData)), 256);

  analysisData.nOverlapX = _overlapx;
  analysisData.nOverlapY = _overlapy;

  int		nBlkX = (analysisData.nWidth - analysisData.nOverlapX)
    / (analysisData.nBlkSizeX - analysisData.nOverlapX);
  int		nBlkY = (analysisData.nHeight - analysisData.nOverlapY)
    / (analysisData.nBlkSizeY - analysisData.nOverlapY);

  // 2.7.36: fallback to no overlap when either nBlk count is less than 2
  if (nBlkX < 2 || nBlkY < 2) {
    analysisData.nOverlapX = 0;
    analysisData.nOverlapY = 0;
    nBlkX = analysisData.nWidth / analysisData.nBlkSizeX;
    nBlkY = analysisData.nHeight / analysisData.nBlkSizeY;
  }

  analysisData.nBlkX = nBlkX;
  analysisData.nBlkY = nBlkY;
  analysisData.nLvCount = 1;


  nLambda = lambda;
  // lambda is finally scaled in PlaneOfBlocks::WorkingArea::MotionDistorsion(int vx, int vy) const
  // as return (nLambda * dist) >> (16 - bits_per_pixel)
  // To have it 8x8 normalized, we would use 
  //   nLambda = lambda * ((_blksizex * _blksizey) / (8 * 8)) << (bits_per_pixel-8);  // normalize to 8x8 block size
  // and use 
  //   (nLambda * dist) >> 8  in PlaneOfBlocks::WorkingArea::MotionDistorsion
  // and change default lambda generation in truemotion=true preset
  // But doing this would kill compatibility, there are scripts which are using lambda properly scaled by the block size.

  pnew = _pnew;
  meander = _meander;

  if (_dctmode != 0)
  {
    _dct_factory_ptr = std::unique_ptr <DCTFactory>(
      new DCTFactory(_dctmode, _isse, _blksizex, _blksizey, pixelsize, bits_per_pixel, *env)
      );
    _dct_pool.set_factory(*_dct_factory_ptr);
  }

  switch (st)
  {
  case 0:
    searchType = ONETIME;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 1:
    searchType = NSTEP;
    nSearchParam = (stp < 0) ? 0 : stp;
    break;
  case 3:
    searchType = EXHAUSTIVE;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 4:
    searchType = HEX2SEARCH;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 5:
    searchType = UMHSEARCH;
    nSearchParam = (stp < 1) ? 1 : stp; // really min is 4
    break;
  case 6:
    searchType = HSEARCH;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 7:
    searchType = VSEARCH;
    nSearchParam = (stp < 1) ? 1 : stp;
    break;
  case 2:
  default:
    searchType = LOGARITHMIC;
    nSearchParam = (stp < 1) ? 1 : stp;
  }

  analysisData.nFlags = 0;
  analysisData.nFlags |= (_isse) ? MOTION_USE_ISSE : 0;
  analysisData.nFlags |= (analysisData.isBackward) ? MOTION_IS_BACKWARD : 0;
  analysisData.nFlags |= (chroma) ? MOTION_USE_CHROMA_MOTION : 0;
  analysisData.nFlags |= conv_cpuf_flags_to_cpu(env->GetCPUFlags());

  _vectorfields_aptr = std::unique_ptr <GroupOfPlanes>(new GroupOfPlanes(
    analysisData.nBlkSizeX,
    analysisData.nBlkSizeY,
    analysisData.nLvCount,
    analysisData.nPel,
    analysisData.nFlags,
    analysisData.nOverlapX,
    analysisData.nOverlapY,
    analysisData.nBlkX,
    analysisData.nBlkY,
    analysisData.xRatioUV,
    analysisData.yRatioUV,
    divideExtra,
    analysisData.pixelsize,
    analysisData.bits_per_pixel,
    (_dct_factory_ptr.get() != 0) ? &_dct_pool : 0,
    _mt_flag,
    analysisData.chromaSADScale,
    env
  ));

  analysisData.nMagicKey = MVAnalysisData::MOTION_MAGIC_KEY;
  analysisData.nHPadding = nSuperHPad;
  analysisData.nVPadding = nSuperVPad;

  // MVAnalysisData and outfile format version: last update v1.8.1
  analysisData.nVersion = MVAnalysisData::VERSION;

  outfilename = _outfilename;
  if (lstrlen(outfilename) > 0)
  {
    outfile = fopen(outfilename, "wb");
    if (outfile == NULL)
    {
      env->ThrowError("MRecalculate: out file can not be created!");
    }
    else
    {
      fwrite(&analysisData, sizeof(analysisData), 1, outfile);
      // short vx, short vy, int SAD = 4 words = 8 bytes per block
      outfilebuf = new int16_t[nBlkX * nBlkY * (1+1+ sizeof(sad_t)/sizeof(int16_t))];
    }
  }
  else
  {
    outfile = NULL;
    outfilebuf = NULL;
  }

  // Defines the format of the output vector clip
  const int		width_bytes = headerSize + _vectorfields_aptr->GetArraySize() * 4;
  ClipFnc::format_vector_clip(
    vi, true, nBlkX, "rgb32", width_bytes, "MRecalculate", env
  );

  if (divideExtra)	//v1.8.1
  {
    memcpy(&analysisDataDivided, &analysisData, sizeof(analysisData));
    analysisDataDivided.nBlkX = analysisData.nBlkX * 2;
    analysisDataDivided.nBlkY = analysisData.nBlkY * 2;
    analysisDataDivided.nBlkSizeX = analysisData.nBlkSizeX / 2;
    analysisDataDivided.nBlkSizeY = analysisData.nBlkSizeY / 2;
    analysisDataDivided.nOverlapX = analysisData.nOverlapX / 2;
    analysisDataDivided.nOverlapY = analysisData.nOverlapY / 2;
    analysisDataDivided.nLvCount = analysisData.nLvCount + 1;
#if !defined(MV_64BIT)
    vi.nchannels = reinterpret_cast <uintptr_t> (&analysisDataDivided);
#else
    uintptr_t p = reinterpret_cast <uintptr_t> (&analysisDataDivided);
    vi.nchannels = 0x80000000L | (int)(p >> 32);
    vi.sample_type = (int)(p & 0xffffffffUL);
#endif
  }
  else
  {
    // we'll transmit to the processing filters a handle
    // on the analyzing filter itself ( it's own pointer ), in order
    // to activate the right parameters.
#if !defined(MV_64BIT)
    vi.nchannels = reinterpret_cast <uintptr_t> (&analysisData);
#else
    uintptr_t p = reinterpret_cast <uintptr_t> (&analysisData);
    vi.nchannels = 0x80000000L | (int)(p >> 32);
    vi.sample_type = (int)(p & 0xffffffffUL);
#endif
  }

  for (int srd_index = 1; srd_index < _nbr_srd; ++srd_index)
  {
    SrcRefData &	srd = _srd_arr[srd_index];
    srd._analysis_data = analysisData;
    srd._analysis_data_divided = analysisDataDivided;
  }

  // ---- GPU recalculation: only for the configuration the GPU code reproduces bit-exactly (GPU_PLAN.md)
  if (gpu_flag)
  {
    const bool super_ok = (params.param & 0x80) && (params.param & 7) == 4 && ((params.param >> 3) & 3) == 2
      && !(params.param & 0x20);
    const int xr = analysisData.xRatioUV, yr = analysisData.yRatioUV;
    _gpu_ok = GpuBackend::instance().usable() && super_ok && (pixelsize == 1 || pixelsize == 2) && !child->GetVideoInfo().IsYUY2() && child->GetVideoInfo().IsPlanar()
      && chroma && nModeYUV == YUVPLANES && !divideExtra && !_dctmode && _nbr_srd == 1 && searchType == EXHAUSTIVE
      && nSearchParam >= 1 && nSearchParam <= 3 && (nSuperPel == 1 || nSuperPel == 2) && smooth >= 0 && smooth <= 1
      && analysisData.nBlkSizeX <= 32 && (xr == 1 || xr == 2) && (yr == 1 || yr == 2) && lstrlen(_outfilename) == 0;
    if (GpuBackend::log_enabled())
    {
      std::fprintf(stderr, "MRecalculate gpu requested: %s\n", _gpu_ok ? "using the GPU" : "not supported here, using the CPU");
      if (!_gpu_ok)
        std::fprintf(stderr,
          "  usable %d super_ok %d (param 0x%x) pixelsize %d yuy2 %d planar %d chroma %d modeYUV %d divide %d nbr_srd %d "
          "searchType %d searchParam %d superPel %d smooth %d blkX %d outfile %d\n",
          GpuBackend::instance().usable() ? 1 : 0, super_ok ? 1 : 0, params.param, pixelsize,
          child->GetVideoInfo().IsYUY2() ? 1 : 0, child->GetVideoInfo().IsPlanar() ? 1 : 0, chroma ? 1 : 0, nModeYUV, divideExtra, _nbr_srd, (int)searchType, nSearchParam, nSuperPel,
          smooth, analysisData.nBlkSizeX, (int)lstrlen(_outfilename));
    }
    if (_gpu_ok)
    {
      _gpu_slices = GpuBackend::slice_threads(mt_flag);
      int lx = 0, ly = 0;
      while ((1 << lx) < xr) lx++;
      while ((1 << ly) < yr) ly++;
      _gpu_effective_chroma_scale = (2 - (lx + ly)) - analysisData.chromaSADScale;
      GpuBackend::instance().add_user();
    }
  }

  // normalize threshold to block size
    // PF _thSAD is originally on 255 scale, for 16 bit we have to scale it.
  thSAD = _thSAD;
  if (pixelsize == 2)
    thSAD = (sad_t)(thSAD * (1 << (bits_per_pixel - 8))); // todo float
// luma only
  thSAD =
    (uint64_t)thSAD
    * (analysisData.nBlkSizeX * analysisData.nBlkSizeY)
    / (8 * 8);  // normalize to 8x8 block size
  if (chroma)
  {
    thSAD += ScaleSadChroma(thSAD * 2, analysisData.chromaSADScale) / 4; // base: YV12
    //thSAD += thSAD / (analysisData.xRatioUV * analysisData.yRatioUV) * 2; // *2: two additional planes: UV
  }
}



MVRecalculate::~MVRecalculate()
{
  if (_gpu_ok)
    GpuBackend::instance().remove_user();
  if (outfile != NULL)
  {
    fclose(outfile);
    outfile = 0;
    delete[] outfilebuf;
    outfilebuf = 0;
  }

  delete pSrcGOF;
  pSrcGOF = 0;
  delete pRefGOF;
  pRefGOF = 0;
}



PVideoFrame __stdcall MVRecalculate::GetFrame(int n, IScriptEnvironment* env)
{
  const int		nsrc = n / _nbr_srd;
  const int		srd_index = n % _nbr_srd;

  SrcRefData &	srd = _srd_arr[srd_index];

  // get pointer to vectors
  ::PVideoFrame	mvn;
  {
    GpuScope sc("recalc: vector clip GetFrame+Update");
    mvn = srd._clip_sptr->GetFrame(nsrc, env);
    srd._clip_sptr->Update(mvn, env);	// force calulation of vectors
  }

  srd._analysis_data.nDeltaFrame = srd._clip_sptr->GetDeltaFrame();
  srd._analysis_data.isBackward = srd._clip_sptr->IsBackward();

  srd._analysis_data_divided.nDeltaFrame = srd._analysis_data.nDeltaFrame;
  srd._analysis_data_divided.isBackward = srd._analysis_data.isBackward;
  /* 2.5.11.3:
    int minframe = (analysisData.isBackward) ? 0 : analysisData.nDeltaFrame;
    int maxframe = (analysisData.isBackward) ? vi.num_frames - analysisData.nDeltaFrame : vi.num_frames;
    int offset = (analysisData.isBackward) ? analysisData.nDeltaFrame : -analysisData.nDeltaFrame;
     2.5.11.22
    int nref, minframe, maxframe;
    int off = analysisData.nDeltaFrame;
    if (off > 0)
    {
      minframe = ( analysisData.isBackward ) ? 0 : off;
      maxframe = ( analysisData.isBackward ) ? vi.num_frames - off : vi.num_frames;
      nref = ( analysisData.isBackward ) ? n + off : n - off;
    }
    else
    {
      minframe = 0;
      maxframe = vi.num_frames;
      nref = -off; // static mode
    }
  */
  // PF: seems that 2.6.0.5 already OK, works as 2.5.11.22
  const int		nbr_src_frames = child->GetVideoInfo().num_frames;
  int				minframe;
  int				maxframe;
  int				nref;
  if (srd._analysis_data.nDeltaFrame > 0)
  {
    const int		offset =
      (srd._analysis_data.isBackward)
      ? srd._analysis_data.nDeltaFrame
      : -srd._analysis_data.nDeltaFrame;
    minframe = std::max(-offset, 0);
    maxframe = nbr_src_frames + std::min(-offset, 0);
    nref = nsrc + offset;
  }
  else // special static mode
  {
    nref = -srd._analysis_data.nDeltaFrame;	// positive fixed frame number
    minframe = 0;
    maxframe = nbr_src_frames;
  }

  PVideoFrame dst = env->NewVideoFrame(vi); // frame prop copy later if needed
  unsigned char *	pDst = dst->GetWritePtr();

  // write analysis parameters as a header to frame
  memcpy(pDst, &headerSize, sizeof(int));
  if (divideExtra)
  {
    memcpy(
      pDst + sizeof(int),
      &srd._analysis_data_divided,
      sizeof(srd._analysis_data_divided)
    );
  }
  else
  {
    memcpy(
      pDst + sizeof(int),
      &srd._analysis_data,
      sizeof(srd._analysis_data)
    );
  }
  pDst += headerSize;

  if (!srd._clip_sptr->IsUsable() || nsrc < minframe || nsrc >= maxframe)
  {
    _vectorfields_aptr->WriteDefaultToArray(reinterpret_cast <int *> (pDst));
  }

  else
  {
    //		DebugPrintf ("MVRecalculate: Get src frame %d",nsrc);
    PVideoFrame src;
    {
      GpuScope sc("recalc: child GetFrame x2 (CPU MSuper)");
      src = child->GetFrame(nsrc, env); // v2.0
    }
    if (has_at_least_v8) env->copyFrameProps(src, dst); // frame prop copy support v8
    load_src_frame(*pSrcGOF, src, srd._analysis_data);

    //		DebugPrintf("MVRecalculate: Get ref frame %d", nref);
    ::PVideoFrame	ref;
    {
      GpuScope sc("recalc: child GetFrame x2 (CPU MSuper)");
      ref = child->GetFrame(nref, env); // v2.0
    }
    load_src_frame(*pRefGOF, ref, srd._analysis_data);

    const int		fieldShift = ClipFnc::compute_fieldshift(
      child,
      vi.IsFieldBased(),
      srd._analysis_data.nPel,
      nsrc,
      nref
    );

    if (outfile != NULL)
    {
      fwrite(&n, sizeof(int), 1, outfile);	// write frame number
    }

#ifdef MV_GOLDEN_DUMP
    const bool golden_on = golden::begin("MRecalculate", nsrc, srd._analysis_data.isBackward);
    if (golden_on)
    {
      const MVAnalysisData &a = srd._analysis_data;
      golden::logf("call n %d nsrc %d nref %d delta %d backward %d fieldShift %d", n, nsrc, nref, a.nDeltaFrame, a.isBackward ? 1 : 0, fieldShift);
      golden::logf("analysis width %d height %d blksizeX %d blksizeY %d overlapX %d overlapY %d blkX %d blkY %d levels %d pel %d",
        a.nWidth, a.nHeight, a.nBlkSizeX, a.nBlkSizeY, a.nOverlapX, a.nOverlapY, a.nBlkX, a.nBlkY, a.nLvCount, a.nPel);
      golden::logf("analysis flags 0x%x xRatioUV %d yRatioUV %d pixelsize %d bits %d chromaSADScale %d hpad %d vpad %d pixelType 0x%x",
        a.nFlags, a.xRatioUV, a.yRatioUV, a.pixelsize, a.bits_per_pixel, a.chromaSADScale, a.nHPadding, a.nVPadding, a.pixelType);
      golden::logf("recalc lambda %d lsad(uninitialised member, unused) pnew %d thSAD %d smooth %d searchType %d searchParam %d meander %d divide %d modeYUV %d mt %d",
        nLambda, pnew, (int)thSAD, smooth, (int)searchType, nSearchParam, meander ? 1 : 0, divideExtra, nModeYUV, _mt_flag ? 1 : 0);
      // input vectors: the frame the MVClip was built from (Analyse output), header included
      // payload only (first word = its own size in int32 words): the frame header holds uninitialised struct
      // bytes (80 of them differ between identical runs), so it is neither reproducible nor needed
      {
        const unsigned char *vp = mvn->GetReadPtr();
        const int hdr = *reinterpret_cast<const int *>(vp);
        const int words = *reinterpret_cast<const int *>(vp + hdr);
        golden::write_bin("vec_in.bin", vp + hdr, size_t(words) * 4);
      }
      golden::dump_gof("src", *pSrcGOF, 1, nModeYUV == YUVPLANES);
      golden::dump_gof("ref", *pRefGOF, 1, nModeYUV == YUVPLANES);
    }
#endif

    bool gpu_done = false;
    if (_gpu_ok && fieldShift == 0 && outfile == NULL)
    {
      gpu_done = gpu_recalc(srd, nsrc, nref, reinterpret_cast <int *> (pDst));
    }
    if (!gpu_done)
    {
    if (_super_lite)
    {
      GpuScope sc_c("recalc: complete lite super (CPU fallback)");
      src = MVSuper::CompleteLite(src, child->GetVideoInfo(), _mt_flag, env);
      load_src_frame(*pSrcGOF, src, srd._analysis_data);
      ref = MVSuper::CompleteLite(ref, child->GetVideoInfo(), _mt_flag, env);
      load_src_frame(*pRefGOF, ref, srd._analysis_data);
    }
    _vectorfields_aptr->RecalculateMVs(
      *(srd._clip_sptr), pSrcGOF, pRefGOF,
      searchType, nSearchParam, nLambda, lsad, pnew,
      srd._analysis_data.nFlags, reinterpret_cast <int *> (pDst),
      outfilebuf, fieldShift, thSAD, smooth, meander
    );
    }

#ifdef MV_GOLDEN_DUMP
    if (golden_on)
    {
      golden::write_bin("vec_out.bin", pDst, size_t(_vectorfields_aptr->GetArraySize()) * 4);
      golden::end();
    }
#endif

    if (divideExtra)
    {
      // make extra level with divided sublocks with median (not estimated)
      // motion
      _vectorfields_aptr->ExtraDivide(
        reinterpret_cast <int *> (pDst),
        srd._analysis_data.nFlags
      );
    }

    //		PROFILE_CUMULATE ();
    if (outfile != NULL)
    {
      fwrite(
        outfilebuf,
        sizeof(int16_t) * 4 * srd._analysis_data.nBlkX
        * srd._analysis_data.nBlkY,
        1,
        outfile
      );
    }
  }

  return dst;
}




// GPU version of GroupOfPlanes::RecalculateMVs for the frames loaded into pSrcGOF / pRefGOF.
bool MVRecalculate::gpu_recalc(SrcRefData &srd, int nsrc, int nref, int *payload)
{
  if (GpuBackend::force_fallback())
    return false;
  GpuScope sc_total("recalc: gpu_recalc total");
  GpuBackend &gpu = GpuBackend::instance();
  const MVAnalysisData &a = srd._analysis_data;

  const MVPlaneSet sets[3] = { YPLANE, UPLANE, VPLANE };
  const void *src_p[3], *ref_p[3];
  int src_pitch[3], ref_pitch[3];
  for (int p = 0; p < 3; p++)
  {
    MVPlane *ps = pSrcGOF->GetFrame(0)->GetPlane(sets[p]);
    MVPlane *pr = pRefGOF->GetFrame(0)->GetPlane(sets[p]);
    src_p[p] = ps->GetAbsolutePelPointer(ps->GetHPadding(), ps->GetVPadding());
    ref_p[p] = pr->GetAbsolutePelPointer(pr->GetHPadding(), pr->GetVPadding());
    src_pitch[p] = ps->GetPitch();
    ref_pitch[p] = pr->GetPitch();
  }

  // the old vectors: finest plane of the vector clip, (vx, vy, sad) per block
  const FakePlaneOfBlocks &old = srd._clip_sptr->GetPlane(0);
  const int n_old = old.GetReducedWidth() * old.GetReducedHeight();
  std::vector<int> old_vec(size_t(n_old) * 3);
  for (int i = 0; i < n_old; i++)
  {
    const VECTOR v = old.GetBlock(i).GetMV();
    old_vec[3 * i + 0] = v.x;
    old_vec[3 * i + 1] = v.y;
    old_vec[3 * i + 2] = (int)v.sad;
  }

  MvCudaSuperParams sp = {};
  sp.width = a.nWidth;
  sp.height = a.nHeight;
  sp.x_ratio_uv = a.xRatioUV;
  sp.y_ratio_uv = a.yRatioUV;
  sp.hpad = a.nHPadding;
  sp.vpad = a.nVPadding;
  sp.pel = a.nPel;
  sp.levels = 1;
  sp.bits_per_pixel = a.bits_per_pixel;
  sp.chroma = 1;
  sp.rfilter = 4;
  sp.sharp = 2;

  MvCudaRecalcParams rp = {};
  rp.blk_x = a.nBlkSizeX;
  rp.blk_y = a.nBlkSizeY;
  rp.overlap_x = a.nOverlapX;
  rp.overlap_y = a.nOverlapY;
  rp.n_blk_x = a.nBlkX;
  rp.n_blk_y = a.nBlkY;
  rp.old_blk_x = old.GetReducedWidth();
  rp.old_blk_y = old.GetReducedHeight();
  rp.old_size_x = old.GetBlockSizeX();
  rp.old_size_y = old.GetBlockSizeY();
  rp.old_overlap_x = old.GetOverlapX();
  rp.old_overlap_y = old.GetOverlapY();
  rp.old_pel = old.GetPel();
  rp.lambda_level = nLambda / (a.nPel * a.nPel);
  rp.pnew = pnew;
  rp.th_sad = (int)thSAD;
  rp.smooth = smooth;
  rp.search_radius = nSearchParam;
  rp.chroma_sad_scale = _gpu_effective_chroma_scale;
  rp.slice_threads = _gpu_slices;
  rp.src_key = GpuBackend::frame_key((const void *)child.operator->(), nsrc);
  rp.ref_key = GpuBackend::frame_key((const void *)child.operator->(), nref);

  const int n_new = a.nBlkX * a.nBlkY;
  std::vector<int> out_vec(size_t(n_new) * 3);
  int rc = -1;
  try
  {
    GpuScope sc_call("recalc: recalc_host call");
    rc = gpu.recalc_host(&sp, &rp, src_p, src_pitch, ref_p, ref_pitch, old_vec.data(), out_vec.data());
  }
  catch (...)
  {
    rc = -2;
  }
  if (GpuBackend::log_enabled())
    std::fprintf(stderr, "MRecalculate gpu frame %d rc=%d%s\n", nsrc, rc, rc == 0 ? "" : " -> CPU");
  if (rc != 0)
    return false;

  // payload as GroupOfPlanes::RecalculateMVs writes it: [total words, validity], [plane words, triples...]
  payload[0] = _vectorfields_aptr->GetArraySize();
  payload[1] = 1;
  payload[2] = n_new * 3 + 1;
  std::memcpy(payload + 3, out_vec.data(), out_vec.size() * sizeof(int));
  return true;
}

void	MVRecalculate::load_src_frame(MVGroupOfFrames &gof, ::PVideoFrame &src, const MVAnalysisData &ana_data)
{
  PROFILE_START(MOTION_PROFILE_YUY2CONVERT);
  const unsigned char *	pSrcY;
  const unsigned char *	pSrcU;
  const unsigned char *	pSrcV;
  int				nSrcPitchY;
  int				nSrcPitchUV;
  if ((ana_data.pixelType & VideoInfo::CS_YUY2) == VideoInfo::CS_YUY2)
  {
    // planar data packed to interleaved format (same as interleved2planar
    // by kassandro) - v2.0.0.5
    pSrcY = src->GetReadPtr();
    pSrcU = pSrcY + src->GetRowSize() / 2;
    pSrcV = pSrcU + src->GetRowSize() / 4;
    nSrcPitchY = src->GetPitch();
    nSrcPitchUV = nSrcPitchY;
  }
  else
  {
    pSrcY = src->GetReadPtr(PLANAR_Y);
    pSrcU = src->GetReadPtr(PLANAR_U);
    pSrcV = src->GetReadPtr(PLANAR_V);
    nSrcPitchY = src->GetPitch(PLANAR_Y);
    nSrcPitchUV = src->GetPitch(PLANAR_U);
  }
  PROFILE_STOP(MOTION_PROFILE_YUY2CONVERT);

  gof.Update(
    nModeYUV,
    (BYTE*)pSrcY, nSrcPitchY,
    (BYTE*)pSrcU, nSrcPitchUV,
    (BYTE*)pSrcV, nSrcPitchUV
  ); // v2.0
}

