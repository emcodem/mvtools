#ifndef __MV_FLOWFPS__
#define __MV_FLOWFPS__

#include "MVClip.h"
#include "MVFilter.h"
#include "MTSlicer.h"
#include "SimpleResize.h"
#include "yuy2planes.h"
#include <atomic>

class MVFlowFps
  : public GenericVideoFilter
  , public MVFilter
{
private:
  bool has_at_least_v8 = true;

  MVClip mvClipB;
  MVClip mvClipF;
  unsigned int numerator;
  unsigned int denominator;
  unsigned int numeratorOld;
  unsigned int denominatorOld;
  int maskmode;
  double ml;
  //bool isse;
  int cpuFlags;
  bool planar;
  bool blend;

  PClip finest; // v2.0

  int _instance_id; // debug unique id
  std::atomic<bool> reentrancy_check;
  int optDebug;

  int nleftLast;
  int nrightLast;

  int64_t fa, fb;

  // fullframe vector mask
  short *VXFullYB; //backward
  short *VXFullUVB;
  short *VYFullYB;
  short *VYFullUVB;
  short *VXFullYF;
  short *VXFullUVF;
  short *VYFullYF;
  short *VYFullUVF;
  short *VXFullYBB; //backward backward
  short *VXFullUVBB;
  short *VYFullYBB;
  short *VYFullUVBB;
  short *VXFullYFF; // forward forward
  short *VXFullUVFF;
  short *VYFullYFF;
  short *VYFullUVFF;

  // Small vector mask
  short *VXSmallYB;
  short *VXSmallUVB;
  short *VYSmallYB;
  short *VYSmallUVB;
  short *VXSmallYF;
  short *VXSmallUVF;
  short *VYSmallYF;
  short *VYSmallUVF;
  short *VXSmallYBB;
  short *VXSmallUVBB;
  short *VYSmallYBB;
  short *VYSmallUVBB;
  short *VXSmallYFF;
  short *VXSmallUVFF;
  short *VYSmallYFF;
  short *VYSmallUVFF;

  BYTE *MaskSmallB;
  BYTE *MaskFullYB;
  BYTE *MaskFullUVB;
  BYTE *MaskSmallF;
  BYTE *MaskFullYF;
  BYTE *MaskFullUVF;

  BYTE *SADMaskSmallB;
  BYTE *SADMaskSmallF;

  int nWidthUV;
  int nHeightUV;
  int	VPitchY, VPitchUV;

  int nBlkXP;
  int nBlkYP;
  int nWidthP;
  int nHeightP;
  int nWidthPUV;
  int nHeightPUV;

  int nHPaddingUV;
  int nVPaddingUV;

  SimpleResize *upsizer;
  SimpleResize *upsizerUV;

  YUY2Planes * DstPlanes;

  bool is444;
  bool isGrey;
  bool isRGB; // avs+ planar
  bool needDistinctChroma;

  int pixelsize_super;
  int bits_per_pixel_super;
  int pixelsize_super_shift;
  int planecount;
  int xRatioUVs[3];
  int yRatioUVs[3];
  int nLogxRatioUVs[3];
  int nLogyRatioUVs[3];

  typedef MTSlicer <MVFlowFps> Slicer;

  void interpolate_slice(Slicer::TaskData &td);
  template <typename pixel_t>
  void interpolate_rows(int y_beg, int y_end);

  bool _mt_flag;

  // Per-frame state read by interpolate_slice()
  BYTE *_cpPDst[3];
  const BYTE *_cpPRef[3];
  const BYTE *_cpPSrc[3];
  int _cpNDstPitches[3], _cpNRefPitches[3];
  int _cpNOffsetY, _cpNOffsetUV;
  int _cpTime256;
  int _cpMode; // 0: FlowInterSimple, 1: FlowInter, 2: FlowInterExtra

public:
  MVFlowFps(PClip _child, PClip _super, PClip _mvbw, PClip _mvfw, unsigned int _num, unsigned int _den, int _maskmode, double _ml,
    bool _blend, sad_t nSCD1, int nSCD2, bool isse, bool _planar, int _optDebug, bool mt_flag, IScriptEnvironment* env);
  ~MVFlowFps();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
  }

};

#endif
