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


#ifndef	__MV_SuperParams64Bits__
#define	__MV_SuperParams64Bits__



// SuperParams64Bits::param bits
// the source of MSuper was YUY2, the super clip is YV16 (Yuy2Planar.h); MShow converts its output back
constexpr unsigned char SUPER_PARAM_YUY2_SOURCE = 1;

// MVSuper parameters packed to 64 bit num_audio_samples
class SuperParams64Bits
{
public:
  uint16_t nHeight;
  unsigned char nHPad;
  unsigned char nVPad;
  unsigned char nPel;
  unsigned char nModeYUV;
  unsigned char nLevels;
  unsigned char param; // SUPER_PARAM_ flags (bit 0, upstream) and our MSuper settings (bits 1-7), see below
};

// Bits 1-7 of param (this fork; upstream uses only bit 0): MSuper's own settings, for the GPU paths of MAnalyse / MRecalculate / MFlowFps
// (they rebuild the sub-pel planes themselves, bit-identical only for matching settings) and for lite super clips.
//   bits 1-4  rfilter * 3 + sharp (rfilter 0..4, sharp 0..2: 15 values, 15 is never written)
//   bit 5     pelclip used
//   bit 6     lite (MSuper gpu=true: picture only)
//   bit 7     written by this MSuper: bits 1-6 are valid. A super clip of another mvtools2 build may carry anything
//             there (the original MSuper left param uninitialised), so nothing below trusts bits 1-6 without it.
constexpr unsigned char SUPER_PARAM_FILTER_SHIFT = 1;
constexpr unsigned char SUPER_PARAM_FILTER_MASK = 0x1E;
constexpr unsigned char SUPER_PARAM_PELCLIP = 0x20;
constexpr unsigned char SUPER_PARAM_LITE = 0x40;
constexpr unsigned char SUPER_PARAM_OURS = 0x80;

inline unsigned char super_param_make(bool yuy2_source, int rfilter, int sharp, bool pelclip, bool lite)
{
  if (rfilter < 0 || rfilter > 4 || sharp < 0 || sharp > 2) // not representable: "unknown", the GPU paths stay off
    return (unsigned char)(yuy2_source ? SUPER_PARAM_YUY2_SOURCE : 0);
  return (unsigned char)((yuy2_source ? SUPER_PARAM_YUY2_SOURCE : 0)
    | (((rfilter * 3 + sharp) << SUPER_PARAM_FILTER_SHIFT) & SUPER_PARAM_FILTER_MASK)
    | (pelclip ? SUPER_PARAM_PELCLIP : 0) | (lite ? SUPER_PARAM_LITE : 0) | SUPER_PARAM_OURS);
}
inline bool super_param_valid(const SuperParams64Bits &p)
{
  return (p.param & SUPER_PARAM_OURS) && ((p.param & SUPER_PARAM_FILTER_MASK) >> SUPER_PARAM_FILTER_SHIFT) < 15;
}
// only meaningful when super_param_valid
inline int super_param_rfilter(const SuperParams64Bits &p) { return ((p.param & SUPER_PARAM_FILTER_MASK) >> SUPER_PARAM_FILTER_SHIFT) / 3; }
inline int super_param_sharp(const SuperParams64Bits &p) { return ((p.param & SUPER_PARAM_FILTER_MASK) >> SUPER_PARAM_FILTER_SHIFT) % 3; }
inline bool super_param_pelclip(const SuperParams64Bits &p) { return (p.param & SUPER_PARAM_PELCLIP) != 0; }
// the GPU can rebuild this super clip's sub-pel planes itself: rfilter 2 or 4, sharp 2, no pelclip
inline bool super_param_gpu_rebuildable(const SuperParams64Bits &p)
{
  if (!super_param_valid(p)) return false;
  const int rf = super_param_rfilter(p);
  return (rf == 2 || rf == 4) && super_param_sharp(p) == 2 && !super_param_pelclip(p);
}

// MSuper(gpu=true) "lite" clips hold only the picture (no padding, pyramid or sub-pel planes); only MAnalyse, MRecalculate
// and MFlowFps know how to use them
#define MV_SUPER_IS_LITE(p) (super_param_valid(p) && ((p).param & SUPER_PARAM_LITE) != 0)
#define MV_REJECT_LITE_SUPER(p, who, env) \
  do { if (MV_SUPER_IS_LITE(p)) (env)->ThrowError(who ": the super clip was made with MSuper(gpu=true) (picture only, for MAnalyse / MRecalculate / MFlowFps); make a super clip without gpu=true for this filter"); } while (0)



#endif	// __MV_SuperParams64Bits__