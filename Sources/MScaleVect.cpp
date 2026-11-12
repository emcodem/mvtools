// MScaleVectors Class
// Function for the MVExtras plugin
// By Vit, 2011
//
// Scale MVTools motion vectors. Can scale the blocks themselves to create vectors for a different frame size (powers of 2 only)

#include "MScaleVect.h"
#include "VECTOR.h"
#include <cmath>

// Constructor - Copy motion vector information. Scale if required for use on different sized frame
MScaleVect::MScaleVect( PClip Child, double ScaleX, double ScaleY, ScaleMode Mode, bool Flip, bool AdjustSubpel, int Bits, IScriptEnvironment* Env )
   : GenericVideoFilter( Child ), mMode(Mode), mScaleX(ScaleX), mScaleY(ScaleY), mAdjustSubpel(AdjustSubpel), mRevert (false), mNewBits(Bits)
{
  has_at_least_v8 = true;
  try { Env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  // Get vector data
  if (vi.nchannels >= 0 &&  vi.nchannels < 9) 
    Env->ThrowError("MScaleVect: Clip does not contain motion vectors");
#if !defined(MV_64BIT)
    mVectorsInfo = *reinterpret_cast<MVAnalysisData *>(vi.nchannels);
#else
    uintptr_t p = (((uintptr_t)(unsigned int)vi.nchannels ^ 0x80000000) << 32) | (uintptr_t)(unsigned int)vi.sample_type;
    mVectorsInfo = *reinterpret_cast<MVAnalysisData *>(p);
#endif
  if (mVectorsInfo.nMagicKey != MVAnalysisData::MOTION_MAGIC_KEY || mVectorsInfo.nVersion != MVAnalysisData::VERSION) 
    Env->ThrowError("MScaleVect: Clip does not contain motion vectors");
#if !defined(MV_64BIT)
  vi.nchannels = reinterpret_cast <uintptr_t> (&mVectorsInfo);
#else
  p = reinterpret_cast <uintptr_t> (&mVectorsInfo);
  vi.nchannels = 0x80000000L | (int)(p >> 32);
  vi.sample_type = (int)(p & 0xffffffffUL);
#endif

  // bit depth adjustments
  if (mNewBits != 0 && mNewBits != 8 && mNewBits != 10 && mNewBits != 12 && mNewBits != 14 && mNewBits != 16)
    Env->ThrowError("MScaleVect: Bits must be 0 (no change), 8, 10, 12, 14 or 16");

  currentBits = mVectorsInfo.bits_per_pixel;
  changeBitDepth = mNewBits != 0 && currentBits != mNewBits;
  if (changeBitDepth) {
    VideoInfo vi;
    vi.pixel_type = mVectorsInfo.pixelType;
    if(vi.IsYUY2())
      Env->ThrowError("MScaleVect: Bits must be 8 for YUY2 based vectors");
    if (vi.IsYV411()) // not supported but for the sake of completeness
      Env->ThrowError("MScaleVect: Bits must be 8 for 4:1:1 clip based vectors");

    // change only bit depth within video format. Not nice but no helper interface for it.
    int newtype = vi.pixel_type;
    if (vi.IsYV12())
      newtype = VideoInfo::CS_YV12;

    newtype = (newtype & ~VideoInfo::CS_Sample_Bits_Mask);
    switch (mNewBits) {
    case 8: newtype |= VideoInfo::CS_Sample_Bits_8; break;
    case 10: newtype |= VideoInfo::CS_Sample_Bits_10; break;
    case 12: newtype |= VideoInfo::CS_Sample_Bits_12; break;
    case 14: newtype |= VideoInfo::CS_Sample_Bits_14; break;
    case 16: newtype |= VideoInfo::CS_Sample_Bits_16; break;
    case 32: newtype |= VideoInfo::CS_Sample_Bits_32; break; // perhaps in the future
    }

    bitDiff = std::abs(currentBits - mNewBits);
    big_pixel_sad = 1 << mNewBits; // for internal use

    mVectorsInfo.bits_per_pixel = mNewBits;
    mVectorsInfo.pixelsize = mNewBits == 8 ? sizeof(uint8_t) : sizeof(uint16_t);
    mVectorsInfo.pixelType = newtype;
  }
  else {
    big_pixel_sad = 1 << currentBits;
  }


  // Scale appropriate fields for different sized frame
  if (mMode == IncreaseBlockSize || mMode == DecreaseBlockSize)
  {
    // Check for valid scale value. [Tests for equality on integers directly stored in floats are OK]
    if (mScaleX != 1.0 && mScaleX != 2.0 && mScaleX != 4.0 && mScaleX != 8.0)
      Env->ThrowError("MScaleVect: Scale must be 1, 2, 4 or 8 for modes 0 or 1");
    if (mScaleY != 1.0 && mScaleY != 2.0 && mScaleY != 4.0 && mScaleY != 8.0)
      Env->ThrowError("MScaleVect: ScaleV must be 1, 2, 4 or 8 for modes 0 or 1");

    // Switch to integer scaling here for simpler code
    int iScaleX = static_cast<int>(mScaleX);
    int iScaleY = static_cast<int>(mScaleY);
    if (mMode == IncreaseBlockSize)
    {
      // Scale to larger frame, check that result will be valid
      if (mVectorsInfo.nBlkSizeX * iScaleX > MAX_BLOCK_SIZE || mVectorsInfo.nBlkSizeY * iScaleY > MAX_BLOCK_SIZE)
        Env->ThrowError("MScaleVect: Scaling creates too large a blocksize");

      mVectorsInfo.nBlkSizeX = mVectorsInfo.nBlkSizeX * iScaleX;
      mVectorsInfo.nBlkSizeY = mVectorsInfo.nBlkSizeY * iScaleY;
      mVectorsInfo.nOverlapX = mVectorsInfo.nOverlapX * iScaleX;
      mVectorsInfo.nOverlapY = mVectorsInfo.nOverlapY * iScaleY;
      mVectorsInfo.nWidth    = mVectorsInfo.nWidth    * iScaleX;
      mVectorsInfo.nHeight   = mVectorsInfo.nHeight   * iScaleY;
      mVectorsInfo.nHPadding = mVectorsInfo.nHPadding * iScaleX;
      mVectorsInfo.nVPadding = mVectorsInfo.nVPadding * iScaleY;

      if (mAdjustSubpel) 
      {
        if (mScaleX != mScaleY)
          Env->ThrowError("MScaleVect: Horizontal and vertical scale must be the same when AdjustSubPel=true");
        mVectorsInfo.nPel /= iScaleX;
        if (mVectorsInfo.nPel < 1)
          Env->ThrowError("MScaleVect: Scaling takes Subpel out of range (use AdjustSubpel=false)");
      }
    }
    else
    {
      // Scale to smaller frame, check that result will be valid
      if (mVectorsInfo.nBlkSizeX / iScaleX < 4 || mVectorsInfo.nBlkSizeY / iScaleY < 4)
        Env->ThrowError("MScaleVect: Scaling creates too small a blocksize");
      if (mVectorsInfo.nOverlapX % iScaleX != 0 || mVectorsInfo.nOverlapY % iScaleY != 0 )
        Env->ThrowError("MScaleVect: Scaling creates non-integer overlap");
      if (mVectorsInfo.nWidth % iScaleX != 0 || mVectorsInfo.nHeight % iScaleY != 0 )
        Env->ThrowError("MScaleVect: Scaling creates non-integer frame dimensions");
      if (mVectorsInfo.nHPadding % iScaleX != 0 || mVectorsInfo.nVPadding % iScaleY != 0 )
        Env->ThrowError("MScaleVect: Scaling creates non-integer padding");

      mVectorsInfo.nBlkSizeX = mVectorsInfo.nBlkSizeX / iScaleX;
      mVectorsInfo.nBlkSizeY = mVectorsInfo.nBlkSizeY / iScaleY;
      mVectorsInfo.nOverlapX = mVectorsInfo.nOverlapX / iScaleX;
      mVectorsInfo.nOverlapY = mVectorsInfo.nOverlapY / iScaleY;
      mVectorsInfo.nWidth    = mVectorsInfo.nWidth    / iScaleX;
      mVectorsInfo.nHeight   = mVectorsInfo.nHeight   / iScaleY;
      mVectorsInfo.nHPadding = mVectorsInfo.nHPadding / iScaleX;
      mVectorsInfo.nVPadding = mVectorsInfo.nVPadding / iScaleY;

      if (mAdjustSubpel)
      {
        if (mScaleX != mScaleY)
          Env->ThrowError("MScaleVect: Horizontal and vertical scale must be the same when AdjustSubPel=true");
        mVectorsInfo.nPel *= iScaleX;
        if (mVectorsInfo.nPel > 4)
          Env->ThrowError("MScaleVect: Scaling takes Subpel out of range (use AdjustSubpel=false)");
      }

      // Store actual scaling used for vectors
      mScaleX = 1.0f / mScaleX;
      mScaleY = 1.0f / mScaleY;
    }
  }

  // Scale mode = VectorsOnly, flip the backwards flag as required
  else if (mMode == VectorsOnly && Flip)
  {
    mVectorsInfo.isBackward = !mVectorsInfo.isBackward;
    mRevert = true;
  }
}

  
// Filter operation - scale the per-frame vectors themselves
PVideoFrame __stdcall MScaleVect::GetFrame( int FrameNum, IScriptEnvironment* Env )
{
  PVideoFrame src = child->GetFrame(FrameNum, Env);
  PVideoFrame dst = has_at_least_v8 ? Env->NewVideoFrameP(vi, &src) : Env->NewVideoFrame(vi); // frame property support
  const int* pData = reinterpret_cast<const int*>(src->GetReadPtr());
  int* pDst = reinterpret_cast<int*>(dst->GetWritePtr());

  // Copy and fix header
  int headerSize = *pData;
  memcpy(pDst, pData, headerSize);

  const MVAnalysisData &	hdr_src =
    *reinterpret_cast <const MVAnalysisData *> (pData + 1);
  MVAnalysisData &	hdr_dst =
    *reinterpret_cast <MVAnalysisData *> (pDst + 1);

  // Use the main header as default value
  // Copy delta and direction from the original frame (for multi-vector clips)
  // Fix the direction if required
  hdr_dst             = mVectorsInfo;
  hdr_dst.nDeltaFrame = hdr_src.nDeltaFrame;
  hdr_dst.isBackward  = (!! hdr_src.isBackward) ^ mRevert;

  // Copy all planes
  pData = reinterpret_cast<const int*>(reinterpret_cast<const char*>(pData) + headerSize);
  pDst  = reinterpret_cast<int*>(reinterpret_cast<char*>(pDst) + headerSize);
  memcpy(pDst, pData, *pData * sizeof(int));

  // Size and validity of all block data
  int* pPlanes = pDst;
  if (pPlanes[1] == 0) return dst; // Marked invalid
  int* pEnd = pPlanes + *pPlanes;
  pPlanes += 2;

  // Length of the plane that starts at p, in ints, including the length word itself. MAnalyse / MRecalculate with
  // divide > 0 append the divided sub-blocks as an extra finest level whose length word is the marker 0xFFFFFFFF
  // (its data runs to the end of the array). Adding the marker as a length stepped back one int and walked off the
  // frame (access violation).
  auto plane_len = [pEnd](const int* p) -> int {
    return (*p == -1) ? static_cast<int>(pEnd - p) : *p;
  };
  bool divided = false;
  for (const int* p = pPlanes; p < pEnd; p += plane_len(p))
    if (*p == -1) { divided = true; break; }

  // Changing blocksize is straightforward since scaled vectors are guaranteed to be valid with scaled blocksizes
  if (mMode == IncreaseBlockSize || mMode == DecreaseBlockSize)
  {
    if (mScaleX == 1.0 && mScaleY == 1.0) {
      // special case: no scale, maybe bit depth change?
      if (changeBitDepth) {
        while (pPlanes != pEnd)
        {
          // Scale each block's vector & SAD
          int blocksSize = plane_len(pPlanes);
          VECTOR* pBlocks = reinterpret_cast<VECTOR*>(pPlanes + 1);
          pPlanes += blocksSize;
          while (reinterpret_cast<int*>(pBlocks) != pPlanes)
          {
            if (currentBits < mNewBits)
              pBlocks->sad <<= bitDiff;
            else
              pBlocks->sad = (pBlocks->sad + (1 << (bitDiff - 1))) >> bitDiff; // round and shift
            pBlocks++;
          }
        }
      }
    }
    else {
      while (pPlanes != pEnd)
      {
        // Scale each block's vector & SAD
        int blocksSize = plane_len(pPlanes);
        VECTOR* pBlocks = reinterpret_cast<VECTOR*>(pPlanes + 1);
        pPlanes += blocksSize;
        while (reinterpret_cast<int*>(pBlocks) != pPlanes)
        {
          if (!mAdjustSubpel)
          {
            pBlocks->x = (int)std::lround(pBlocks->x * mScaleX); // 2.7.23: proper rounding for negative vectors!
            pBlocks->y = (int)std::lround(pBlocks->y * mScaleY);
          }
          pBlocks->sad = (sad_t)(pBlocks->sad * mScaleX * mScaleY + 0.5);
          if (changeBitDepth) {
            if (currentBits < mNewBits)
              pBlocks->sad <<= bitDiff;
            else
              pBlocks->sad = (pBlocks->sad + (1 << (bitDiff - 1))) >> bitDiff; // round and shift
          }
          pBlocks++;
        }
      }
    }
  }

  // If scaling vectors only (blocksize remains same) then must check if new vectors go out of frame
  else if (mMode == VectorsOnly)
  {

    // Go through blocks at each level, coarsest first. With divide > 0 the header describes the divided sub-blocks
    // (half block size and overlap, twice the block count, one level more); every level but the finest holds
    // undivided blocks, so their geometry is the undivided one, one level shallower.
    int level = mVectorsInfo.nLvCount - 1;
    while (level >= 0)
    {
      const bool undividedLevel = divided && level > 0;
      const int bsX = undividedLevel ? mVectorsInfo.nBlkSizeX * 2 : mVectorsInfo.nBlkSizeX;
      const int bsY = undividedLevel ? mVectorsInfo.nBlkSizeY * 2 : mVectorsInfo.nBlkSizeY;
      const int ovX = undividedLevel ? mVectorsInfo.nOverlapX * 2 : mVectorsInfo.nOverlapX;
      const int ovY = undividedLevel ? mVectorsInfo.nOverlapY * 2 : mVectorsInfo.nOverlapY;
      const int nBlkX0 = undividedLevel ? mVectorsInfo.nBlkX / 2 : mVectorsInfo.nBlkX;
      const int nBlkY0 = undividedLevel ? mVectorsInfo.nBlkY / 2 : mVectorsInfo.nBlkY;
      const int lev = (divided && level > 0) ? level - 1 : level; // pyramid level of this plane

      // Dimensions of frame covered by blocks (where frame is not exactly divisible by block size there is a small border that will not be motion compensated)
      int widthCovered  = (bsX - ovX) * nBlkX0 + ovX;
      int heightCovered = (bsY - ovY) * nBlkY0 + ovY;

      int blocksSize = plane_len(pPlanes);
      VECTOR* pBlocks = reinterpret_cast<VECTOR*>(pPlanes + 1);
      pPlanes += blocksSize;

      // Width and height of this level in blocks...
      int levelNumBlocksX = ((widthCovered  >> lev) - ovX) / (bsX - ovX);
      int levelNumBlocksY = ((heightCovered >> lev) - ovY) / (bsY - ovY);
      // ... and in pixels
      int levelWidth  = mVectorsInfo.nWidth;
      int levelHeight = mVectorsInfo.nHeight;
      for (int i = 1; i <= lev; i++)
      {
        int xRatioUV = mVectorsInfo.xRatioUV;
        int yRatioUV = mVectorsInfo.yRatioUV;
        levelWidth  = (mVectorsInfo.nHPadding >= xRatioUV) ? ((levelWidth /xRatioUV + 1) / 2) * xRatioUV : ((levelWidth /xRatioUV) / 2) * xRatioUV;
        levelHeight = (mVectorsInfo.nVPadding >= yRatioUV) ? ((levelHeight/yRatioUV + 1) / 2) * yRatioUV : ((levelHeight/yRatioUV) / 2) * yRatioUV;
      }
      int extendedWidth  = levelWidth  + 2 * mVectorsInfo.nHPadding; // Including padding
      int extendedHeight = levelHeight + 2 * mVectorsInfo.nVPadding;

      // Padding is effectively smaller on coarser levels
      int paddingXScaled = mVectorsInfo.nHPadding >> lev;
      int paddingYScaled = mVectorsInfo.nVPadding >> lev;

      // Loop through block positions (top-left of each block, coordinates relative to top-left of padding)
      int x = mVectorsInfo.nHPadding;
      int y = mVectorsInfo.nVPadding;
      int xEnd = x + levelNumBlocksX * (bsX - ovX);
      int yEnd = y + levelNumBlocksY * (bsY - ovY);
      while (y < yEnd)
      {
        // Max/min vector length for this block
        int yMin = -mVectorsInfo.nPel * (y - mVectorsInfo.nVPadding + paddingYScaled);
        int yMax =  mVectorsInfo.nPel * (extendedHeight - y - bsY - mVectorsInfo.nVPadding + paddingYScaled);

        while (x < xEnd)
        {
          int xMin = -mVectorsInfo.nPel * (x - mVectorsInfo.nHPadding + paddingXScaled);
          int xMax =  mVectorsInfo.nPel * (extendedWidth - x - bsX - mVectorsInfo.nHPadding + paddingXScaled);
        
          // Scale each block's vector & SAD
          pBlocks->x = (int)std::lround(pBlocks->x * mScaleX); // 2.7.23: proper rounding for negative vectors!
          pBlocks->y = (int)std::lround(pBlocks->y * mScaleY);
          if (pBlocks->x < xMin || pBlocks->x > xMax || pBlocks->y < yMin || pBlocks->y > yMax)
          {
            // Scaling vector makes motion go out of frame, set 0 vector instead and large SAD
            pBlocks->x = pBlocks->y = 0;
            pBlocks->sad = bsX * bsY * big_pixel_sad;
          }
          else {
            pBlocks->sad = (int)(pBlocks->sad * mScaleX * mScaleY + 0.5); // Vector is OK, scale SAD for larger blocksize
            if (changeBitDepth) {
              if (currentBits < mNewBits)
                pBlocks->sad <<= bitDiff;
              else
                pBlocks->sad = (pBlocks->sad + (1 << (bitDiff - 1))) >> bitDiff; // round and shift
            }
          }
          pBlocks++;

          // Next block position
          x += bsX - ovX;
        }
        x =  mVectorsInfo.nHPadding;
        y += bsY - ovY;
      }
      if (reinterpret_cast<int*>(pBlocks) != pPlanes) Env->ThrowError("MScaleVect: Internal error"); // Debugging check
      level--;
    }
    if (pPlanes != pEnd) Env->ThrowError("MScaleVect: Internal error"); // Debugging check
  }

  return dst;
}

