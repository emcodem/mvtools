# uc_FRC chain, GPU build profile\build_nvofa. The search used by MAnalyse is chosen by the environment of the process:
#   MVCUDA_NVOFA unset         -> exact GPU search (k_search, md5 = CPU)
#   MVCUDA_NVOFA=fast|medium|slow -> NVOFA optical flow (opticalflow.md), MRecalculate refines as usual
LoadPlugin("C:\dev\uconvert\dlls\LSMASHSource.dll")
LoadPlugin("C:\dev\uconvert\dlls\RgTools.dll")
LoadPlugin("C:\dev\mvtools\profile\build_nvofa\mvtools2.dll")
SEGFILE = "C:\dev\mvtools\profile\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, threads=4, cachefile=SEGFILE + ".lwi")
pre = m.RemoveGrain(22)
supf = MSuper(pre, pel=2, hpad=16, vpad=16, rfilter=4, gpu=true)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, gpu=true)
ba = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fa = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
bv = MRecalculate(sup, ba, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fv = MRecalculate(sup, fa, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
# MAnalyse vectors (before MRecalculate) that MFlowFps's inputs come from: output frame 41 = source 61.25 (bv 61, fv 62),
# output frame 42 = source 62.5 (bv 62, fv 63)
return Interleave(ba.Trim(61, -1), fa.Trim(62, -1), ba.Trim(62, -1), fa.Trim(63, -1))
