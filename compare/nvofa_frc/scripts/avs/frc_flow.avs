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
bv = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fv = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false, mt=true, gpu=true)
return out.Trim(8, 247).Prefetch(3)
