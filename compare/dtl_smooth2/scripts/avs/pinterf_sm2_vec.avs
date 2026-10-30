LoadPlugin("C:\dev\uconvert\dlls\LSMASHSource.dll")
LoadPlugin("C:\dev\uconvert\dlls\RgTools.dll")
LoadPlugin("C:\dev\mvtools\profile\build_m1\mvtools2.dll")
SEGFILE = "C:\dev\mvtools\profile\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, cachefile=SEGFILE + ".lwi")
pre = m.RemoveGrain(22)
supf = MSuper(pre, pel=2, hpad=16, vpad=16, rfilter=4, mt=false)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, mt=false)
bv = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)
fv = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)
bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, smooth=2, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, smooth=2, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)return Interleave(bv.Trim(10, -1), fv.Trim(10, -1), bv.Trim(50, -1), fv.Trim(50, -1))

