LoadPlugin("C:\dev\uconvert\dlls\LSMASHSource.dll")
LoadPlugin("C:\dev\uconvert\dlls\RgTools.dll")
LoadPlugin("C:\Users\User\AppData\Local\Temp\claude\C--dev-mvtools-mvtools\91946ad5-43ab-4a78-8fef-4a9c93d7c949\scratchpad\dtl\buildfix\mvtools2.dll")
SEGFILE = "C:\dev\mvtools\profile\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, cachefile=SEGFILE + ".lwi")
pre = m.RemoveGrain(22)
supf = MSuper(pre, pel=2, hpad=16, vpad=16, rfilter=4, mt=false)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, mt=false)
bv = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)
fv = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)
bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, smooth=2, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, smooth=2, search=3, searchparam=2, truemotion=true, chroma=true, mt=false)out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false)
return out.Trim(18, 25)


