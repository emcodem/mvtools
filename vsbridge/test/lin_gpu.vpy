# Same chain as the AviSynth+ Linux reference ~/build/test/lin_gpu.avs (MSuper, MAnalyse, MRecalculate, MFlowFps;
# UHD 10-bit 4:2:2, CudaMAnalyse / CudaMRecalculate on the GPU): the framemd5 must be identical.
import os
import vapoursynth as vs
from vstest import core, load_mvtools, raw_source

load_mvtools()
m = raw_source(os.environ.get('MVPF_RAW', os.path.expanduser('~/clips/seg1_40.yuv')), 3840, 2160, vs.YUV422P10,
               30000, 1001)
m = m * int(os.environ.get('MVPF_LOOP', '1'))  # longer clips for prefetch / warm-up checks (md5 differs then)
mv = core.mvpf
supf = mv.MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4)
sup = mv.MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1)
bv = mv.CudaMAnalyse(supf, isb=True, blksize=32, overlap=16, search=3, searchparam=2, truemotion=True, chroma=True)
fv = mv.CudaMAnalyse(supf, isb=False, blksize=32, overlap=16, search=3, searchparam=2, truemotion=True, chroma=True)
bv = mv.CudaMRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=True, chroma=True)
fv = mv.CudaMRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=True, chroma=True)
mv.MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=False, mt=True).set_output()
