# Smallest chain: MSuper + MAnalyse + MCompensate on a blank-ish clip (load / arguments / vector clips).
import vapoursynth as vs
from vstest import core, load_mvtools

load_mvtools()
src = core.std.BlankClip(format=vs.YUV420P8, width=320, height=240, length=10, color=[64, 128, 128])
sup = core.mvpf.MSuper(src, pel=2)
vec = core.mvpf.MAnalyse(sup, isb=True, blksize=16)
core.mvpf.MCompensate(src, sup, vec).set_output()
