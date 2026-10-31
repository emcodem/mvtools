# Frame-rate conversion with MVTools on the GPU (MSuper / MAnalyse / MRecalculate / MFlowFps with gpu=true), the chain of
# uConvert's uc_FRC: a coarse search, then a refine pass with smaller blocks. gpu=true is opt-in per filter; where a
# setting has no GPU version the filter runs its CPU code with the same result.
# pel 1, 2 or 4 run on the GPU (pel 4: MAnalyse / MRecalculate / MFlowFps; MSuper then makes a normal super clip).
# Exhaustive search (search=3) radii up to 64 and trymany=true run on the GPU too.
# Frames at scene changes (vectors flagged unusable by thSCD1 / thSCD2) are blended by MFlowFps's CPU path, as before.
# Progressive input only (deinterlace first, e.g. with QTGMC). See mflowfps_nvof.avs for the optical flow variant.
clip_file = "C:\path\to\progressive.mp4"
fps_num   = 60000
fps_den   = 1001
pel       = 2

Import(ScriptDir() + "..\QTGMC_gpu_load.avsi")

c = FFVideoSource(clip_file)
supf = c.MSuper(pel=pel, hpad=16, vpad=16, rfilter=4, gpu=true)            # coarse search: full pyramid
sup  = c.MSuper(pel=pel, hpad=16, vpad=16, rfilter=4, levels=1, gpu=true)  # refine / interpolation: one level
bv = supf.MAnalyse(isb=true,  blksize=16, overlap=8, search=3, searchparam=2, truemotion=true, gpu=true)
fv = supf.MAnalyse(isb=false, blksize=16, overlap=8, search=3, searchparam=2, truemotion=true, gpu=true)
bv = sup.MRecalculate(bv, blksize=8, overlap=4, thSAD=100, search=3, searchparam=2, truemotion=true, gpu=true)
fv = sup.MRecalculate(fv, blksize=8, overlap=4, thSAD=100, search=3, searchparam=2, truemotion=true, gpu=true)
c.MFlowFps(sup, bv, fv, num=fps_num, den=fps_den, mask=2, ml=200, blend=false, gpu=true)
Prefetch(6)
