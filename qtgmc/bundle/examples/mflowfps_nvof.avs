# Frame-rate conversion with the coarse motion search done by NVIDIA's optical flow accelerator (CudaNVOFAnalyse of
# cudafilters.dll) instead of MSuper + MAnalyse; MRecalculate refines those vectors and MFlowFps interpolates, both on
# the GPU. This is uConvert's uc_frc_nvof option.
# NOT bit-exact with anything: NVOFA is another algorithm. It follows true motion better on fast pans and is faster
# than the exhaustive search, but finds vectors with a lower SAD less often; judge the result by eye.
# Needs an NVIDIA GPU with NVOFA (Turing / RTX 20xx or newer; this bundle's kernels need RTX 30xx or newer). Without it
# CudaNVOFAvailable() is false and the script takes the exact MAnalyse branch below.
# Progressive input only (deinterlace first, e.g. with QTGMC).
clip_file = "C:\path\to\progressive.mp4"
fps_num   = 60000
fps_den   = 1001
pel       = 2      # CudaNVOFAnalyse: 1 or 2
preset    = 2      # NVOFA preset: 1 fast, 2 medium, 3 slow
blk       = 32     # coarse block size (uc_FRC uses 32 at UHD, 16 at HD)

Import(ScriptDir() + "..\QTGMC_gpu_load.avsi")

c = FFVideoSource(clip_file)
nv = CudaNVOFAvailable()
# hpad / vpad of CudaNVOFAnalyse must be those of the MSuper used downstream
sup = c.MSuper(pel=pel, hpad=16, vpad=16, rfilter=4, levels=1, gpu=true)
supf = nv ? NOP() : c.MSuper(pel=pel, hpad=16, vpad=16, rfilter=4, gpu=true)
bv = nv ? c.CudaNVOFAnalyse(isb=true,  blksize=blk, overlap=blk / 2, pel=pel, hpad=16, vpad=16, chroma=true, preset=preset)
\       : supf.MAnalyse(isb=true,  blksize=blk, overlap=blk / 2, search=3, searchparam=2, truemotion=true, gpu=true)
fv = nv ? c.CudaNVOFAnalyse(isb=false, blksize=blk, overlap=blk / 2, pel=pel, hpad=16, vpad=16, chroma=true, preset=preset)
\       : supf.MAnalyse(isb=false, blksize=blk, overlap=blk / 2, search=3, searchparam=2, truemotion=true, gpu=true)
bv = sup.MRecalculate(bv, blksize=blk / 2, overlap=blk / 4, thSAD=100, search=3, searchparam=2, truemotion=true, gpu=true)
fv = sup.MRecalculate(fv, blksize=blk / 2, overlap=blk / 4, thSAD=100, search=3, searchparam=2, truemotion=true, gpu=true)
c.MFlowFps(sup, bv, fv, num=fps_num, den=fps_den, mask=2, ml=200, blend=false, gpu=true)
Subtitle(nv ? "NVOFA" : "MAnalyse (no NVOFA)")
Prefetch(6)
