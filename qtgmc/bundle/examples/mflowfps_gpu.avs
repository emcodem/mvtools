# Frame-rate conversion with MVTools on the GPU (CudaMSuper / CudaMAnalyse / CudaMRecalculate / CudaMFlowFps), the chain
# of uConvert's uc_FRC: a coarse search, then a refine pass with smaller blocks. The Cuda* filters take the same
# arguments as MSuper / MAnalyse / ... and give the same result as those CPU filters.
# Their output stays on the GPU (token frames): CudaDownload at the end turns it into a normal clip.
# A setting without a GPU version stops the script with an error that names it; with global cuda_cpu_fallback = true
# (before the first Cuda* call) such a filter runs its CPU code instead, so the script also runs without a GPU.
# pel 1, 2 or 4, exhaustive search (search=3) radii up to 64 and trymany=true run on the GPU.
# Frames at scene changes (vectors flagged unusable by thSCD1 / thSCD2) are blended on the CPU, as by MFlowFps.
# Progressive input only (deinterlace first, e.g. with QTGMC). See mflowfps_nvof.avs for the optical flow variant.
clip_file = "C:\path\to\progressive.mp4"
fps_num   = 60000
fps_den   = 1001
pel       = 2

Import(ScriptDir() + "..\QTGMC_gpu_load.avsi")

c = FFVideoSource(clip_file)
supf = c.CudaMSuper(pel=pel, hpad=16, vpad=16, rfilter=4)            # coarse search: full pyramid
sup  = c.CudaMSuper(pel=pel, hpad=16, vpad=16, rfilter=4, levels=1)  # refine / interpolation: one level
bv = supf.CudaMAnalyse(isb=true,  blksize=16, overlap=8, search=3, searchparam=2, truemotion=true)
fv = supf.CudaMAnalyse(isb=false, blksize=16, overlap=8, search=3, searchparam=2, truemotion=true)
bv = sup.CudaMRecalculate(bv, blksize=8, overlap=4, thSAD=100, search=3, searchparam=2, truemotion=true)
fv = sup.CudaMRecalculate(fv, blksize=8, overlap=4, thSAD=100, search=3, searchparam=2, truemotion=true)
c.CudaMFlowFps(sup, bv, fv, num=fps_num, den=fps_den, mask=2, ml=200, blend=false)
CudaDownload()
Prefetch(6)
