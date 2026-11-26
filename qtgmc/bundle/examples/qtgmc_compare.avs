# QTGMC on the GPU against the stock CPU QTGMC, in one script.
#   mode 0: GPU only   1: CPU only   2: GPU and CPU frames interleaved (100 fps for 25i input)
#   3: GPU above CPU   4: difference x8 (grey = equal)
# The GPU motion search uses meander=false (the wavefront search), the stock CPU QTGMC meander=true, so the two are
#   close but not identical. With the environment variable MVTOOLS_GPU_FORCE_FALLBACK=1 the GPU branch computes the
#   same algorithm on the CPU, byte-identical to the GPU result.
mode      = 2
clip_file = "C:\path\to\interlaced.mxf"
preset    = "Slow"

Import(ScriptDir() + "..\QTGMC_gpu_load.avsi")

src = FFVideoSource(clip_file).AssumeTFF()

QTGMC_gpu_off()
cpu = (mode == 0) ? NOP() : src.QTGMC(Preset=preset)
QTGMC_gpu_on()
gpu = (mode == 1) ? NOP() : src.QTGMC(Preset=preset)

out = (mode == 0) ? gpu.Subtitle("GPU") : \
      (mode == 1) ? cpu.Subtitle("CPU") : \
      (mode == 2) ? Interleave(gpu.Subtitle("GPU"), cpu.Subtitle("CPU")) : \
      (mode == 3) ? StackVertical(gpu.Subtitle("GPU"), cpu.Subtitle("CPU")) : \
                    mt_lutxy(gpu, cpu, "x y - 8 * range_half +", use_expr=1, U=3, V=3).Subtitle("GPU - CPU, x8 (grey = equal)")
out.Prefetch(12)
