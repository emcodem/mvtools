# QTGMC with the whole chain on the GPU. Set the clip and the preset; open in AvsPmod / VirtualDub / ffmpeg.
# 8 to 16 bit input: 8 and 10 bit run every stage on the GPU. A stage without a GPU path (above 10 bit: lookup tables,
# NNEDI3; NNEDI3 also on CPUs where the nnedi3 plugin does not take its AVX2 + FMA path) stops the script with an error
# that names it; with global cuda_cpu_fallback = true (set it before QTGMC) such stages run their CPU filters with the
# same result.
clip_file = "C:\path\to\interlaced.mxf"
preset    = "Fast"   # any QTGMC preset: Ultra Fast .. Placebo

Import(ScriptDir() + "..\QTGMC_gpu_load.avsi")

# ffms2: LWLibav (L-SMASH r1310) returns wrong frames on random access in some MXF files, which Prefetch triggers
src = FFVideoSource(clip_file).AssumeTFF()   # AssumeBFF() for bottom field first material

QTGMC_gpu_on()
src.QTGMC(Preset=preset)
Prefetch(12)   # 12 was the fastest setting on the test machine (Core Ultra 9 285K, RTX 5090)
