# UHD 50p (3840x2160, YUV422P10) -> 1080i25 TFF, luma bit-exact to the measured reference; GPU engine
LoadPlugin("CudaConvolution.dll")                  # not needed if the DLL is in the AviSynth+ plugins64+ folder
FFVideoSource("source_uhd50p.mov", threads=0)   # any source filter; e.g. YUV422P10 3840x2160 progressive
Prefetch(1)                                      # decode ahead (directly after the source, not at the end)
CudaConvolution(width=1920, height=1080, clip_min=1, clip_max=1020)   # factor 2, bit-exact to the reference
