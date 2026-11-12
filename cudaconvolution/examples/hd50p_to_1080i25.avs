# 1080p50 (1920x1080) -> 1080i25 TFF, vertical interlace filter (any size works the same way)
LoadPlugin("CudaConvolution.dll")
FFVideoSource("source_1080p50.mov", threads=0)
Prefetch(1)
BitsPerComponent() < 10 ? ConvertBits(10) : last    # e.g. 8-bit sources
Is422() ? last : ConvertToYUV422()                  # e.g. 4:2:0 sources
CudaConvolution()                                          # same size: interlace filter only (factor 1)
