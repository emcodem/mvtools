# CPU engine (no NVIDIA GPU needed, identical output), bottom field first
LoadPlugin("CudaConvolution.dll")
FFVideoSource("source_uhd50p.mov", threads=0)
Prefetch(1)
CudaConvolution(width=1920, height=1080, cpu=true, field_order="bff")   # uses all CPU threads internally
