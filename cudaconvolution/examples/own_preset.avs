# Own preset file instead of the built-in one: copy presets\balanced.json, edit, point to it
LoadPlugin("CudaConvolution.dll")
FFVideoSource("source_uhd50p.mov", threads=0)
Prefetch(1)
CudaConvolution(width=1920, height=1080, preset="my_preset.json")
