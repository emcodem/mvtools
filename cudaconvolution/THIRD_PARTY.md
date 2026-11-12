# Third-party components in CudaConvolution.dll

| component | use | license |
|---|---|---|
| NVIDIA CUDA Runtime 12.9 (`cudart_static`) | linked statically | NVIDIA CUDA Toolkit EULA; the CUDA runtime is listed there as redistributable |
| nlohmann/json 3.11.3 | header-only, preset parsing | MIT, Copyright (c) 2013-2022 Niels Lohmann |
| AviSynth+ 3.7.3 headers (`avisynth.h`, `avs/*.h`) | plugin interface | GPL v2 plus the special exception in `avisynth.h` for "independent modules that communicate with Avisynth solely through the interfaces defined in avisynth.h" (full text in the header) |
| Microsoft C/C++ runtime (static, `/MT`) | linked statically | Visual Studio license |

Test-only (not in the DLL): Catch2 3.7.1 (BSL-1.0).
