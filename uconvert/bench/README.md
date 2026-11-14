# UConvert benchmark

Measures how fast UConvert's stages run on this machine, in about a minute, and writes one result file to post.

## Run

Double-click `bench_uconvert.cmd` (in `UConvert\bench\`), or from a command prompt:

```
bench_uconvert.cmd [-FFmpeg C:\path\ffmpeg.exe] [-Seconds 5] [-NoCpu] [-Stages deint,frc] [-Threads n]
```

Needs:
- **ffmpeg** with AviSynth input: `ffmpeg.exe` next to the script, in the UConvert folder or on the PATH, otherwise
  give its path with `-FFmpeg C:\path\ffmpeg.exe`.
- **AviSynth+ 3.7 or newer, x64**: `AviSynth.dll` either in `C:\Windows\System32` (the AviSynth+ installer puts it
  there) or next to `ffmpeg.exe` (the x64 `AviSynth.dll` from the AviSynth+ release archive).
- for the GPU stages an NVIDIA GPU with a current driver.

The benchmark checks the first two before anything else and says what is missing. Nothing is
installed or changed. Close other heavy programs (games, renders) while it runs: they change the numbers.

## What it measures

The source is generated: MovingPattern (`MovingPattern.dll` in this folder) makes a 1080i25 YUV422P10 stress clip
(textured pan, zoom, moving objects with occlusion, fine stripes, grain, a scene cut) once, before each run, and
keeps it in memory, so reading the source costs nothing. Each stage runs through UConvert itself, with a target that
leaves only that stage to do:

| stage | what |
| --- | --- |
| `deint` | deinterlacing (QTGMC on the GPU) 1080i25 -> 1080p50 |
| `deint_wave` | the same with the wavefront motion search (`uc_qtgmc_wavefront`; slightly different vectors than the CPU QTGMC) |
| `deint_nvof` | the same with QTGMC's motion vectors from the NVIDIA optical flow accelerator (`uc_qtgmc_nvof`) |
| `deint_nvenc` | the same with QTGMC's motion vectors from NVENC's motion estimation (`uc_qtgmc_nvenc`, `nvencmv.dll`) |
| `frc` | motion-compensated frame-rate conversion (CudaSplatFps) 1080p50 -> 1080p59.94 |
| `frc_nvof` | the same with SplatFps' candidates from the NVIDIA optical flow accelerator (`uc_frc_nvof`) |
| `frc_nvenc` | the same with SplatFps' candidates from NVENC's motion estimation (`uc_frc_nvenc`, `nvencmv.dll`) |
| `interlace` | interlacing (CudaConvolution) 1080p59.94 -> 1080i29.97 |
| `scale` | scaling 2160p50 -> 1080p50 |
| `chain_gpu` | the whole chain 1080i25 -> 1080i29.97 on the GPU |
| `chain_cpu` | the whole chain on the CPU only (`gpu=false`; `-NoCpu` skips it) |

The `_nvof` and `_nvenc` stages compare the optical flow accelerator (Turing or newer) and the video encoder's motion
estimation (Pascal or newer) with the built-in motion searches. Both run on their own chips next to the CUDA cores,
so the whole stage is timed, not the search alone. Where the hardware (or `nvencmv.dll`) is missing they are listed
as skipped.

Every stage runs with UConvert's default number of frames in flight (GPU chain: `uc_gpu_threads`, 10; CPU chain: 3);
`-Threads n` sets another one for all stages, to find the best value for a GPU.

Every stage runs for `-Seconds` (5) after its start-up. Its speed is the median of the frames per second over 1 s
windows (the first second is skipped); the result also lists the slowest and fastest window and the start-up time.

## Results

`results\uconvert_bench_<date>.json` (to post) and `.txt` (to read). They contain the speeds, the chain UConvert
chose for each stage (`plan`), and the machine: CPU with cores (performance / efficiency cores on hybrid CPUs),
clock, caches and instruction sets (SSE4, AVX2, AVX-512), RAM, GPUs with driver, CUDA version, compute capability,
clocks, VRAM, PCIe link and power limit, Windows version, AviSynth+ and ffmpeg versions, power plan, whether a laptop ran on battery. No paths, user names or
computer names. A stage that fails is listed with its error (no NVIDIA GPU, old driver, out of memory, ...).

Please post the `.json` file in the project's GitHub Discussions (https://github.com/emcodem/mvtools/discussions).
