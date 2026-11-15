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
- for the GPU stages an NVIDIA GPU with a current driver. On Windows, hardware-accelerated GPU scheduling should be on
  (Settings > System > Display > Graphics > Change default graphics settings; restart after changing it): with it
  off the GPU stages run several times slower. The result file records the setting.

The benchmark checks the first two before anything else and says what is missing. Nothing is
installed or changed. Close other heavy programs (games, renders) while it runs: they change the numbers.

## What it measures

The benchmark measures the worst case of UConvert's 1080 broadcast conversions, **1080i29.97 -> 1080i25**: the
deinterlacer makes 59.94 pictures per second (more than for 25i -> 29.97i), and nearly every output picture of the
frame-rate conversion is a newly made one. The single stages are the parts of that chain. QTGMC runs with UConvert's
preset ("Fast").

The source is generated: MovingPattern (`MovingPattern.dll` in this folder) makes a YUV422P10 stress clip
(textured pan, zoom, moving objects with occlusion, fine stripes, grain, a scene cut) once, before each run, and
keeps it in memory, so reading the source costs nothing. Each stage runs through UConvert itself, with a target that
leaves only that stage to do:

| stage | what |
| --- | --- |
| `deint` | deinterlacing (QTGMC on the GPU) 1080i29.97 -> 1080p59.94 |
| `deint_nvof` | the same with QTGMC's motion vectors from the NVIDIA optical flow accelerator (`uc_qtgmc_nvof`) |
| `deint_wave` | the same with the wavefront motion search (`uc_qtgmc_wavefront`; slightly different vectors than the CPU QTGMC) |
| `deint_nvenc` | the same with QTGMC's motion vectors from NVENC's motion estimation (`uc_qtgmc_nvenc`, `nvencmv.dll`) |
| `deint_sd` | deinterlacing 576i25 -> 576p50: one SD picture's motion search fills only a small part of a GPU, so this stage shows whether the searches of the frames in flight run at the same time on this machine |
| `deint_sd_wave` | the same with the wavefront motion search |
| `frc` | motion-compensated frame-rate conversion (CudaSplatFps) 1080p59.94 -> 1080p50 |
| `frc_nvof` | the same with SplatFps' candidates from the NVIDIA optical flow accelerator (`uc_frc_nvof`) |
| `frc_nvenc` | the same with SplatFps' candidates from NVENC's motion estimation (`uc_frc_nvenc`, `nvencmv.dll`) |
| `interlace` | interlacing (CudaConvolution) 1080p50 -> 1080i25 |
| `scale` | scaling 2160p50 -> 1080p50 (UHD sources) |
| `chain_gpu` | **the whole chain 1080i29.97 -> 1080i25 on the GPU** |
| `chain_gpu_nvof` | the same with NVOFA vectors for the deinterlacer |
| `chain_cpu` | the whole chain on the CPU only (`gpu=false`; `-NoCpu` skips it) |

Each stage reports frames per second and the realtime factor (frames per second / output frame rate; 1.0 =
realtime, `chain_*`: 25 interlaced frames per second).

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
clocks, VRAM, PCIe link maximum and power limit, whether hardware-accelerated GPU scheduling is on, Windows version,
AviSynth+ and ffmpeg versions, power plan, whether a laptop ran on battery. No paths, user names or computer names.
A stage that fails is listed with its error (no NVIDIA GPU, old driver, out of memory, ...).

While each stage runs the benchmark also records how the GPU ran, from the timed part only (an idle GPU parks its
PCIe link at gen 1 and its clocks low, so readings taken before or after a run say nothing about it):

- `gpu_under_load`: PCIe link generation and width, power state, median SM / memory clock and load, peak power
  and temperature (`nvidia-smi`, every 0.5 s). A link narrower or slower than the maximum listed for the GPU, or a
  power state other than P0 / P1 / P2, slows the GPU stages.
- `gpu_memory_mib`: peak GPU memory of the run (`dedicated_peak`) and peak system RAM the GPU reaches over PCIe
  (`non_local_peak`; Windows counter "GPU Process Memory"). Non-local memory is normally the transfer buffers (a
  few hundred MiB). When the GPU's memory is nearly full, Windows also moves GPU memory to system RAM there,
  without an error, and the stage gets much slower: a non-local peak far above other machines' for the same stage
  points to that.
- `hw_gpu_scheduling`: hardware-accelerated GPU scheduling (Settings > System > Display > Graphics), as the driver
  reports it in force; `hw_gpu_scheduling_setting` is the chosen setting, which takes effect after a restart. With
  it off, Windows schedules the GPU work on the CPU, which can keep the work of the frames in flight from running
  at the same time.

Please post the `.json` file in the project's GitHub Discussions (https://github.com/emcodem/mvtools/discussions).
