# splatfps (in development)

Motion-compensated frame-rate conversion, any progressive rate to any other: `SplatFps` (CPU) and `CudaSplatFps`
(GPU). Its own plugin DLL, `splatfps.dll`; the device code goes into `mvtools_cuda.dll`, which the plugin opens at
run time, as cudafilters does. User documentation: `Documentation/splatfps.md`. Vector sources: the built-in phase
correlation, or any candidate clip (`vectors=`, layout in `cand_clip.h`): `SplatAnalyse` (the built-in one as a clip),
`MSplatVectors` of mvtools2 (MAnalyse / CudaMAnalyse / CudaNVOFAnalyse vector clips).

Each output picture is built from its two source neighbours: both are splatted along their motion vectors with
per-pixel weights into a Laplacian pyramid (5 bands at 1080p), normalised, with an in-place blend where the vectors
are inconsistent and a cross-fade where nothing lands. A scene cut repeats the nearer source picture; an optional
temporal fallback (off by default) pulls the output time towards the nearer source picture when the vectors match
badly.

## Stages

input (codes -> 4:4:4 float samples), ME pictures (downscaled luma: A for matching, B for phase correlation), phase
correlation (candidate vectors), refinement (one Lucas-Kanade step per candidate), match errors, assignment (per
pixel forward / backward vectors, occlusion handling), local fallback, schedule (copy or built frame), pair statistic
(scene cut, temporal fallback), build (Laplacian-pyramid forward splatting), output (4:4:4 -> the input format).
The descriptions are in the headers of `core/` (`splat_core.h`, `splat_me.h`, `splat_pcorr.h`, `splat_motion.h`,
`splat_phase_pull.h`); every tuned value and its default is in `core/splat_config.h`.

## State

| Stage | Exact CPU path (`core/`) | CUDA exact | CUDA fast |
|---|---|---|---|
| input YUV 4:2:0 / 4:2:2 / 4:4:4, 8-16 bit -> 4:4:4 float | done | done, bit-identical | shared with exact |
| ME pictures | done | done, bit-identical | done (FMA: up to 3e-5 off) |
| phase-correlation vector source | done | done, bit-identical | uses the exact variant |
| other vector sources (candidate clips: MVTools, NVOFA) | done (`vectors=`) | done (candidates uploaded) | same |
| refinement, match errors, assignment, local fallback | done | done, bit-identical | done (FMA): same decisions on >= 99.999 % of ME pixels |
| schedule, pair statistic | done | sums on the GPU, formula shared (`core/splat_phase_pull.h`); schedule on the host | same |
| build | done | done, bit-identical (sorted splats) | done (FMA, atomics): a few dozen codes +-1 per frame |
| output float -> the input format | done | done, bit-identical | shared with exact |
| AviSynth+ filter front-end | `SplatFps` (fast path by default, `exact=true`) | `CudaSplatFps(exact=true)` | `CudaSplatFps` (default) |

The exact path is the deterministic scalar path: float32 in a fixed evaluation order, no contraction, IEEE division
and sqrt, exp through double; the fast CPU path and CUDA (exact=true) are checked bit for bit against it. Formats:
planar YUV 4:2:0 / 4:2:2 / 4:4:4 at 8-16 bit, output = input format, copy frames passed on unchanged.

## Phase correlation

Built from the published technique (G. A. Thomas, BBC RD 1987/11; US 4,890,160) and measurements on a ground-truth
test corpus (local errors, artefacts and flicker against the real frames): Tukey-windowed blocks of 32 / 64 / 128 with
the windowed mean removed, an optional Gaussian high-pass prefilter, the cross-power spectrum weighted by q / (q + k^2)
against the blocks' median amplitude, the highest positive peaks per block, the near and far covering blocks per
pixel, a global candidate by weighted vote. Design and arithmetic in `core/splat_pcorr.h`; settings `block_peaks`,
`nearest_block`, `highpass_sigma`, `block_taper`, `noise_weight`, `peak_ratio`, `global_candidate`,
`secondary_peak_cost`.
- Defaults `block_peaks=[2, 2, 3]` (29 candidates at 1080p), `highpass_sigma=8`. Cheaper layouts measured equal
  within noise: `nearest_block=[0, 0, 1]` (20 candidates), `block_peaks=[1, 2, 3], nearest_block=[0, 0, 1]` (16).
- Arithmetic: float only, two real blocks per complex FFT both ways, no trigonometric functions, fixed-point vote
  sums; exact CPU path == fast CPU path (AVX2 FFT lines) == CUDA, bit for bit.
- Speed (Core Ultra 9 285K 24 threads / RTX 5090, 1080p 50 -> 59.94): ingest 3.7 ms, pair 4 ms per picture on the
  CPU; GPU kernels 196 ms per 600 frames.

## CPU fast path (`SplatFps(exact=false)`, `core/splat_fast*`, `core/splat_par*`)

The exact stages (input, ME pictures, phase correlation, refinement to local fallback, build, output) on all cores
(`splat_par.h`: a pool owned by the filter instance, dynamic scheduling, short spin before sleeping) and, with AVX2 +
FMA (run-time check), 8-wide kernels (`splat_fast_avx2.cpp`, the only file compiled with `/arch:AVX2` /
`-mavx2 -mfma`). The kernels keep the exact operation order (lanes = neighbouring pixels; long sums run several
independent chains side by side), the FFTs are pocketfft's passes on 4 / 8 lines at once, the build's splats run in
bands of output rows over samples filed by target row (same per-cell order), and the refinement and the match errors
stream each candidate through small ring buffers. Only exp, atan2, sin and cos differ: double-precision polynomials
rounded to float (equal to the C library's rounded result except within ~1e-15 of a rounding boundary). Without AVX2
the build and the conversions run the portable code on all threads and the motion stages the exact functions. The
pair candidates are made once per pair (the exact path makes them twice). Default `exact=false` (both filters);
`exact=true` is the path the tests check against. run_suite runs the exact CPU path as a whole filter only in
`check_fast` and `check_avs cuda`; the other cases compare the fast CPU path with `CudaSplatFps(exact=true)`.

Stage times in ms at 1080p (`sc_profile`, best of 5, Core Ultra 9 285K, 24 threads), exact -> fast: ME picture A
12.9 -> 0.8, B 11.3 -> 0.8; phase-correlation ingest 48.6 -> 5.5, pair 46.7 -> 6.3 (made once per pair instead of
twice); fields 126 -> 8.2 (assignment 49 -> 1.5, local fallback 12 -> 1.7); build 442 -> 26. Through AviSynth+ (1080p
50 -> 59.94, 300 source pictures, decode included, High priority): 1.5 fps -> 20.4 fps.

## CUDA

`CudaSplatFps(clip, num, den, exact=false)`: a device context in mvtools_cuda (`mvcuda_splat_ctx_*`: source pictures
uploaded and analysed once, fields kept per picture). exact=true gives the frames of the exact CPU path; exact=false
(FMA, single-precision exp, atomic splats) differs by a few dozen codes +-1 per frame. Kernels: build in
`Sources/cuda/mvcuda_splat.cuh`, ME pictures / match errors / local fallback / pair-statistic sums in
`mvcuda_splat_me.cuh`, phase correlation / refinement / assignment in `mvcuda_splat_mo.cuh`, compiled twice as
`mvcuda_splat_exact.cu` (`-fmad=false`) and `mvcuda_splat_fast.cu`. Returns normal frames (no device-token output
yet). Speed at 1080p 50 -> 59.94 (RTX 5090, High priority): about 200 fps, host-bound; GPU time per source picture
(kernels only) about 2.4 ms. MT_SERIALIZED.

## Build

- Part of the repo build (`MVTOOLS_SPLATFPS`, default on): `splat_core` (static) and `splatcore_test.dll` (C ABI
  for the tests, not shipped, `SPLATFPS_TESTS`). The exact path needs float arithmetic in source order: MSVC
  `/fp:precise` (no contraction), GCC / Clang `-ffp-contract=off`.
- `third_party/pocketfft/fftp_pow2.h`: pocketfft's complex FFT (BSD) for power-of-two sizes, restructured into
  per-pass work items for the GPU, so the CPU and the GPU transforms give the same bits (peak positions on surfaces
  with exact ties depend on them).

## Tests

`python -I test/run_suite.py <dll dir> <source clip> [--quick] [--lsmash <LSMASHSource.dll>] [--rate num/den]` runs
every check and prints pass / fail per case. `<dll dir>` holds the plugin set (`splatfps.dll`, `mvtools2.dll`,
`mvtools_cuda.dll`, `cudafilters.dll`) and `splatcore_test.dll` (or the file the environment variable
`SPLATCORE_TEST` names). The source clip is a 10-bit 4:2:2 clip (1080p for the default settings), read through
L-SMASH Works: the environment variable `SPLATFPS_LSMASH` (or `--lsmash`) points at `LSMASHSource.dll`. Needs numpy,
ffmpeg and ffprobe on PATH; every child process runs hidden. The stage inputs are derived from the first pictures of
the clip by the exact CPU path (`test/stage_inputs.py`), so no stored data is needed.
- `test/check_fast.py`: every stage of the fast CPU path (with and without the AVX2 kernels) against the exact path,
  and the whole filter `exact=false` against `exact=true`; bounds: final vectors equal on >= 99.9 % of ME pixels,
  built frames <= 500 codes different, none by more than 2 (expected: 0).
- `test/check_motion.py`: phase correlation, refinement, assignment, the fields and the chain from luma on CUDA exact
  (bit-identical with the exact CPU path) and CUDA fast (final vectors within 0.01 px on >= 99 % of ME pixels).
- `test/check_cuda.py`: ME pictures, match errors, local and temporal fallback and the build on CUDA exact
  (bit-identical) and CUDA fast (reported).
- `test/check_avs.py`: the whole filter through AviSynth+, random access: every copy frame equals its source picture,
  built frames equal the other backend.
- `test/check_vectors.py`: the candidate-clip sources (SplatAnalyse round trip, MSplatVectors from MAnalyse and NVOFA,
  CudaSplatFps with `vectors=`).
- In run_suite itself: sequential runs, other frame sizes, non-default named arguments and every pixel format, each
  `SplatFps` (fast) == `CudaSplatFps(exact=true)`.

## Quality

Results on the ground-truth test corpus: to be filled in.
