# SplatFps performance TODO

## State 2026-11-11, branch `splatfps-perf` (GPU work per output frame)

Goal: less GPU work per output frame of CudaSplatFps (default fast path), measured against a commercial
motion-compensated converter on the same GPU. Test case: Statue 1080p50 4:2:2 10 bit -> 59.94p (1318 output frames),
RTX 5090. Quality gate: the 68-case ground-truth corpus (local block error, worst block, artefact blocks, flicker;
+-2 % = equal) and the SplatFps suite; exact=true must stay bit-identical.

### Done (commits on `splatfps-perf`)
| commit | change | kernel time (nsys sum) |
|---|---|---|
| 08ab0b3 | sliding box sums (fast variant); weighted Lucas-Kanade terms + refine_box sums + 2x2 solve fused per tile (`k_refine_fused`, the 5 term planes per candidate stay in shared memory) | 2.21 -> 1.84 s |
| 7be4df1 | band pyramid built once per source in the device pipeline (was per output frame and side); exact=true output identical | -> 1.69 s |
| 3f6ca78 | float4 atomicAdd splats on an interleaved accumulator (cc >= 9.0), in-place part added by `k_splat` (no `k_inplace`), 16 x 16 refinement tiles | -> 1.58 s |
| 0abaf4f | below cc 9.0 the planar accumulator and `k_inplace` stay (scalar atomics on interleaved cells were 2x slower) | sm_75 code: 2.45 -> 1.85 s |

Also: GPU energy above idle 0.80 -> ~0.61 J per output frame, DRAM traffic 1.81 -> ~1.0 GB per output frame,
instructions issued -37 %. Corpus 68/68 equal, suite 27/27 for sm_120 and 75-virtual builds. Wall time of the
Statue run (~9 s) unchanged: decode + v210 write bound.

### Lessons
- At 1080p the per-candidate ME planes (~16 MB for ~30 candidates) stay in the 96 MB L2: fusing those stages saves no
  DRAM and only adds apron recomputation (a fused |dfd| + box + edge-filter kernel was tried: no gain, reverted).
  Fusing paid off only for the 5 x n refinement term planes (~80 MB).
- Box filters were memory bound, not arithmetic bound: sliding sums cut instructions / energy, not time.
- Shared memory per SM: ~100 KB on cc 12.0, 64 KB on Turing. A 47 KB block = 2 blocks per SM (latency bound); size
  tiles so at least 2 (better 3) blocks fit. Set the carve-out preference for shared-memory-heavy kernels.
- Old architectures: `-CudaArch 75-virtual` (or 61-virtual) builds run as JIT PTX on the 5090 with those code paths
  (__CUDA_ARCH__ 750): good for correctness and for gross regressions, not for real old-card speed (SM count, L2,
  shared memory, atomics stay the 5090's). Compare against the base built the same way, not against native sm_120.
- Bilinear sampling (`tex_bilinear`) is only used by `k_dfd` / `k_absdfd` (~2 %): texture objects not worth it.
- Uploads / downloads already use pinned staging rings; their overlap with kernels would lower GPU busy time only.

### Remaining steps (by expected gain, 5090 kernel time 1.58 s)
1. `k_refine_fused` (~21 %): apron recomputation of the terms (16 x 16 tile + 2 x 7 apron = 3.5x the samples);
   options: keep the error plane and d in shared memory from the error box, wider tiles where shared memory allows,
   fewer passes over shared memory.
2. Phase correlation (~10 %: k_fft, peaks, cross-power): fewer FFT passes (real-to-complex), fused cross / split.
3. `k_assign` (~5 %) and the remaining error box sums (`k_boxs_h` / `k_boxs_v` ~10 %): box of |d| inside the refine
   kernel, error planes in half precision (quality gate).
4. Picture build: `k_splat` (~12 %), `k_norm`, `k_band`: tile-local shared-memory accumulation for short vectors
   (global atomics only for far targets); normalisation fused into the next level's reconstruction.
5. Candidate count: `block_peaks=[1,2,1]` instead of 2/2/3 measured -25 % kernel time; a defaults change, needs the
   quality gate on the corpus and the user's decision.
6. Determinism of the fast path (see TODO.md): fixed-point (integer) atomics in `k_splat` would make it reproducible;
   cost to be measured.
7. Real old-card numbers (RTX 20xx / 30xx): needs the hardware; JIT runs on the 5090 only show correctness.

## State 2026-11-09, branch `splatfps-speed`

From the HD tuning state of 2026-11-09. Measurement tools and raw
numbers are kept locally (bench, md5 comparison, CPU stage timing scripts).
Test case: Statue 1080p 4:2:2 10 bit, 50 -> 59.94, Core Ultra 9 285K (8P + 16E), RTX 5090.

## Done (this branch)
All bit-identical except item 7 (GPU fast mode only): CPU frame md5 == tag build, CudaSplatFps(exact=true) md5 ==
tag build, every parity check of the suite passes.

1. GPU peak search: chunked parallel top-K (k_peaks_part / k_peaks_merge) instead of one thread block per surface
   (the largest surface took ~800 us); every chosen peak excluded, as on the CPU. k_peaks was 28 % of GPU time.
2. GPU: two frames in flight (the next built frame is started before the current one is handed on; two download
   buffers; mvtools_cuda API 27), host copies over 4 threads.
3. CPU: Laplacian pyramid per source picture (FastEngine::pyramid_into + build_pyr_into), not per built frame.
4. CPU: parallel field_reduce2; pyramid levels under 64k samples run on the calling thread (Pool::serial).
5. CPU: thread pool finishes a job when its indices are done (generation + index ticket) instead of waiting for every
   worker; 24 threads were 10 % slower than 16 before, now equal.
6. CPU: level-0 reconstruction written straight into the cropped result.
7. GPU fast mode: phase-correlation forward FFT, amplitude / phase and cross-power cos / sin in single precision
   (accepted: fast vs exact deviation unchanged, 2.58 % -> 2.46 % samples differing).

Results: CPU 1080p decode-free 21.6 -> 24.5 fps; GPU 1080p decode-free ~198 -> ~209 fps (host bound), exact 89 -> 97;
GPU UHD fast 49.7 -> 60.4 (before item 7), 43.5 -> 55.1 with item 7 (noisy session, interleaved).

## Where the time is now
- GPU 1080p: host bound. With real LSMASH decode ~137 fps (decode alone ~200 fps); decode, filter host work
  (~2.3 ms / frame: source fetch 0.46, upload 0.76, frame_start 0.59, copy-out 0.45) and ffmpeg run in series on the
  main thread. GPU busy ~45-50 %; of the GPU busy time about a third is the 8 MB upload + download per frame.
- CPU 1080p (~40 ms per built frame): build ~18 ms (memory bound), fields (refinement to local fallback) ~6,
  pyramids ~3.4 (per frame share), phase-correlation pair ~3.4, phase-correlation ingest ~2.7, decode ~3-7, ME
  pictures ~1.6, input + output conversion ~1.3.

## To do, by expected gain at 1080p (estimates, not measured)

### GPU
1. **Host work of CudaSplatFps on a worker thread**: source staging, kernel launches, copy-out in parallel with
   decode and ffmpeg instead of in series. Biggest 1080p gain left (~137 -> 170-190 fps with real decode). Care:
   AviSynth frames handed across threads (fetch child frames on the main thread, pass PVideoFrame references), one
   thread owning the device context.
2. **Fewer launches**: ~154 kernel launches per frame, ~0.9 ms host time. CUDA graphs per source picture / per frame
   (the phase as device memory or a node parameter update), or fused small kernels.
3. Copy stream: uploads / downloads on their own stream with events, overlapping the kernels (mostly UHD). Checked
   2026-11-11: staging is already pinned; overlap would lower GPU busy time only, not GPU work or energy.
4. ~~Band pyramids cached per source on the device~~: done 2026-11-11 (7be4df1), -8 % kernel time.
5. k_tf_rows: h / 64 blocks of 64 threads, ~87 us per frame (5.6 %); exact mode needs the sequential double sums
   (a parallel version for fast mode only).

### CPU
1. **Overlap source analysis with the build**: the analysis (ME pictures to local fallback) of the next source
   while the current frame is built; both have phases that use the threads poorly. Estimate +15-25 %. Bigger
   restructuring (two work streams on one pool, or split thread sets).
2. **Fields stage (~6 ms)**: ~36 whole-plane tasks (candidates x sides) for 24 threads, E-cores straggle. Split each
   candidate into row bands (halos for the box filters of refinement / error); about half.
3. **Float FFT in the phase correlation on the CPU fast path (~6 ms there)**: double runs at half the SIMD rate;
   maybe 2-3 ms. Ends the identity CPU exact == CPU fast (the suite checks it; would need new bounds). NEEDS THE USER'S OK.
4. Build level 0 (~4 ms side planes, accumulator start, normalisation): fuse the passes, normalise only inside the
   border at level 0; maybe 1-2 ms.

### Outside the plugin (both paths)
- Decoding is the largest single cost of a real chain (~7 ms per source frame through LSMASH on the main thread).
  Try LWLibavVideoSource(threads=...) / AviSynth+ Prefetch and document the best setting.

### Postponed
- Portable SIMD (Google Highway) for ARM / AVX-512: only if ARM or macOS becomes a target.
