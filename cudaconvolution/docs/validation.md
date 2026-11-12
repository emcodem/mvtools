# Validation on real material (2026-10-15)

Material: a 60 s excerpt (1500 frames) of UHD 50p ProRes 422 HQ, a 1080i25 ProRes HQ reference output of the same
frames, and the earlier stock-filter solution (fmtconv) for comparison. Dev PC: Xeon Gold 6128 (24 threads), RTX 5050
on PCIe Gen3 x8, ffmpeg + ffms2.

## Speed (60 s UHD 50p -> ProRes HQ 1080i25 incl. audio, wall time)

| pipeline | time | real time factor |
|---|---|---|
| fmtconv AviSynth graph (previous solution) | 369 s | 0.19x |
| CudaConvolution GPU, `Prefetch(4)` at the end of the script | 241 s | 0.25x |
| CudaConvolution GPU, `Prefetch(1)` directly after the source | **68 s** | **0.88x** |

The placement of `Prefetch` matters more than the filter: with a `Prefetch(N)` at the end, several threads request
frames from ffms2 out of order and it keeps seeking and flushing its decoder. `Prefetch(1)` after the source lets it
decode ahead sequentially in its own thread. Remaining limit: ProRes UHD decoding and prores_ks encoding (26.9 fps
alone) share the CPU.

Stage throughput (output fps; 25 = real time):

| stage | fps |
|---|---|
| ffms2 source alone (source fps / 2) | 17.9 |
| ffms2 + `Prefetch(1)` + CudaConvolution GPU | 34.6 |
| ffms2 + `Prefetch(1)` + CudaConvolution CPU (internal threads) | 21.1 |
| CudaConvolution GPU, filter only | 32-69 (PCIe bound, varies with load) |
| CudaConvolution CPU, filter only: `threads=1` / all 24 threads / `threads=1` + `Prefetch(16)` | 3.1 / 26.6 / 39.2 |
| prores_ks HQ 1080i encode alone | 26.9 |

## Agreement with the reference (all 1500 frames, `tools/compare_streams.py`)

| vs reference output | PSNR Y / U / V |
|---|---|
| CudaConvolution filter output (uncompressed) | 51.6 / 53.7 / 53.9 dB |
| CudaConvolution -> prores_ks HQ | 47.24 / 50.68 / 50.57 dB |
| fmtconv -> prores_ks HQ | 47.24 / 50.67 / 50.57 dB |

The ceiling is set by the reference file itself (ProRes coded, decoded here with ffmpeg instead of Apple's decoder);
on the uncompressed probes the same filter is luma bit-exact. Note: ffmpeg's `psnr` filter reported 38 dB for the
same raw data in this setup; the numbers above are computed directly from the decoded samples.
