# SplatFps

`splatfps.dll` converts progressive video from any frame rate to any other by motion compensation. It is its own
plugin, built with this mvtools fork; its GPU code lives in `mvtools_cuda.dll`, which must sit in the same folder.
Under VapourSynth the functions are in the namespace `splat` (through `mvtools_vsbridge`, as for mvtools2).

| Function | Plugin | What it does |
| --- | --- | --- |
| `SplatFps(clip, num, den, vectors, exact, ...settings)` | splatfps | the conversion on the CPU (fast path, or the exact path) |
| `CudaSplatFps(clip, num, den, exact, vectors, ...settings)` | splatfps | the same on the GPU |
| `SplatAnalyse(clip, ...settings)` | splatfps | the built-in motion measurement as a candidate clip |
| `MSplatVectors(mvbw, mvfw, refine, thSCD1, thSCD2, me_scale, pad_x, pad_y, pyramid_levels)` | mvtools2 | MVTools vector clips as a candidate clip |

`...settings` are the optional named arguments of the section [Settings](#settings-named-arguments); without them
every value is the default for the clip's frame size.

## How it works

Output frame k sits at source time k * rate_in / rate_out. When that time is within 1/128 of a source picture the output
is that picture; otherwise it is built from the two source pictures around it, P before and N after:

The stages, in order: input (conversion to 4:4:4 float), ME pictures, phase correlation, refinement, match errors,
assignment, local fallback, schedule, pair statistic (scene cut, temporal fallback), build, output (conversion back to
the input format).

1. **Candidate vectors** for every cell of the "ME grid" (4 x 4 pixels from 1080p on, 2 x 2 up to 720p:
   `me_scale`) between each pair of neighbouring source pictures. The built-in source is phase correlation over
   blocks of three sizes (the classical television motion measurement: G. A. Thomas, BBC Research Department Report
   1987/11, https://downloads.bbc.co.uk/rd/pubs/reports/1987-11.pdf): each block offers its strongest peaks, every
   cell gets the vectors of the blocks covering it, plus the vector most blocks agree on, giving 29 candidates per
   cell with the default settings. Other sources can be plugged in through a candidate clip (`vectors=`, below).
2. **Refinement and assignment.** Each candidate is refined against the pictures (a Lucas-Kanade step), its matching
   error measured, and every cell gets one forward vector (into the next picture) and one backward vector (into the
   previous one), each the candidate with the lowest error plus cost (or chosen as a pair when `speed_change_cost` is
   set); where one side matches much worse than the other (an object covers or uncovers the background), that side
   takes the other side's vector.
3. **Fallbacks.** Where the vector field is locally inconsistent, the cells are blended towards the plain pictures. At
   a scene cut (the two pictures do not match at all) the output repeats the nearer source picture. An optional
   temporal fallback (off by default) moves the output time towards the nearer picture where the vectors match badly.
4. **Picture build.** Both source pictures are split into a Laplacian pyramid and each band is splatted (projected
   forward) along its vectors to the output time, weighted by temporal distance, then the pyramid is
   put back together.

## Formats and limits (so far)

- Planar YUV 4:4:4, 4:2:2 or 4:2:0, 8 to 16 bit integer samples (YUV420P8 ... YUV444P16; no alpha, no float, no
  RGB), progressive. The output has the format of the input, and output frames that fall on a source picture are that
  source frame, passed on unchanged.
- Chroma siting (`_ChromaLocation`, read from frame 0): left (0, MPEG-2: horizontally with the even luma samples) or
  centre (1, JPEG / MPEG-1: horizontally half way between two luma samples); 4:2:0 chroma vertically half way between
  two luma rows. 4:2:2 also accepts top left / bottom left (2, 4) and top / bottom (3, 5). 4:2:0 with chroma on the
  luma rows (2 .. 5) is refused with an error, and so is a clip whose siting changes; nothing is converted silently.
  Without the property, left siting is assumed.
- Internally every picture is 4:4:4 float: 4:2:0 chroma is interpolated vertically (3 taps at the quarter-row
  phases), 4:2:2 chroma horizontally (left siting: 14-tap half-band; centre siting: the same 3 taps as vertically);
  on the way out left-sited chroma takes the even samples and centre-sited chroma the half-band value half way between
  two columns, 4:2:0 additionally the half-band value half way between two rows, so a picture that passes through
  unchanged keeps almost every code. Output rounds to nearest.
- Any frame size whose ME picture ((width + 2 pad_x) / me_scale x (height + 2 pad_y) / me_scale) holds the largest
  phase-correlation block (4 x block_size, 128 by default); with the defaults that is about 224 x 224 pixels at
  scale 2, 480 x 480 at scale 4. The defaults per size are described under [Settings](#settings-named-arguments).
- The last one to five output frames of a clip are copies of the nearest source picture (the last source pair is not
  interpolated yet).
- `MT_SERIALIZED` (one frame at a time per instance).

## SplatFps(clip, int num, int den, clip vectors, bool exact = false, ...settings)

The CPU path.

- `num`, `den`: the output frame rate.
- `vectors` (optional): a candidate clip of the same clip (`SplatAnalyse`, `MSplatVectors`). Without it the built-in
  phase correlation is used.
- `exact = false` (default): the fast path: the same stages spread over all CPU cores, with AVX2 + FMA code where the CPU has it
  (a portable version otherwise). It does the same arithmetic in the same order as `exact = true`, so it gives the
  same frames; the one exception are exp, atan2, sin and cos, which it evaluates with its own polynomials, rounded to
  the same float result except in a few cases in 10^8, which can (rarely) change a motion decision and with it a few
  output samples. In the tests so far every frame was identical with `exact = true`.
- `exact = true`: the exact path: the deterministic scalar path: float32 in a fixed evaluation order, no
  contraction, IEEE division and sqrt, exp through double; the fast CPU path and CUDA (exact=true) are checked bit for
  bit against it. Scalar C++, mostly one thread, about 1.5 frames per second at 1080p. Use it for checks.

Speed, 1920 x 1080 4:2:2 10 bit, 50 -> 59.94 fps, built-in vectors, decoding included, 300 source pictures (360 output
frames), Core Ultra 9 285K (24 cores, 24 threads), High priority: `exact = true` 1.5 fps, `exact = false` 20 fps.

## CudaSplatFps(clip, int num, int den, bool exact = false, clip vectors, ...settings)

The GPU path; same parameters as `SplatFps`, returns normal frames.

- `exact = false` (default): fast arithmetic; a few output samples per frame differ from `SplatFps` by one code.
- `exact = true`: bit-identical with `SplatFps(exact = true)` (about 30 % slower than the fast mode).
- `vectors`: as for `SplatFps`; candidate frames are uploaded per source picture.

Speed, 1920 x 1080 4:2:2 10 bit, 50 -> 59.94 fps, RTX 5090, fast mode, decoding included; output frames per second
over a clip of 300 source pictures, including the start-up of about 2 s (on long clips the built-in path reaches about
140 fps):

| Vector source | fps |
| --- | --- |
| built-in | ~73 |
| NVOFA (`CudaNVOFAnalyse`, see below) | ~57 |
| `CudaMAnalyse(..., meander=false)` | ~42 |
| `MAnalyse` (CPU) | ~20 |

## SplatAnalyse(clip, ...settings)

The built-in phase correlation as a candidate clip (29 candidates per cell with the 1080p defaults). With the same
settings, `SplatFps(c, n, d, vectors=SplatAnalyse(c))` gives exactly `SplatFps(c, n, d)` (the candidate costs travel
in the clip); the clip is for inspection and for building other sources. CPU only. Of the settings, the geometry
(`me_scale`, `pad_x`, `pad_y`, `pyramid_levels`) must be the same as in the SplatFps call that reads the clip.

## MSplatVectors(clip mvbw, clip mvfw, bool refine = true, int thSCD1 = 400, int thSCD2 = 130, int me_scale, int pad_x, int pad_y, int pyramid_levels) (mvtools2.dll)

Turns MVTools vector clips into a candidate clip: `mvbw` = `MAnalyse(isb=true, delta=1)`, `mvfw` =
`MAnalyse(isb=false, delta=1)` of the clip `SplatFps` converts (the same pair `MFlowFps` takes). Each ME-grid cell takes
the vector of the block whose centre is nearest; SplatFps adds the zero vector as a second candidate.

- `refine = true`: SplatFps refines the vectors against the pictures; `false` uses them as they are (assignment only).
- `thSCD1`, `thSCD2`: MVTools' scene-change test as in `MFlowFps`; a frame that fails it gives zero vectors.
- `me_scale`, `pad_x`, `pad_y`, `pyramid_levels`: the ME grid the vectors are put on; give the same values as to
  SplatFps when you set them there (defaults: those of the picture size, as SplatFps). SplatFps checks the grid and
  refuses a clip made for another one.

Any filter that makes MAnalyse-format vector clips works: `MAnalyse`, `MRecalculate`, `CudaMAnalyse`,
`CudaMRecalculate`, `CudaNVOFAnalyse`. The vectors must be of the full-size picture (no `MScaleVect` mismatch).

## Examples

```
LoadPlugin("splatfps.dll")
c = LWLibavVideoSource("in.mov")             # YUV422P10 1920 x 1080
CudaSplatFps(c, 60000, 1001)                 # built-in vectors
```

MVTools vectors (CPU analysis):

```
LoadPlugin("mvtools2.dll")
sup = MSuper(c, pel=2, hpad=16, vpad=16)
bw = MAnalyse(sup, isb=true, delta=1, blksize=16, overlap=8)
fw = MAnalyse(sup, isb=false, delta=1, blksize=16, overlap=8)
CudaSplatFps(c, 60000, 1001, vectors=MSplatVectors(bw, fw))
```

The same with the GPU analysis: `CudaMSuper` / `CudaMAnalyse`. Use `meander=false` there: with the default
`meander=true` the GPU analysis is currently slower than the CPU one.

NVOFA (Turing or newer; `cudafilters.dll`): blocks of 4 x 4 match SplatFps' grid one to one; `hpad` / `vpad` limit the
vector length (use at least the fastest motion in pixels per frame), `pel` 1 or 2:

```
LoadPlugin("cudafilters.dll")
bw = CudaNVOFAnalyse(c, isb=true, delta=1, blksize=4, overlap=0, pel=2, hpad=64, vpad=64)
fw = CudaNVOFAnalyse(c, isb=false, delta=1, blksize=4, overlap=0, pel=2, hpad=64, vpad=64)
CudaSplatFps(c, 60000, 1001, vectors=MSplatVectors(bw, fw))
```

## Candidate clips

A candidate clip has one Y32 (float) frame per source picture. Its rows hold, for picture c, a forward set of n vector
fields (pointing from c into c+1) and, depending on the properties, a backward set, each field a vx and a vy plane of
the ME grid ((width + 2 pad_x) / me_scale x (height + 2 pad_y) / me_scale cells, values in cells = me_scale pixels;
at 1080p (width + 32) / 4 x (height + 40) / 4). Frame properties: `SplatCand` (layout version 1), `SplatN`
(candidates per set, at most 31), `SplatGridW` / `SplatGridH`, `SplatScale` / `SplatPadX` / `SplatPadY` (optional:
the grid's scale and borders, checked by SplatFps when present), `SplatFwd`, `SplatBwd` (0 none, 1 in this frame
pointing into c-1, 2 in this frame pointing from c-1 into c, 3 the forward set of frame c-1), `SplatRefine`,
`SplatBias` (n + 1 per-candidate costs, zero vector last). The layout is defined in `splatfps/cand_clip.h`; any filter
that writes it can be a vector source.

## Settings (named arguments)

`SplatFps`, `CudaSplatFps` and `SplatAnalyse` take the same optional named arguments (VapourSynth: the same names).
Each one left out keeps its default; most defaults are fixed numbers, the geometry and the ME filters follow the frame
size (below). Distances marked "ME px" are in ME pixels (cells of the ME grid), so the same number covers more picture
pixels at a larger `me_scale`. Arrays are written `[a, b, c]` (AviSynth+) or as lists (VapourSynth). Errors are mean
absolute sample differences in 8-bit units (0 .. 255, whatever the bit depth of the clip). The defaults live in
`splatfps/core/splat_config.h` (`config_defaults`, `config_for_size`).

### Defaults by frame size

| setting | rule | 720 x 576 | 1280 x 720 | 1920 x 1080 | 3840 x 2160 |
| --- | --- | --- | --- | --- | --- |
| `me_scale` | 2 below 900 rows, else 4 (the width does not matter; 8 only when set) | 2 | 2 | 4 | 4 |
| `pyramid_levels` | log2(me_scale) + 2 | 3 | 3 | 4 | 4 |
| `pad_x`, `pad_y` | per axis the smallest value >= 16 that makes size + 2 pad a multiple of 2^pyramid_levels (16 for an odd size) | 16, 16 | 16, 16 | 16, 20 | 16, 16 |
| ME grid | (size + 2 pad) / me_scale | 376 x 304 | 656 x 376 | 488 x 280 | 968 x 548 |
| `block_size` | 32 | 32 | 32 | 32 | 32 |
| `nearest_block` | auto: a stage whose block grid has one block on an axis | - | - | - | - |
| `field_stride_max` | = me_scale | 2 | 2 | 4 | 4 |
| ME filters | derived from me_scale (below) | 2:1 | 2:1 | 4:1 | 4:1 |

At UHD scale 4 measured clearly better than 8 (about 12 % on the test corpus) for about 7 % of speed. When you set
`me_scale` or `pyramid_levels`, the borders, block size, stride limit and filters that you do not set follow the
rules for your values.

### Geometry

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `me_scale` | int | by size | 2, 4, 8 | ME grid cell size: motion is measured on pictures decimated by this in both directions |
| `pad_x`, `pad_y` | int | by size | 0 .. 256 | border of edge-repeated pixels around the picture (internal grid). For an even width / height, size + 2 pad must be a multiple of 2^pyramid_levels and of me_scale. In the build, the border cells (pad / me_scale ME px) take the vector, error and weights of the nearest cell inside the picture, so motion is carried up to the picture edge |
| `pyramid_levels` | int | by size | 1 .. 7 | Laplacian-pyramid levels below the full-resolution band (bands = levels + 1) of the picture build |
| `field_stride_max` | int | me_scale | me_scale / 2 .. 2^pyramid_levels | pyramid levels with stride 2^i up to this read the up-sampled full-resolution vector field; coarser levels use their own reduced field |

### ME pictures (motion-measurement input)

The luma is low-pass filtered and decimated by `me_scale` twice: picture A (sharper; errors, refinement, local
fallback) and picture B (softer; phase correlation). Both are symmetric Kaiser-windowed sinc low-pass filters with DC
gain 1 (J. F. Kaiser, 1974), designed from the cutoff / taps / beta arguments below. By default the cutoffs are
fractions of the ME grid's Nyquist frequency (0.5 / me_scale cycles per picture pixel), the window is set for 60 dB
stop-band attenuation, and the taps follow Kaiser's length estimate for that attenuation with a transition width equal
to the cutoff.

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `me_filter_generated` | bool | true | true | kept for scripts; must be true |
| `me_filter_a_cutoff` | float | 0.75 x 0.5 / me_scale (0.09375 at scale 4) | (0, 0.5) | -6 dB point of filter A in cycles per picture pixel |
| `me_filter_b_cutoff` | float | 0.6 x 0.5 / me_scale (0.075 at scale 4) | (0, 0.5) | -6 dB point of filter B |
| `me_filter_a_taps` | int | from the cutoff: 11 / 21 / 40 at scale 2 / 4 / 8 | 1 .. 64 | taps per side of filter A, centre included (21 at scale 4: 41 taps) |
| `me_filter_b_taps` | int | from the cutoff: 14 / 26 / 50 at scale 2 / 4 / 8 | 1 .. 64 | taps per side of filter B (26 at scale 4: 51 taps) |
| `me_filter_beta` | float | 5.653 (0.1102 x (60 - 8.7)) | >= 0 | Kaiser window beta (higher: less ripple, wider transition) |

### Phase correlation (candidate vectors)

Three block stages of `block_size`, 2 `block_size` and 4 `block_size` with half-overlapping blocks on picture B. Each
block: its windowed mean removed, a tapered-cosine window, FFT; the cross-power spectrum of the two pictures' blocks
is reduced to its phase and weighted by how far each frequency stands above the block's noise floor (the median
amplitude); the inverse FFT gives a correlation surface whose highest peaks are the block's vectors. Every cell is
offered the vectors of the blocks that contain it (the near and the far block along each axis: four per peak), and
the global candidate is the vector most blocks agree on. The candidate count is the sum of `block_peaks` x 4 (x 1 for
a nearest-block stage), plus 1 for the global one (29 with the defaults); at most 31.

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `block_size` | int | by size (32) | 16, 32, 64, 128 | block size of the first stage |
| `block_peaks` | int[3] | [2, 2, 3] | 0 .. 4 each | correlation peaks taken per block in each stage (0: the stage is off) |
| `nearest_block` | int[3] | [-1, -1, -1] | -1, 0, 1 | per stage: 1 = each peak gives one candidate, the vector of the block whose centre is nearest to the cell; 0 = four candidates, the near and the far block covering the cell along each axis; -1 = auto (1 when the stage's grid has a single block on an axis) |
| `highpass_sigma` | float | 8 | 0 .. 16 | prefilter: picture B minus its Gaussian low-pass of this sigma (ME pixels), so slow shading does not dominate the spectra; 0 = off |
| `block_taper` | float | 0.5 | 0 .. 1 | block window: the tapered part of a block, both edges together (1 = a Hann window); the taper keeps the block edges from correlating with themselves |
| `noise_weight` | float | 1 | >= 0 | weighting of the spectrum: a frequency counts q / (q + noise_weight^2), q = its amplitude product over the blocks' noise floors; 0 = phase only (every frequency counts the same) |
| `peak_ratio` | float | 0 | 0 .. 1 | a block's further peak counts when it reaches this fraction of the block's strongest; otherwise the strongest takes its place |
| `global_candidate` | int | 1 | 0, 1 | 1 = add the vector most blocks agree on (weighted by peak height) as a candidate for every cell |
| `secondary_peak_cost` | float | 3 | >= 0 | cost added to the error of a candidate from a block's further peaks (the strongest peak costs nothing): such a peak is often an alias of the strongest one (a periodic structure shows peaks one period apart), so it must match clearly better to win |

### Refinement, errors, assignment

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `refine_err_scale` | float | 15 | > 0 | refinement weight exp(-error / this): pixels a candidate matches badly count less |
| `refine_reg` | float | 0.001 | | added to the squared gradients of the refinement (keeps flat areas solvable, pulls them to zero) |
| `refine_box` | int | 15 | odd, 1 .. 63 | window of the refinement sums, ME px |
| `error_box` | int | 7 | odd, 1 .. 31 | window of the mean absolute difference that is a candidate's error, ME px |
| `error_edge_weight` | float | 0.75 |  | weight w of the edge-aware error filter: (1 - w) e + w m, m from `error_edge_mode` |
| `error_edge_dist` | int | 3 | 0 .. 16 | distance of the neighbours that filter looks at, ME px |
| `error_edge_mode` | int | 1 | 0, 1 | 0: m = the mean of the smaller of the errors left / right and the smaller of those up / down; 1: shiftable windows, m = the smallest box error at the four diagonal shifts by error_edge_dist (as in area-based stereo matching) |
| `speed_change_cost` | float | 0 |  | 0: the forward and the backward vector are chosen independently, each the lowest error plus cost of its side. Above 0: they are chosen as a pair, and the pair pays this per ME px of the difference between the forward and the (turned) backward vector, which prefers constant motion |
| `vector_length_cost` | float | 0 | >= 0 | cost per ME px of the length (L1) of each chosen vector: equal matches go to the shorter vectors (the length bias of BBC RD 1987/11 section 4.2.2) |
| `occlusion_error` | float | 6 |  | occlusion: a side whose error is above this ... |
| `occlusion_error_factor` | float | 1.5 |  | ... and above this times the other side's error takes the other side's vector reversed (covered or uncovered background is interpolated from the side that sees it); the errors stay as measured |
| `global_cost` | float | 0 |  | cost of the global candidate |
| `zero_cost` | float | 0 |  | cost of the zero-vector candidate (always added after the others) |

### Local fallback

Where a vector differs from the Gaussian-smoothed vectors around it (an inconsistent field), motion compensation is
not trusted there: the build blends the two pictures in place with a weight that grows with the inconsistency. Weights
are in units of one full splat.

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `fallback_box` | int | 31 | odd, 1 .. 255 | window over which the inconsistency weight is spread (box mean), ME px |
| `fallback_dead_zone` | float | 3 | | inconsistency (distance of a vector from its smoothed neighbourhood, ME px) below this gives no weight |
| `fallback_range` | float | 2 | > 0 | ... the weight rises linearly over this much above it to its full value |
| `fallback_strength` | float | 1.5 | >= 0 | full weight of an inconsistent vector |
| `fallback_cap` | float | 1.25 | > 0 | cap of the spread weight |
| `fallback_err_gain` | float | 0 | >= 0 | error term: the weight also grows by this x clamp((e - fallback_err_lo) / fallback_err_range, 0, 1), e the assigned match error, where even the best vectors match badly (water, leaves); 0 = off |
| `fallback_err_lo`, `fallback_err_range` | float | 8, 16 | range > 0 | ... the error where it starts and over which it reaches the full gain |

### Schedule, scene cut, temporal fallback

The pair statistic m is the mean match error of the two fields that meet at an output frame (the forward field of P
and the backward field of N), each error capped at `error_window_cap`, over the ME grid without a border of
1 / `error_window_div` of its width and height on each side.

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `copy_snap` | int | 128 | 0 or >= 2 | output phases within 1 / copy_snap of a source picture are copies of it; 0: only exact hits |
| `error_window_div` | int | 8 | >= 3 | border of the statistic's window: 1 / this of the grid on each side |
| `error_window_cap` | float | 64 | > 0 | errors are capped at this in the statistic |
| `scene_cut_error` | float | 15 | >= 0 | scene cut: when m reaches this, the two pictures are taken as unrelated and the output is a copy of the nearer source picture; 0 = off |
| `pull_strength` | float | 0 | >= 0 | temporal fallback: the output time moves towards the nearer source picture by pull_strength x smoothstep(m, pull_error_lo, pull_error_hi) of its distance; 0 = off (the default: on the test corpus it cost more, as judder, than it saved) |
| `pull_error_lo`, `pull_error_hi` | float | 6, 18 | hi > lo | ... the statistic where the pull starts and where it reaches its full strength |

### Picture build

| argument | type | default | range | what it does |
| --- | --- | --- | --- | --- |
| `splat_err_scale` | float | 0 | >= 0 | splat weight splat_gain x max(splat_min_weight, exp(-error / this)) x temporal proximity; 0: no error weighting (splat_gain x the temporal proximity alone) |
| `splat_min_weight` | float | 0.02 | | lower limit of that error confidence |
| `splat_gain` | float | 1 | > 0 | gain of the splat weight against the local-fallback weight (both in units of one full splat) |
| `still_vector` | float | 1/128 (0.0078125) | | vectors up to this long (pixels of the pyramid level) are not splatted; their weight joins the in-place blend |
| `hole_weight` | float | 1e-4 | >= 0 | output samples with less total weight are holes and get the plain cross-fade |
| `pyramid_sigma` | float | 4 | 0.3 .. 16 | width of the pyramid's reduce filter (a 7-tap sampled Gaussian): larger keeps less detail in the coarse bands, which move with averaged vectors, and more in the fine bands, which move with the full-resolution vectors |
