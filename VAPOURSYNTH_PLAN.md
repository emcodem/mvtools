# VapourSynth for mvtools2 and cudafilters (AviSynth-API shim, one code base)

Moved out of `NVENCMEAnalyse.md` (Phase 0b) on 2026-10-31. Steps 1-3 done on branch `vs_shim` and merged into
`gpu_work` on 2026-10-31. User documentation: README section "VapourSynth"; decision log: `DECISIONS.md` D9. Phase 0 there (CMake as the only build system) is a prerequisite only for the
Windows build of the VS entry point; Linux CMake already builds everything.

User's decisions (2026-10-31):
- No VapourSynth rewrite of the filters (dubhater/vapoursynth-mvtools is a separate port). A small internal layer
  implements the part of the AviSynth+ C++ API that the filters use, on top of the VapourSynth API 4 C interface
  (`VapourSynth4.h`). Filter sources stay unchanged, the AviSynth+ build stays bit-identical, and under VapourSynth the
  same arithmetic runs, so output must match AviSynth+ bit for bit (same OS, same slicing).
- Namespaces: **`mvpf`** for mvtools2 (`core.mvpf.Analyse(...)`), **`cuda`** for cudafilters (`core.cuda.QTGMC(...)`),
  so this plugin and dubhater's `mv` can be loaded together.

## Step 1 outcome: how VapourSynth itself does it (read 2026-10-31)

Source read: vapoursynth master `ae0ca10` (2026-10-27, API 4.3), `src/avisynth/avisynth_compat.cpp` (1447 lines),
`src/core/vsapi.cpp`, `src/core/vsthreadpool.cpp`; dubhater/vapoursynth-mvtools master.

**VS's compat layer cannot load this plugin, so it is a reference, not a base.** `avscompat` implements the classic
AviSynth 2.6 interface (`AVISYNTH_INTERFACE_VERSION = 6`, no `AVS_Linkage` table, no `NewVideoFrameP`, frame props,
`GetEnvProperty`); our plugins need interface 8+ (cudafilters refuses below V8). It also drops the VideoInfo side
channel (`VSClip` rebuilds `VideoInfo` from `VSVideoInfo`, audio fields zeroed, `avisynth_compat.cpp:520`), so even a
2.6 mvtools build fails in MAnalyse with "not a super clip". What we take over from it:
- Filter getFrame model: `arInitial` requests the frames named by a prefetch rule for each input clip
  (`requestFrameFilter`), `arAllFramesReady` runs the AviSynth `GetFrame` with the frame context parked in the
  environment; the clip wrapper's `GetFrame(n)` then calls `getFrameFilter(n, node, ctx)`.
- Fallback when a requested frame was not prefetched: synchronous `vsapi->getFrame` plus a rate-limited warning.
  **This is safe in current VS** (no pool deadlock): `getFrame` from a pool thread marks the request `reserveThread`,
  and `VSThreadPool::startExternal` lowers `activeThreads` so another worker runs in its place
  (`vsthreadpool.cpp:471-490`, including the cache-flush case). Cost: the frame is computed serially. So a gap in a
  prefetch rule costs speed, never a hang (user's rule holds).
- Filter mode: `fmParallelRequests` when the filter has inputs, `fmFrameState` without (source-like).
- Its mvtools table (`avisynth_compat.cpp:733-752`) is approximate: `MFlowFps` / `MBlockFps` use `R1` with
  `div = mul = 1`, which is wrong for any rate change (output n needs source n·fb/fa), and `MDegrainN` / `tr` is
  missing. Ours is computed per filter from its arguments (below).

### Decision 1: frame requests: learned prefetch offsets, slow path as safety net

(Revised in step 2. The first plan was a hand-written rule table per function; reading the filters showed that the
frames they fetch depend on internal state: per-delta direction in MAnalyse, a static-frame mode, MCompensate taking
the delta from the vectors' header. A table would be fragile.)

Generic rule in `vsbridge/bridge.cpp`: for input k, base(n) = floor(n · frames_k / frames_out) (covers rate changes:
MFlowFps, MBlockFps, MAnalyse multi), plus a set of offsets learned per input, starting with {0}. `arInitial`
requests base(n) + every offset. When the filter fetches a frame that was not requested, the bridge adds that offset to
the set and serves the frame through the synchronous path. Only speed depends on this, never the output. Measured on
the UHD chain over 160 output frames: misses only during the warm-up, i.e. while VS has about one frame per thread
in flight before anything has been learned (last miss at output frame 23-28 with 24 threads), none after. `MVBRIDGE_DEBUG=1`
prints the misses and a summary per filter.

### Decision 2: VideoInfo side channel: registry keyed by `VSNode*`

In API 4 a `VSNode*` is the node object itself: `mapGetNode` returns another reference to the same pointer, and the
core inserts no cache nodes between filters (caching lives inside each node). So the shim keeps
`std::unordered_map<VSNode*, VideoInfo>` (mutex-protected) for every node it creates: the full AviSynth `VideoInfo`
the filter reported, `num_audio_samples` (MSuper params) and `nchannels` / `sample_type` (pointer to the
`MVAnalysisData` of the MAnalyse / MRecalculate / MScaleVect / MRestoreVect instance) included. The entry is erased
in the node's free callback; the instance that owns the `MVAnalysisData` lives as long as any consumer holds the node,
so the pointer stays valid. A node from a foreign VS filter (e.g. `std.Trim` on a super clip) is not in the registry
and gets a `VideoInfo` built from its `VSVideoInfo` with no side channel, so mvtools gives its normal "not a super /
vector clip" error. No frame-property fallback for now.

### Decision 3 (new, not in the earlier inventory): the shim provides the `AVS_Linkage` table

The AviSynth+ headers (interface 12 in `Sources/include/avisynth.h`) dispatch every non-virtual method of
`VideoInfo`, `VideoFrame`, `VideoFrameBuffer`, `PVideoFrame`, `PClip`, `AVSValue` and the frame-property helpers
through the global `AVS_linkage` pointer that the host hands over in `AvisynthPluginInit3`: 137 function pointers.
Under VS, `VapourSynthPluginInit2` sets `AVS_linkage` to the shim's own table before any filter runs. Content:
- `VideoInfo` methods (pure arithmetic: plane counts, subsampling, `BytesFromPixels`, `MulDivFPS`, ...) and `AVSValue`:
  taken from AviSynth+ `avs_core/core/interface.cpp` (GPL 2, same licence as mvtools2), so they behave identically.
- `VideoFrame`: the shim's subclass wraps a `const VSFrame*` / `VSFrame*` and fills offsets / pitches per plane, so
  `GetReadPtr` / `GetPitch` / `GetRowSize` / `GetHeight` stay the inline arithmetic of the header (as VS's
  `buildFrameView` does). Planar RGB: AviSynth plane order G,B,R vs VS R,G,B mapped here.
- `IScriptEnvironment` (+ `IScriptEnvironment2` / Neo parts the filters call): a class of the shim; everything not in
  the inventory below throws "not supported under VapourSynth: <method>".
The plugin keeps one global `AVS_linkage`; a process never has both hosts driving the same plugin instance.

### Decision 5: the shim is its own library, `mvtools_vsbridge`, a small AviSynth+ host for VS

The `AVS_Linkage` entries are member-function pointers of the header's own classes (`bool (VideoInfo::*)() const`,
...). AviSynth+ implements them by compiling `avisynth.h` in core mode (`BUILDING_AVSCORE`: real member definitions in
`interface.cpp`), while plugins compile the same header in plugin mode (inline members that call through the table).
Both modes in one binary break the one-definition rule: the strong core definitions would replace the plugin's inline
ones, and under AviSynth+ the plugin would then run our code on AviSynth+'s frames. So:
- `vsbridge/` builds `mvtools_vsbridge.dll` / `libmvtools_vsbridge.so` in core mode: AviSynth+'s `interface.cpp`
  (VideoInfo, AVSValue, PClip, PVideoFrame, VideoFrame accessors, the table), the environment, VS clip and frame
  wrappers, node registry, prefetch rules, argument translation. Exports only `mvbridge_register`.
- mvtools2 (later cudafilters) gets one small file, `VapourSynthEntry.cpp`: `VapourSynthPluginInit2` opens the bridge
  from the plugin's own folder (`dyn_lib.h`) and calls `mvbridge_register(plugin, vspapi, id, namespace, name,
  &AvisynthPluginInit3)`. The bridge runs `AvisynthPluginInit3` with its environment and table, records every
  `AddFunction`, and registers a VS function per entry. Filter code is untouched; the AviSynth+ binary only gains one
  export.
- Bridge missing or broken: the plugin still configures its namespace and registers no functions (and writes the
  reason to stderr), so VS loads without error and `core.mvpf` is empty. Never a crash (user's rule).
- VS's plugin autoloader ignores libraries without `VapourSynthPluginInit2` (`VSNoEntryPointException` is swallowed,
  `vscore.cpp:2332`), so the bridge can sit in the plugin folder next to mvtools2.
- One bridge instance per process serves every plugin that registers through it, so `Invoke("MSuper")` from
  cudafilters (step 4) is a direct call into mvtools2's registered function, as in AviSynth+.

Frame memory: VS allocates each plane separately (`vscore.cpp:220-251`); AviSynth+'s `VideoFrame` addresses planes
as `int` offsets from one buffer. The bridge wraps a VS frame without copying when the plane distances fit in `int`,
otherwise it copies into one contiguous buffer (input) / copies back at return (output), counted and logged once.

### Decision 6: function and argument names

VS functions keep the AviSynth names (`core.mvpf.MAnalyse`, `core.mvpf.MDegrain2`) and the AviSynth parameter names
(`thSAD`, `blksize`), so mvtools2's documentation applies unchanged. Positional parameters without a name in the
AviSynth string get names from a table in the bridge (`MDegrain2`: `clip, super, mvbw, mvfw, mvbw2, mvfw2`;
`MAnalyse`: `super`; `MDegrainN`: `clip, super, mvmulti, tr`); otherwise `c1`, `c2`, ... as VS's compat layer does.
`c+` (MStoreVect) becomes a clip array.

### Decision 4: vector clips travel as Gray8 carriers; super clips need nothing

Super clips are ordinary planar YUV / Y frames: same format in VS. Vector clips are **packed** clips:
`ClipFnc::format_vector_clip` (`ClipFnc.cpp:77-138`, used by MAnalyse, MRecalculate, MStoreVect, MRestoreVect,
CudaNVOFAnalyse) sets `CS_BGR32` (or BGR24 / YUY2 on request), width = data bytes / 4, height 1 (MStoreVect:
several rows). VS has no packed formats, so the shim maps such a node to a **Gray8 carrier with width × bytes per
pixel** (BGR32 ×4, BGR24 ×3, YUY2 ×2): every row holds the same bytes, and the filters only see the AviSynth
`VideoInfo` from the registry (decision 2), so `GetRowSize` / `GetPitch` come out as under AviSynth. A packed
`VideoInfo` without a registry entry cannot occur (no VS source produces one); MRestoreVect from a stored file under VS
accepts a Gray8 clip of the right width (later). VS strides are aligned to 32 bytes (64 with AVX-512,
`vscore.cpp:3067-3072`) vs AviSynth's 64: no filter may rely on the pitch being ≥ AviSynth's; checked in step 2 for
the vector writers (they write `size_bytes` ≤ row size, so the pitch does not matter).

For reference: dubhater's port carries `MVTools_MVAnalysisData` and `MVTools_vectors` as frame properties
(`vsmv/src/MVAnalyse.cpp:226-234`) and leaves the pixels unused, which survives `std.Trim` / `std.Splice` on vector
clips. Our registry does not (decision 2); the frame-property carrier stays the fallback if that turns out to matter.

## What the shim has to cover (inventory 2026-10-31 over `Sources/` and `cudafilters/`)

- `IClip` / `PClip`: `GetFrame`, `GetVideoInfo`, `GetParity`, `SetCacheHints` (MT modes become VS filter modes),
  `GetAudio` (no-op).
- `VideoFrame` / `PVideoFrame`: `GetReadPtr`, `GetWritePtr`, `GetPitch`, `GetRowSize`, `GetHeight` per plane over a
  `VSFrame`; `IsWritable` / `MakeWritable`, `SubFrame` (3 uses).
- `IScriptEnvironment`: `ThrowError` (225 uses, becomes `setError` + filter abort), `NewVideoFrame` / `NewVideoFrameP`
  (74), `BitBlt` (28), `GetCPUFlags` (22), `CheckVersion` (28), `copyFrameProps`, `getFramePropsRO`, `propGetInt`,
  `propNumElements`, `GetEnvProperty` (interface version), `Allocate`, `SetVar` (5), `FunctionExists` (2),
  `AcquireGlobalLock` / `ReleaseGlobalLock` (FFTW plan creation, becomes a process mutex), `Invoke` (21, see below).
- `AVSValue` / argument parsing: the AviSynth parameter strings of `AddFunction` (`"c[blksize]i[overlap]i..."`) are
  translated automatically into VS argument strings (`clip:vnode;blksize:int:opt;...`), so each filter is registered
  once for both hosts.
- Pixel types: the VS formats with an AviSynth equivalent (Gray, YUV 4:2:0 / 4:2:2 / 4:4:4, planar RGB, 8-16 bit and
  float where the filter supports it). YUY2 and packed RGB do not exist in VS and are rejected at creation.

## Remaining design problems (in order of risk)

1. **`Invoke`.** Done for mvtools2: `InternalCache` (MFlowBlur / MFlowFps / MFlowInter) is a pass-through; `Invoke`
   of a function registered through the bridge calls it directly (positional and named arguments). `MSuper` /
   `MAnalyse` from CudaNVOFAnalyse will go that way. The CPU fallbacks of cudafilters: stand-ins (findings of step 4);
   so far `Merge` / `MergeLuma` / `MergeChroma`, the others follow (VapourSynth todo list).
2. **Threading / MT modes.** Done: `MT_NICE_FILTER` → `fmParallel` with one instance; `MT_MULTI_INSTANCE` (almost
   every mvtools filter) → `fmParallel` with an instance pool: an idle instance per call, a new one created on demand
   (one at a time per filter), as AviSynth+ creates one per prefetch thread; `MT_SERIALIZED` (MAnalyse with
   `temporal` or `outfile`) and unknown → `fmParallelRequests`, one instance. `MVBRIDGE_SERIAL=MAnalyse,...` (or
   `all`) forces the serialised mode, for debugging. avstp is Windows-only and AviSynth-only; under VS the CPU search
   runs single-sliced as on Linux AviSynth+ (GPU: `MVTOOLS_GPU_SLICES`).
3. **GPU token frames.** mvtools2's GPU path done (findings of the GPU step below); token frames themselves only
   appear with cudafilters (`CudaUpload`), so their end-to-end test comes with step 4. VS now has its own GPU frames
   (Vulkan, `ffGPUOutput`, `vnode:gpu`, API 4.2+); token frames stay CPU-resident VS frames for now, interop with
   Vulkan GPU frames is a later item.

## Findings of step 2 / 3 (2026-10-31)

- **One pitch per format.** mvtools assumes that all frames of one format share one pitch: `FlowInterExtra` reads
  the src frame with the ref frame's pitch (`MVFlowInter.cpp:437`). The bridge first copied frames whose VS planes
  are too far apart for int offsets (happens under memory pressure, 24 threads) into buffers with 64-aligned pitches,
  while zero-copy frames keep VS's 32-aligned stride; MFlowInter then read past the Y plane (ASan: 1 byte after an
  8.49 MB VS plane) and crashed or gave different output, only at high thread counts. The copy now keeps the VS
  stride. Rule for any future frame path in the bridge: never change the pitch of a format.
- **Unwritten bytes in vector clips.** MAnalyse writes 4 + sizeof(MVAnalysisData) = 96 bytes of its 256-byte header;
  bytes 96..255 were whatever the frame memory held (deterministic under AviSynth+, not under VS). New frames of packed
  (vector) formats are zero-filled now; the vector clips are byte-identical to AviSynth+'s.
- **Argument names.** AviSynth names are case-insensitive, VS's are not, and mvtools2 is not consistent (MRecalculate
  `thsad`, MCompensate / MDegrain `thSAD`). The bridge registers each name in the casing most frequent among the
  plugin's functions, so the documented casing works everywhere.
- Planar RGB: mvtools2's MAnalyse accepts YUV only, so no mvtools2 chain uses the RGB plane mapping; it gets tested
  with cudafilters.
- Bridge missing next to libmvtools2.so: VS loads, `core.mvpf` exists with no functions, the reason is on stderr.

Test tools (`vsbridge/test/`): `vsmd5.py` (framemd5 like `avsmd5`; planar RGB hashed G, B, R as AviSynth),
`vstest.py` (`raw_source` = avsmd5's RawSource, `load_mvtools`), `gen_inputs.py` (HD 4:2:0 8-bit and 4:4:4 10-bit raw
from the UHD source), `compare.py` (each chain in both hosts with the same libmvtools2.so, framemd5 diff),
`lin_cpu.vpy` / `lin_gpu.vpy` (the stored Linux reference chains; `MVPF_LOOP` for longer clips). `compare.py cpu` /
`compare.py gpu` run one group; the GPU chains need `libmvtools_cuda.so` in `MVPF_PLUGIN_DIR` (without it they test the
fallback).

Result 2026-10-31 (WSL, VS R81, AviSynth+ 3.7.5, CPU): 17 chains (vectors, MStoreVect / MRestoreVect, MCompensate,
MDegrain2, MDegrainN, MFlow, MFlowInter, MFlowBlur, MFlowFps, MBlockFps, MMask, MSCDetection, MShow, MScaleVect,
MRecalculate, the UHD lin_cpu chain) give identical framemd5 in both hosts, 3 rounds at 24 VS threads; lin_cpu also
at 1 thread.

## Findings of the GPU step (mvtools2 `gpu=true`, 2026-10-31)

- **Token frames through the bridge (code review; tested end to end with cudafilters in step 4).** A token frame is an
  ordinary frame whose first luma bytes hold the `MvCudaTokenHeader`; the device frame lives in mvtools_cuda's registry
  with its own budget and is made again by its producer after eviction. Nothing in the bridge depends on that: input
  frames are wrapped without copying or copied byte for byte (header included), output frames are handed to VS as they
  are or copied row by row, VS's cache only holds references to immutable frames, and the bridge marks a frame read-only
  once VS has it. How long VS keeps a token frame changes only how often a device frame is made again, never the
  result. Regeneration: the producer instance runs inside another filter's `GetFrame` (`mvcuda_token_regenerate` ->
  `gpu_regen`, guarded by the instance's own mutex); its input fetches are not in that filter's request set and take the
  synchronous path. The learned offsets are only updated for an input of the filter whose `GetFrame` is running.
- **Shared uploads need one IClip per node.** The GPU filters upload each super picture once and share it under the key
  `GpuBackend::frame_key(child, n)`, i.e. the consumer's input `IClip*`. In AviSynth+ every consumer of a script
  variable holds the same clip; the bridge made one `VSClip` per consumer argument, so each consumer uploaded its own
  copy (`gpu_degrain2`, 40 frames: 246 uploads under VS vs 80 under AviSynth+; output identical). The bridge now keeps one
  `VSClip` per `VSNode*` (`acquire_vsclip` / `release_vsclip`: the map holds a reference of its own plus a count of the
  filters using it, so a lookup never revives a clip whose last reference is going away); the learned prefetch moved
  from the clip to the consuming filter (`BridgeFilter::input_of`). Now 80 uploads at 1 VS thread (= AviSynth+), 99 at
  24 threads (more frames in flight, more evictions).
- **Instances.** Every MT_MULTI_INSTANCE GPU filter gets up to one instance per VS thread (93 MAnalyse instances in
  `gpu_degrain2` at 24 threads, vs 4 in single-threaded AviSynth+); they share the device frame registry, so this costs
  host memory, not device frames.
- **Broken `libmvtools_cuda.so`.** A truncated library made glibc's `dlopen` die of SIGBUS (it maps segments past the end
  of the file), in AviSynth+ as well as VS. `dynlib::open` (`Sources/cuda/dyn_lib.h`, used by mvtools2, cudafilters,
  the NVOFA loader and the VS entry point) now refuses on Linux a file that is shorter than its ELF segments, with the
  reason in the log; the filters then run on the CPU.

Result 2026-10-31 (WSL, VS R81, AviSynth+ 3.7.5, RTX 5090, CUDA 13.4): 7 GPU chains (`compare.py gpu`: GPU vectors on
a lite and a normal super, MCompensate, MDegrain2, MRecalculate + MCompensate, MFlowFps `mask=2` on UHD 4:2:2 10-bit,
the stored `lin_gpu` chain) identical in both hosts, 4 rounds at 24 VS threads and 1 round at 1 thread; VS `lin_gpu` =
the stored AviSynth+ reference `lin_gpu.md5`. Without `libmvtools_cuda.so` and with a truncated one, both hosts run the
CPU fallback and stay identical (`lin_gpu` = `lin_cpu.md5`). python and vspipe exit cleanly (rc 0, no hang). The 17 CPU
chains still identical, 3 rounds.

## Findings of step 4 (cudafilters, in progress, 2026-10-31)

- **Entry point.** `cudafilters/VapourSynthEntry.cpp` (namespace `cuda`), a copy of mvtools2's: loads the bridge from its
  own folder, empty namespace plus the reason on stderr when it is missing. CMake picks it up by glob, the export map
  already lists `VapourSynthPluginInit2`.
- **Every cudafilter builds its CPU fallback in the constructor**, also when the GPU will compute every frame:
  `_cpu = env->Invoke("Merge", ...)` (`CudaMerge.cpp:43`, likewise Bob, TemporalSoften, Gauss/BilinearResize,
  RemoveGrain, nnedi3, mt_logic / mt_clamp / mt_expand / mt_inpand, QTGMC_KeepOnlyBobShimmerFixes) and takes `vi` from
  it. Fallback frames are pulled only when the GPU is unusable (library / device missing, format not covered) or fails
  for one frame (`make() != 0`: device allocation failed after freeing the pools, kernel error). An evicted token is
  not such a failure: the consumer has its producer make it again. Under VS these AviSynth core / plugin filters do not
  exist, so a missing `Invoke` target would also break the GPU path.
- **Stand-ins** (`vsbridge/bridge.cpp`, `StandInClip`): `Invoke` of such a name returns a clip with the VideoInfo the
  AviSynth+ filter reports (Merge*: the first clip's). On the GPU path it is never asked for a frame, so the output stays
  bit-identical. A fallback frame depends on the global switch `MVBRIDGE_FALLBACK` (read when the stand-in is created):
  `strict` (default) gives an error naming the filter and the causes (VS output = AviSynth+ output or an error, never
  different); `nearest` builds the closest VS filter on the stand-in's inputs (Merge*: `std.Merge` with per-plane
  weights), writes one line per stand-in to stderr at its first frame, and is not bit-identical in general. Names without
  a nearest mapping stay an error in both modes. For `nearest` the bridge turns AviSynth clips back into VS nodes
  (`clip_node`: one instance, no declared inputs, its fetches take the synchronous path).
- **Lookup tables from masktools (decided 2026-11-01, `DECISIONS.md` D10).** CudaKobs, CudaLutxy, CudaQtgmcSharpLimit
  (smode 1 / 2) and CudaQtgmcAddNoise read their device tables from a real `mt_lutxy` (AddNoise: `mt_adddiff(x,
  mt_lut(y))`) frame on an n x n pattern at creation (`lut_from_cpu`, `CudaKobs.cpp:103`), so a stand-in cannot help:
  whatever builds that table decides the GPU output. Findings that decided it:
  - masktools' `use_expr` chooses who computes: 0, 1 at 8 bit and 2 without realtime (8 / 10 / 12 bit lutxy, any
    lut) use masktools' own evaluator (double, comparisons 1 / -1, `==` within 1e-6, `scalef` = x * max / 255 with the
    luma rule on every plane, store clamp + round half up); 1 above 8 bit, 2 with realtime (lutxy from 14 bit) and 3
    call AviSynth+ Expr (float32, comparisons 1 / 0, `scalef` with the chroma rule (x - 128) * 511.5 / 127.5 + 512 on
    U / V, the JIT's `round` = half to even, store trunc(clamp + 0.5f)). So chroma tables differ from luma tables only
    on the Expr path.
  - VS `std.Expr` is not a substitute: half-to-even store and no `scalef` (sharp 0.75 at 10 bit: 74898 of 1M entries
    off by one; kobs1 at 10 bit off by up to 507 with a naive `scalef`).
  - masktools quirk: `dupN` with N >= 1 drops the old top of stack (`[a, b] dup1` -> `[a, a, a]`; Expr: `[a, b, a]`).
  - AviSynth+ crashes (not an error) on an unbalanced expression handed over by masktools; the port refuses it.
  Done: `vsbridge/mt_port.cpp` serves `mt_lut`, `mt_lutxy`, `mt_adddiff` from `Invoke` (before the stand-ins, in both
  `MVBRIDGE_FALLBACK` modes, because it is exact). Plane modes 1 / 2 / 3 (/ 4); memset, offsets, `scale_inputs` and
  formats above 16 bit are refused. Tables up to 10 bit are built once per expression; above that the expression is
  evaluated per pixel (CPU fallback only, the GPU tables stop at 10 bit). The float32 code is compiled with
  `-ffp-contract=off` (Expr does not fuse multiply-add).

Result 2026-10-31 (WSL, VS R81, AviSynth+ 3.7.5, RTX 5090): `compare.py cf`, 7 chains (CudaUpload / CudaDownload round
trip; CudaMerge 0.3, 0.5, CudaMergeLuma 0.7, CudaMergeChroma 0.6 on three formats; CudaUpload -> MSuper / MAnalyse /
MDegrain2 `gpu=true` -> CudaDownload; MFlowFps `gpu=true` on tokens merged with a CPU MFlowFps clip) identical in both
hosts at 24 and 1 VS threads, with `MVCUDA_CACHE_MB=16` and `=1` (cf_merge at 1 MB: 160 evictions, 41 regenerations,
nested through CudaMerge to CudaUpload; no fallback frame, the strict stand-in would have failed), and with
`MVBRIDGE_FALLBACK=nearest` (no nearest frame used). Without `libmvtools_cuda.so` or with a truncated one: AviSynth+
runs its CPU Merge; VS strict stops with the error; VS nearest runs (cf_merge 0.3 and cf_mergeluma 0.7 differ from
AviSynth+ in every frame, 0.5, MergeChroma 0.6 and the 10-bit 0.4 merge happened to match; not measured further, todo);
python and vspipe exit cleanly. `compare.py cpu` (17) and `gpu` (7) still identical after the `create_filter` refactor.

Result 2026-11-01 (masktools port): `vsbridge/test/mt_tables.py`, 94 cases (the 8 QTGMC_gpu table expressions, operator
coverage for both evaluators, AddNoise chains, plane modes 1 / 2 / 3 / 4 on 4:2:0; full tables at 8 / 10 bit, 1024 x
1024 spreads at 12 / 16 bit), references from masktools2 2.2.30 + AviSynth+ 3.7.5 on Windows (`mt_tables.md5`):
identical without `libmvtools_cuda.so` (port as CPU fallback, 94 cases) and with it (70 cases on the GPU from the port's
tables, the rest on the port). `compare.py cf` (7), `gpu` (7), `cpu` (17) still identical.

## VapourSynth todo list

- Measure `MVBRIDGE_FALLBACK=nearest` per mapping against AviSynth+ (runs without `libmvtools_cuda.so` send every frame
  through the fallback): differing frames, share of differing pixels, largest difference; document per filter.
- Stand-ins for the other fallback names (Bob, TemporalSoften, Gauss/BilinearResize, RemoveGrain, nnedi3, mt_logic /
  mt_clamp / mt_expand / mt_inpand / mt_makediff, QTGMC_KeepOnlyBobShimmerFixes, CudaQtgmc* chains): VideoInfo rule
  each, nearest mapping where one exists (std.AverageFrames, resize.*, rgvs / zsmooth, znedi3, std.Maximum / Minimum /
  Expr).
- CudaNVOFAnalyse through the bridge (Invoke of MSuper / MAnalyse), planar RGB test.

## Files

`vsbridge/` (decision 5: `avs_interface.cpp` and `avs_cpuid.cpp` from AviSynth+, `bridge.cpp`, `bridge_core.h`,
`mt_port.cpp` / `mt_port.h` (masktools port, D10), `exports.map`, `CMakeLists.txt`, `test/`),
`Sources/VapourSynthEntry.cpp` (and one in cudafilters): `VapourSynthPluginInit2` next to `AvisynthPluginInit3` (one library
per plugin, each host ignores the other entry point), `cmake/plugin_exports.map` exports both, `VapourSynth4.h`
(+ `VSHelper4.h` if used) vendored under `Sources/include/` (no build dependency on an installed VapourSynth). Option
`MVTOOLS_VAPOURSYNTH=ON`.

## Steps

1. ~~Read VS's `avisynth_compat.cpp` and the API 4 rules for frame requests inside filters; decide the request hook
   and the VideoInfo side channel.~~ Done 2026-10-31 (above).
2. ~~Shim core: `AVS_Linkage` table, environment, clip / frame wrappers, arg translation, node registry; first filters
   MSuper, MAnalyse, MCompensate.~~ Done 2026-10-31 on Linux (findings above).
3. ~~MDegrain1..N, MRecalculate, MFlowFps / MFlowInter / MBlockFps (temporal requests), then the rest of
   `Interface.cpp`.~~ Done 2026-10-31 on Linux, CPU path; GPU path (`gpu=true`) done 2026-10-31 (findings above).
   Open: MDepan, Windows build (needs Phase 0 step 3 of NVENCMEAnalyse.md: CMake on Windows).
4. cudafilters through the same shim; CudaNVOFAnalyse. In progress: entry point, token frames end to end, Merge*
   stand-ins with `MVBRIDGE_FALLBACK` done 2026-10-31, masktools port (`mt_lut` / `mt_lutxy` / `mt_adddiff`, D10)
   2026-11-01 (findings above); the rest is on the todo list.
5. DePan / DePanEstimate (optional, same shim).
6. Docs: README (VS usage, namespace, formats), `Documentation/`.

## Verification

- The same chains in AviSynth+ and VS on the same OS give identical framemd5 (Linux: `avsmd5` vs `vspipe` → framemd5;
  Windows: AviSynth+ vs VS portable). Chains: the Linux test chain (MAnalyse + MRecalculate + MFlowFps, UHD 10-bit
  4:2:2), MDegrain2 / MDegrainN, MCompensate, planar RGB, 4:2:0 8-bit, GPU on and off, NVOFA.
- Prefetch misses only in the warm-up (decision 1).
- Multi-threaded VS (`core.num_threads` 1 vs 24) gives the same md5; 3 runs stable; no hang at exit (the submitter
  atexit fix applies to VS too, the host process is python / vspipe).
- Fallbacks under VS: libmvtools_cuda missing, NVOFA missing, a cudafilters CPU fallback without its VS equivalent →
  a clear error, never a crash or hang.
- AviSynth+ builds: unchanged md5s (golden 85/85, run_partm 27/27, run_cf_regression, uc_FRC).

Effect on the nvencme plan: once this shim exists, nvencme's VS front end registers through it and produces this
repo's vector clips, so the research item on dubhater's vector format is only needed if interoperability with `mv.*`
is still wanted.
