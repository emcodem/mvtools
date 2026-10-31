# VapourSynth for mvtools2 and cudafilters (AviSynth-API shim, one code base)

Moved out of `NVENCMEAnalyse.md` (Phase 0b) on 2026-10-31; work on branch `vs_shim` (worktree), merged into
`gpu_work` when a step is verified. Phase 0 there (CMake as the only build system) is a prerequisite only for the
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

### Decision 1: frame requests: prefetch rule per filter, computed at creation, slow path as safety net

`Sources/vs/vs_prefetch.cpp`: table keyed by the AviSynth function name. Each entry is a function
`(args, input VideoInfos) -> per-input rule`, where a rule maps output frame n to a list of input frames. The rule
is evaluated once at creation from the filter's arguments, so filter sources stay unchanged. Examples:
- MSuper, MMask, MShow, MSCDetection, MFlow, MStoreVect, MScaleVect, most cudafilters: {n}.
- MAnalyse: super n, n±delta (multi: all deltas up to `delta`); MRecalculate: same on the super, {n} on the vectors.
- MCompensate / MDegrain1..6 / MDegrainN: super and source n-tr..n+tr, vectors {n} (tr from the call; MDegrainN `tr`).
- MFlowFps / MBlockFps / MFlowInter: source floor(n·fb/fa) and +1, vectors of that frame pair (from `num` / `den`
  and the source rate, as `MVFlowFps::GetFrame` computes it).
- Unknown function: {n} on every input.
A request outside the rule goes through the slow path with one warning per filter instance (name, n, input index),
which is how gaps in the table show up in testing; the test chains must run without that warning.

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

1. **`Invoke`.** `InternalCache` (MFlowBlur / MFlowFps / MFlowInter) becomes a pass-through. `MSuper` / `MAnalyse`
   from CudaNVOFAnalyse call this plugin's own filters through the shim. The CPU fallbacks of cudafilters (`Bob`,
   `TemporalSoften`, `RemoveGrain`, `mt_lutxy`, `nnedi3`, `QTGMC_KeepOnlyBobShimmerFixes`) map to VS equivalents where
   one exists (`std.*`, `rgvs.RemoveGrain`, `znedi3` / `nnedi3`, `std.Expr` for the LUT) and otherwise throw a clear
   error at creation; the GPU path does not need them.
2. **Threading / MT modes.** Default `fmParallelRequests` (see step 1). Filters with cross-frame state (MAnalyse with
   temporal predictor, MRecalculate chains with state) → `fmUnordered` / `fmFrameState`. avstp is Windows-only and
   AviSynth-only; under VS the CPU search runs single-sliced as on Linux AviSynth+ (GPU: `MVTOOLS_GPU_SLICES`).
3. **GPU token frames.** cudafilters and the mvtools GPU path pass token frames between filters; check that they
   survive VS's frame cache and copy-on-write (frame props copied, `MakeWritable` never on a token frame). VS now has
   its own GPU frames (Vulkan, `ffGPUOutput`, `vnode:gpu`, API 4.2+); token frames stay CPU-resident VS frames for
   now, interop with Vulkan GPU frames is a later item.

## Files

`vsbridge/` (decision 5: `avs_interface.cpp` from AviSynth+, `bridge.cpp`, `prefetch.cpp`, `CMakeLists.txt`),
`Sources/VapourSynthEntry.cpp` (and one in cudafilters): `VapourSynthPluginInit2` next to `AvisynthPluginInit3` (one library
per plugin, each host ignores the other entry point), `cmake/plugin_exports.map` exports both, `VapourSynth4.h`
(+ `VSHelper4.h` if used) vendored under `Sources/include/` (no build dependency on an installed VapourSynth). Option
`MVTOOLS_VAPOURSYNTH=ON`.

## Steps

1. ~~Read VS's `avisynth_compat.cpp` and the API 4 rules for frame requests inside filters; decide the request hook
   and the VideoInfo side channel.~~ Done 2026-10-31 (above).
2. Shim core: `AVS_Linkage` table, environment, clip / frame wrappers, arg translation, node registry; first filters
   MSuper, MAnalyse, MCompensate. Test host on Linux (WSL): VapourSynth from pip in the uv venv + `vspipe`.
3. MDegrain1..N, MRecalculate, MFlowFps / MFlowInter / MBlockFps (temporal requests), then the rest of `Interface.cpp`.
4. cudafilters through the same shim; CudaNVOFAnalyse.
5. DePan / DePanEstimate (optional, same shim).
6. Docs: README (VS usage, namespace, formats), `Documentation/`.

## Verification

- The same chains in AviSynth+ and VS on the same OS give identical framemd5 (Linux: `avsmd5` vs `vspipe` → framemd5;
  Windows: AviSynth+ vs VS portable). Chains: the Linux test chain (MAnalyse + MRecalculate + MFlowFps, UHD 10-bit
  4:2:2), MDegrain2 / MDegrainN, MCompensate, planar RGB, 4:2:0 8-bit, GPU on and off, NVOFA.
- No "not prefetched" warning in any test chain.
- Multi-threaded VS (`core.num_threads` 1 vs 24) gives the same md5; 3 runs stable; no hang at exit (the submitter
  atexit fix applies to VS too, the host process is python / vspipe).
- Fallbacks under VS: libmvtools_cuda missing, NVOFA missing, a cudafilters CPU fallback without its VS equivalent →
  a clear error, never a crash or hang.
- AviSynth+ builds: unchanged md5s (golden 85/85, run_partm 27/27, run_cf_regression, uc_FRC).

Effect on the nvencme plan: once this shim exists, nvencme's VS front end registers through it and produces this
repo's vector clips, so the research item on dubhater's vector format is only needed if interoperability with `mv.*`
is still wanted.
