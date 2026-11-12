# UConvert - one AviSynth+ entry point for source, deinterlacing, frame-rate conversion and scaling.
# Tests: tests\run_tests.py and tests\run_matrix.py (their media sit outside the repo, see tests\ucdata.py).
# Design and decision tables: DESIGN.md.
#
#   UConvert(file, out_h, out_w, out_interlace, out_rate_nom, out_rate_den, ...)
#   UConvertClip(clip, out_h, out_w, out_interlace, out_rate_nom, out_rate_den, ...)   (same, for an opened clip)
#
# out_interlace: "tff" | "bff"; not specified (or "") = progressive. out_rate_*: frame rate, never field rate.
# The output always has exactly the requested size, scan and frame rate. Colour is never touched: the output
# has the input's pixel format and colour properties. Audio is passed through as PCM (trimmed/padded to the
# video length; resampled to 48 kHz only when merged streams have different sample rates).
#
# gpu=true (default): deinterlacing (QTGMC with its GPU switches), frame-rate conversion (CudaSplatFps) and the
# interlace step (CudaConvolution) run on an NVIDIA GPU. Without a usable GPU UConvert stops with an error; gpu=false
# runs the CPU chain (QTGMC, MVTools MBlockFps, mt_convolution). Inside the GPU chain, single stages that have no GPU
# path for a format (e.g. greyscale or alpha in QTGMC) run on the CPU with the same result (cuda_cpu_fallback).
#
# Required plugins: MxfSource, LSMASHSource, MVTools fork gpu_work (mvtools2 + mvtools_cuda + avstp, cudafilters,
# splatfps; one build, one folder), CudaConvolution 0.10+, masktools2, RgTools, plugins_JPSDR, NNEDI3CL, avsresize,
# QTGMC_gpu.avsi + Zs_RF_Shared.avsi. In the redistributable folder (build_dist.py) they sit in dlls\ next to this
# file and are loaded from there; otherwise the caller loads them (load_plugins.avs). uc_interlace.json must sit next
# to this file.

global uc_dir = ScriptDir()
Exist(uc_dir + "dlls\uc_load.avs") ? Import(uc_dir + "dlls\uc_load.avs") : NOP()

# 9-tap vertical low-pass kernel of the interlace step, as integers / 4096.
# Set to "" after the Import to take the fields without filtering. The GPU path reads the same taps from
# uc_interlace.json (uc_interlace_preset); a changed kernel string needs a matching preset file there.
global uc_interlace_kernel = "31 -66 -235 1075 2486 1075 -235 -66 31"
global uc_interlace_preset = uc_dir + "uc_interlace.json"
global uc_interlace_preset_kernel = "31 -66 -235 1075 2486 1075 -235 -66 31"   # the taps in uc_interlace.json

# GPU QTGMC motion vectors from the NVIDIA optical flow accelerator (faster, not identical to the CPU search).
# false: the GPU search gives the same vectors as the CPU QTGMC.
global uc_qtgmc_nvof = false

# QTGMC preset of the deinterlacer (both chains). "Super Fast" is about 1.5x faster on the GPU chain, at lower quality.
global uc_qtgmc_preset = "Fast"

# default of the gpu parameter (set to false after the Import to run every UConvert call on the CPU chain)
global uc_gpu_default = true


function UConvert(string file, int out_h, int out_w, string "out_interlace", int "out_rate_nom", int "out_rate_den",
\                 string "source", string "cachedir", string "scan_in", int "sar_num", int "sar_den",
\                 int "out_sar_num", int "out_sar_den", string "aspect_mode", bool "frc_8bit", int "threads",
\                 bool "gpu")
{
    threads = Default(threads, 3)
    c = uc_Source(file, source, cachedir)
    c = threads > 0 ? c.Prefetch(3) : c
    return UConvertClip(c, out_h, out_w, out_interlace, out_rate_nom, out_rate_den, scan_in=scan_in,
    \                   sar_num=sar_num, sar_den=sar_den, out_sar_num=out_sar_num, out_sar_den=out_sar_den,
    \                   aspect_mode=aspect_mode, frc_8bit=frc_8bit, threads=threads, gpu=gpu)
}


# threads > 0: UConvertClip places the thread pool itself - Prefetch(threads) on the progressive chain before the
# interlace step and Prefetch(3) at the end. Do not add a large Prefetch after it: with the pool behind the
# interlace step, threads compute the same deint/FRC frames twice (measured: 25i output at half speed).
# GPU frame-rate conversion: CudaSplatFps takes one frame at a time and wants its frames in order, so the pool sits
# in front of it instead, with Prefetch(1) after it and at the end (each stage in its own thread).
function UConvertClip(clip c, int out_h, int out_w, string "out_interlace", int "out_rate_nom", int "out_rate_den",
\                     string "scan_in", int "sar_num", int "sar_den", int "out_sar_num", int "out_sar_den",
\                     string "aspect_mode", bool "frc_8bit", int "threads", bool "gpu")
{
    threads = Default(threads, 0)
    gpu = Default(gpu, uc_gpu_default)
    Assert(Defined(out_rate_nom) && Defined(out_rate_den) && out_rate_nom > 0 && out_rate_den > 0,
    \      "UConvert: out_rate_nom and out_rate_den are required")
    oi = LCase(Default(out_interlace, ""))
    Assert(oi == "" || oi == "tff" || oi == "bff", "UConvert: out_interlace must be 'tff', 'bff' or not specified")
    aspect_mode = LCase(Default(aspect_mode, "pillar"))
    Assert(aspect_mode == "pillar" || aspect_mode == "crop" || aspect_mode == "stretch",
    \      "UConvert: aspect_mode must be 'pillar', 'crop' or 'stretch'")
    frc_8bit = Default(frc_8bit, true)

    # ---- probe (caller overrides win); a missing or invalid input SAR counts as square pixels
    src = c
    scan = Defined(scan_in) ? LCase(scan_in) : uc_ScanOf(c)
    Assert(scan == "p" || scan == "tff" || scan == "bff", "UConvert: scan_in must be 'p', 'tff' or 'bff'")
    sar_n = Defined(sar_num) ? sar_num : uc_PropInt(c, "_SARNum", 1)
    sar_d = Defined(sar_den) ? sar_den : uc_PropInt(c, "_SARDen", 1)
    sar_ok = sar_n > 0 && sar_d > 0
    sar_n = sar_ok ? sar_n : 1
    sar_d = sar_ok ? sar_d : 1
    same_raster = c.Width == out_w && c.Height == out_h
    osar_n = Defined(out_sar_num) ? out_sar_num : same_raster ? sar_n : 1
    osar_d = Defined(out_sar_den) ? out_sar_den : same_raster ? sar_d : 1
    Assert(osar_n > 0 && osar_d > 0, "UConvert: out_sar_num and out_sar_den must be greater than 0")
    plan = "in=" + string(c.Width) + "x" + string(c.Height) + " " + scan + " " + string(c.FrameRateNumerator) + "/" +
    \      string(c.FrameRateDenominator) + " " + c.PixelType + " sar " + string(sar_n) + ":" + string(sar_d)

    # ---- work format: lossless detour for formats the stage plugins cannot take, no audio, no colour props
    orig_pt = c.PixelType
    was_rgb = c.IsRGB
    full = uc_PropInt(c, "_ColorRange", was_rgb ? 0 : 1) == 0   # no property: RGB full range, YUV limited
    c = uc_ToWork(c.KillAudio())
    c = c.propDelete("_Matrix").propDelete("_Primaries").propDelete("_Transfer").propDelete("_ColorRange")
    bits = c.BitsPerComponent

    # ---- spatial plan: scaled picture size (sc_*), source window (s_*), borders (pad_*)
    sub_w = orig_pt == "YV411" ? 4 : c.Is420 || c.Is422 ? 2 : 1
    sub_h = c.Is420 ? 2 : 1
    mod_w = sub_w
    mod_h = oi == "" ? sub_h : 2 * sub_h
    Assert(out_w % mod_w == 0 && out_h % mod_h == 0, "UConvert: output size " + string(out_w) + "x" +
    \      string(out_h) + " must be a multiple of " + string(mod_w) + "x" + string(mod_h) + " for " + orig_pt +
    \      (oi == "" ? "" : " interlaced"))
    in_dar = float(c.Width) * sar_n / (float(c.Height) * sar_d)
    out_par = float(osar_n) / osar_d
    out_dar = float(out_w) * out_par / out_h
    wide = in_dar > out_dar     # the input picture is wider than the output frame
    narrow = in_dar < out_dar
    sc_w = aspect_mode == "pillar" && narrow ? Min(out_w, uc_RoundMod(out_h * in_dar / out_par, mod_w)) : out_w
    sc_h = aspect_mode == "pillar" && wide ? Min(out_h, uc_RoundMod(out_w * out_par / in_dar, mod_h)) : out_h
    s_w = aspect_mode == "crop" && wide ? c.Height * out_dar * sar_d / sar_n : float(c.Width)
    s_h = aspect_mode == "crop" && narrow ? c.Width * float(sar_n) / sar_d / out_dar : float(c.Height)
    s_left = (c.Width - s_w) / 2.0
    s_top = (c.Height - s_h) / 2.0
    pad_l = (out_w - sc_w) / 2 / mod_w * mod_w
    pad_t = (out_h - sc_h) / 2 / mod_h * mod_h
    spatial = !same_raster || sc_w != out_w || sc_h != out_h || s_w != c.Width || s_h != c.Height
    pre_scale = sc_w * sc_h < c.Width * c.Height
    scale_plan = " | scale=" + string(sc_w) + "x" + string(sc_h) + " in " + string(out_w) + "x" + string(out_h)

    # ---- temporal plan (decision tables in DESIGN.md; all rates as num/den)
    i_in = scan != "p"
    i_out = oi != ""
    r_n = i_in ? 2 * c.FrameRateNumerator : c.FrameRateNumerator
    r_d = c.FrameRateDenominator
    m_n = i_out ? 2 * out_rate_nom : out_rate_nom
    m_d = out_rate_den
    r = float(r_n) / r_d
    m = float(m_n) / m_d
    half = r < 0.75 * m && !uc_IsInt(m / r)
    p_n = m_n
    p_d = half ? 2 * m_d : m_d
    p = float(p_n) / p_d
    frc = uc_Same(r, p) ? "none" : r > p && uc_IsInt(r / p) ? "drop" : p > r && uc_IsInt(p / r) ? "repeat" : "mc"
    passthrough = !spatial && uc_Same(float(c.FrameRateNumerator) / c.FrameRateDenominator,
    \             float(out_rate_nom) / out_rate_den) && scan == (i_out ? oi : "p")
    wbits = frc_8bit ? 8 : bits == 32 ? 16 : bits   # bit depth for deinterlacing and MC frame-rate conversion

    # ---- execute
    gpu = gpu && !passthrough && (i_in || frc == "mc" || (i_out && uc_interlace_kernel != ""))
    if (gpu) {
        uc_GpuCheck()
        global cuda_cpu_fallback = true   # GPU present: only stages without a GPU path for the format use the CPU
    }
    mc_gpu = gpu && frc == "mc"
    if (!passthrough) {
        c = (i_in || frc == "mc") && wbits != bits ? c.ConvertBits(wbits, dither=-1, fulls=full, fulld=full) : c
        if (i_in) {
            c = uc_Deint(c, scan == "tff", gpu)
            plan = plan + " | deint=qtgmc" + (gpu ? "-gpu" + (uc_qtgmc_nvof ? "-nvof" : "") : "") + " " + scan
        }
        if (spatial && pre_scale) {
            c = uc_Scale(c, sc_w, sc_h, s_left, s_top, s_w, s_h, pad_l, pad_t, out_w, out_h, full, was_rgb)
            plan = plan + scale_plan
        }
        if (frc == "mc") {
            c = mc_gpu && threads > 0 ? c.Prefetch(threads) : c
            c = uc_FRC(c, p_n, p_d, gpu)
            c = mc_gpu && threads > 0 ? c.Prefetch(1) : c
            plan = plan + " | frc=" + (gpu ? "splat " : "mc ") + string(p_n) + "/" + string(p_d)
        }
        if (frc == "drop") {
            c = c.SelectEvery(round(r / p), 0).AssumeFPS(p_n, p_d)
            plan = plan + " | frc=drop 1/" + string(round(r / p))
        }
        c = c.BitsPerComponent != bits ? c.ConvertBits(bits, dither=-1, fulls=full, fulld=full) : c
        if (spatial && !pre_scale) {
            c = uc_Scale(c, sc_w, sc_h, s_left, s_top, s_w, s_h, pad_l, pad_t, out_w, out_h, full, was_rgb)
            plan = plan + scale_plan
        }
        # repeating after scaling: identical frames are scaled once
        if (frc == "repeat") {
            c = uc_Repeat(c, round(p / r)).AssumeFPS(p_n, p_d)
            plan = plan + " | frc=repeat x" + string(round(p / r))
        }
        if (half) {
            c = c.SelectEvery(1, 0, 0).AssumeFPS(m_n, m_d)
            plan = plan + " | pictures x2"
        }
        c = threads > 0 && !mc_gpu ? c.Prefetch(threads) : c
        if (i_out) {
            c = uc_Interlace(c, oi == "tff", gpu)
            plan = plan + " | interlace=" + oi + (gpu && uc_interlace_kernel != "" ? " gpu" : "")
        }
    } else {
        plan = plan + " | passthrough"
    }

    # ---- output: original pixel format, colour properties, scan / SAR tags, audio
    c = uc_FromWork(c, orig_pt, was_rgb)
    c = uc_CopyProp(c, src, "_Matrix")
    c = uc_CopyProp(c, src, "_Primaries")
    c = uc_CopyProp(c, src, "_Transfer")
    c = uc_CopyProp(c, src, "_ColorRange")
    c = uc_CopyProp(c, src, "_ChromaLocation")
    c = i_out ? uc_Order(c, oi == "tff") : c.AssumeFrameBased()
    c = c.propSet("_FieldBased", oi == "tff" ? 2 : oi == "bff" ? 1 : 0)
    c = c.propSet("_SARNum", osar_n).propSet("_SARDen", osar_d)
    c = propGetType(src, "_uc_source") == 3 ? c.propSet("_uc_source", propGetString(src, "_uc_source")) : c
    plan = plan + " | out=" + string(out_w) + "x" + string(out_h) + " " + (i_out ? oi : "p") + " " +
    \      string(out_rate_nom) + "/" + string(out_rate_den) + " " + c.PixelType
    c = c.propSet("_uc_plan", plan)
    c = src.HasAudio ? AudioDub(c, src).Trim(0, 0) : c
    return threads > 0 ? c.Prefetch(mc_gpu ? 1 : 3) : c
}


# ---------------------------------------------------------------------------------------------------------------
# 1. Source stage: MXF -> MxfSource, MOV/MP4 -> L-SMASH (no index), everything else -> LWLibav; with fallbacks.
#    Records what was used in the frame property _uc_source.

function uc_Source(string file, string "source", string "cachedir")
{
    source = LCase(Default(source, "auto"))
    Assert(source == "auto" || source == "mxf" || source == "lsmash" || source == "lwlibav",
    \      "UConvert: source must be 'auto', 'mxf', 'lsmash' or 'lwlibav'")
    ext = LCase(uc_Ext(file))
    kind = source != "auto" ? source : ext == "mxf" ? "mxf" : ext == "mov" || ext == "mp4" || ext == "m4v" ? "lsmash" : "lwlibav"
    cf = Defined(cachedir) ? uc_CacheFile(file, cachedir) : ""
    info = ""

    if (kind == "mxf") {
        ok = true
        try {
            c = MxfSource(file, use_audio="always")
            info = "video=mxfsource audio=" + (c.HasAudio ? "mxfsource" : "none")
        } catch (e1) {
            ok = false
        }
        if (!ok) {
            ok = true
            try {
                c = MxfSource(file, use_audio="never")
            } catch (e2) {
                ok = false
            }
            if (ok) {
                a = uc_LWAudio(file, cf)
                c = IsClip(a) ? AudioDub(c, a) : c
                info = "video=mxfsource audio=" + (IsClip(a) ? "lwlibav" : "none") + " (MxfSource audio: " + e1 + ")"
            } else {
                kind = "lwlibav"
                info = " (MxfSource: " + e2 + ")"
            }
        }
    }
    if (kind == "lsmash") {
        ok = true
        try {
            c = LSMASHVideoSource(file)
        } catch (e3) {
            ok = false
        }
        if (ok) {
            a = uc_LSAudio(file)
            c = IsClip(a) ? AudioDub(c, a) : c
            info = "video=lsmash audio=" + (IsClip(a) ? "lsmash" : "none")
        } else {
            kind = "lwlibav"
            info = " (L-SMASH: " + e3 + ")"
        }
    }
    if (kind == "lwlibav") {
        c = cf != "" ? LWLibavVideoSource(file, cachefile=cf) : LWLibavVideoSource(file, cache=false)
        a = uc_LWAudio(file, cf)
        c = IsClip(a) ? AudioDub(c, a) : c
        info = "video=lwlibav audio=" + (IsClip(a) ? "lwlibav" : "none") + info
    }
    return c.propSet("_uc_source", uc_OneLine(info))
}

# All audio streams of a file via LWLibav, merged in stream order; "none" if there is no audio.
# Every probe of a stream index that is not audio makes LWLibav re-index the file (without cachedir every call
# indexes it), so the search assumes contiguous audio streams: from stream 1 up to the first gap, stream 0 only if
# 1 and 2 are not audio.
function uc_LWAudio(string file, string cf)
{
    acc = "none"
    miss = 0
    i = 1
    done = false
    while (i < 32 && !done) {
        a = uc_LWAudioStream(file, cf, i)
        acc = IsClip(a) ? (IsClip(acc) ? uc_AudioJoin(acc, a) : a) : acc
        miss = IsClip(a) ? miss : miss + 1
        done = !IsClip(a) && (IsClip(acc) || miss >= 2)
        i = i + 1
    }
    return IsClip(acc) ? acc : uc_LWAudioStream(file, cf, 0)
}

function uc_LWAudioStream(string file, string cf, int i)
{
    try {
        a = cf != "" ? LWLibavAudioSource(file, stream_index=i, cachefile=cf, av_sync=true)
        \            : LWLibavAudioSource(file, stream_index=i, cache=false, av_sync=true)
    } catch (e) {
        a = "none"
    }
    return a
}

# All audio tracks of a QuickTime/MP4 file via L-SMASH (track numbers start at 1).
function uc_LSAudio(string file)
{
    acc = "none"
    miss = 0
    i = 1
    while (i < 32 && miss < 4) {
        ok = true
        try {
            a = LSMASHAudioSource(file, track=i)
        } catch (e) {
            ok = false
        }
        miss = ok ? 0 : (i > 1 ? miss + 1 : 0)
        acc = ok ? (IsClip(acc) ? uc_AudioJoin(acc, a) : a) : acc
        i = i + 1
    }
    return acc
}

# Append the channels of b to a. Different sample rates: both to 48 kHz; different sample types: common
# lossless type (wider integer, or float if one is float).
function uc_AudioJoin(clip a, clip b)
{
    to48 = a.AudioRate != b.AudioRate
    a = to48 && a.AudioRate != 48000 ? a.ResampleAudio(48000) : a
    b = to48 && b.AudioRate != 48000 ? b.ResampleAudio(48000) : b
    flt = a.IsAudioFloat || b.IsAudioFloat
    same = a.IsAudioFloat == b.IsAudioFloat && a.AudioBits == b.AudioBits
    wide = Max(a.AudioBits, b.AudioBits)
    a = same ? a : flt ? a.ConvertAudioToFloat() : wide == 32 ? a.ConvertAudioTo32bit() : a.ConvertAudioTo24bit()
    b = same ? b : flt ? b.ConvertAudioToFloat() : wide == 32 ? b.ConvertAudioTo32bit() : b.ConvertAudioTo24bit()
    return MergeChannels(a, b)
}

# LWLibav index file in cachedir, named after the full source path (never next to the source).
function uc_CacheFile(string file, string cachedir)
{
    name = ReplaceStr(ReplaceStr(ReplaceStr(ReplaceStr(file, ":", "_"), "\", "_"), "/", "_"), " ", "_")
    dir = RightStr(cachedir, 1) == "\" || RightStr(cachedir, 1) == "/" ? cachedir : cachedir + "\"
    return dir + name + ".lwi"
}


# ---------------------------------------------------------------------------------------------------------------
# 2. Probe helpers

# Scan type from _FieldBased (0 progressive, 1 BFF, 2 TFF) at up to five frames spread over the clip; the first
# interlaced one wins (no further frames are read). No property at all = progressive.
function uc_ScanOf(clip c)
{
    res = "p"
    k = 0
    while (res == "p" && k <= 4) {
        fb = uc_PropInt(c, "_FieldBased", 0, offset=(c.FrameCount - 1) * k / 4)
        res = fb == 2 ? "tff" : fb == 1 ? "bff" : "p"
        k = k + 1
    }
    return res
}


# ---------------------------------------------------------------------------------------------------------------
# 4. Stage functions

# Stops with the reason when the GPU stages cannot run here (no NVIDIA GPU, driver too old, ...).
function uc_GpuCheck()
{
    msg = ""
    try {
        t = BlankClip(length=2, width=64, height=64, pixel_type="YUV422P10").CudaConvolution()
        t = propGetType(t, "_FieldBased")   # renders frame 0: the GPU engine runs once
    } catch (e) {
        msg = e
    }
    Assert(msg == "", "UConvert: gpu=true needs a usable NVIDIA GPU - " + uc_OneLine(msg) + " - use gpu=false for the CPU chain")
}

# Bob deinterlace to field rate. Alpha (unsupported by QTGMC) is deinterlaced as a separate luma clip.
# gpu: QTGMC_gpu.avsi's switches (motion search, search clip and output side on the GPU; NNEDI3 stays on the CPU,
# CudaNNEDI3 refuses CPUs without its AVX2 + FMA path).
function uc_Deint(clip c, bool tff, bool "gpu")
{
    gpu = Default(gpu, false)
    global QTGMC_gpu = gpu
    global QTGMC_gpu_search = gpu
    global QTGMC_gpu_tokens = gpu
    global QTGMC_gpu_out = gpu
    global QTGMC_gpu_edi = false
    global QTGMC_gpu_nvof = gpu && uc_qtgmc_nvof
    m = c.HasAlpha ? c.RemoveAlphaPlane() : c
    d = uc_Order(m, tff).QTGMC(Preset=uc_qtgmc_preset)
    d = c.HasAlpha ? d.AddAlphaPlane(uc_Order(c.ExtractA(), tff).QTGMC(Preset=uc_qtgmc_preset)) : d
    return d.AssumeFrameBased()
}

# Motion-compensated frame-rate conversion (benchmarked settings; HD: pel 4 / 16x16, UHD: pel 2 / 32x32).
# Clean cut: blend=false + MVTools scene-change detection. Alpha is compensated with the picture's vectors.
function uc_FRC(clip c, int num, int den, bool "gpu")
{
    return Default(gpu, false) ? uc_FRCGpu(c, num, den) : uc_FRCCpu(c, num, den)
}

# GPU: CudaSplatFps with the benchmarked light settings (16 candidates). Greyscale runs as 4:4:4 with neutral chroma;
# alpha is converted with the picture's motion candidates (SplatAnalyse of the picture). SplatFps refuses 4:2:0 sited
# on the luma rows (_ChromaLocation 2..5): such clips are treated as left-sited here (the output gets the source's
# property back at the end).
function uc_FRCGpu(clip c, int num, int den)
{
    bp = [1, 2, 3]
    nb = [0, 0, 1]
    m = c.HasAlpha ? c.RemoveAlphaPlane() : c
    grey = m.IsY
    m = grey ? m.ConvertToYUV444() : m
    m = m.Is420 && uc_PropInt(m, "_ChromaLocation", 0) >= 2 ? m.propSet("_ChromaLocation", 0) : m
    out = m.CudaSplatFps(num, den, block_peaks=bp, nearest_block=nb)
    if (c.HasAlpha) {
        ac = c.ExtractA().ConvertToYUV444()
        vec = m.SplatAnalyse(block_peaks=bp, nearest_block=nb)
        av = ac.CudaSplatFps(num, den, vectors=vec, block_peaks=bp, nearest_block=nb).ExtractY()
    }
    out = grey ? out.ExtractY() : out
    return c.HasAlpha ? out.AddAlphaPlane(av) : out
}

function uc_FRCCpu(clip c, int num, int den)
{
    m = c.HasAlpha ? c.RemoveAlphaPlane() : c
    big = m.Width * m.Height > 2500000
    pel = big ? 2 : 4
    bs = big ? 32 : 16
    sup = MSuper(m, pel=pel, hpad=16, vpad=16, rfilter=4)
    bv = MAnalyse(sup, isb=true, blksize=bs, overlap=bs / 2, search=3, searchparam=2, truemotion=true, chroma=false)
    fv = MAnalyse(sup, isb=false, blksize=bs, overlap=bs / 2, search=3, searchparam=2, truemotion=true, chroma=false)
    out = MBlockFps(m, sup, bv, fv, num=num, den=den, mode=3, blend=false)
    if (c.HasAlpha) {
        ac = CombinePlanes(c.ExtractA(), m, m, planes="YUV", source_planes="YUV", pixel_type=m.PixelType)
        asup = MSuper(ac, pel=pel, hpad=16, vpad=16, rfilter=4)
        out = out.AddAlphaPlane(MBlockFps(ac, asup, bv, fv, num=num, den=den, mode=3, blend=false).ExtractY())
    }
    return out
}

# Resize (Spline36) the source window to w x h, then pad to out_w x out_h with black. Resize only: no matrix,
# range or subsampling change.
function uc_Scale(clip c, int w, int h, float s_left, float s_top, float s_w, float s_h, int pad_l, int pad_t,
\                 int out_w, int out_h, bool full, bool rgb)
{
    c = c.z_ConvertFormat(width=w, height=h, resample_filter="spline36",
    \                     src_left=s_left, src_top=s_top, src_width=s_w, src_height=s_h)
    black = rgb ? (full ? $000000 : $101010) : full ? $008080 : $108080
    return w == out_w && h == out_h ? c
    \      : c.AddBorders(pad_l, pad_t, out_w - w - pad_l, out_h - h - pad_t, color_yuv=black)
}

# Progressive frames at field rate -> interlaced: a 9-tap vertical low-pass filter on the
# progressive picture, then each field's lines (first field from the earlier frame). 4:2:0 is filtered at 4:2:2
# and subsampled per field, so the chroma of each field comes from that field's lines.
function uc_Interlace(clip c, bool tff, bool "gpu")
{
    pt = c.PixelType
    x = c.Is420 ? c.z_ConvertFormat(pixel_type=uc_PT422(pt), resample_filter="spline36") : c
    f = Default(gpu, false) && uc_interlace_kernel != "" ? uc_Order(uc_InterlaceGpu(x, tff), tff).SeparateFields()
    \                                                    : uc_FieldsCpu(x, tff)
    # the fields are woven last: with an odd number of input frames the missing last field repeats the field before
    f = c.Is420 ? f.AssumeFrameBased().z_ConvertFormat(pixel_type=pt, resample_filter="spline36").AssumeFieldBased() : f
    return uc_Order(uc_Order(f, tff).Weave(), tff)
}

# CPU: filter (mt_convolution; none with uc_interlace_kernel = ""), then the two fields of each output frame.
function uc_FieldsCpu(clip c, bool tff)
{
    x = uc_interlace_kernel == "" ? c : c.mt_convolution(horizontal="1", vertical=uc_interlace_kernel, total=4096, Y=3, U=3, V=3, A=3)
    return uc_Order(x, tff).SeparateFields().SelectEvery(4, 0, 3)
}

# GPU: CudaConvolution with uc_interlace.json filters and takes the fields in one step (same taps; 10-bit
# arithmetic, so 8-bit results can differ from mt_convolution by one code). Formats it does not take are prepared
# losslessly where possible: greyscale as 4:4:4, 14 bit / float via 16 bit, alpha as a separate greyscale clip.
function uc_InterlaceGpu(clip c, bool tff)
{
    Assert(uc_interlace_kernel == uc_interlace_preset_kernel, "UConvert: uc_interlace_kernel differs from the taps in " +
    \      uc_interlace_preset + " (uc_interlace_preset_kernel); change both, or use gpu=false")
    m = c.HasAlpha ? c.RemoveAlphaPlane() : c
    bits = m.BitsPerComponent
    x = m.IsY ? m.ConvertToYUV444() : m
    x = bits == 14 || bits == 32 ? x.ConvertBits(16) : x
    w = x.CudaConvolution(preset=uc_interlace_preset, field_order=tff ? "tff" : "bff", convert=false)
    w = m.IsY ? w.ExtractY() : w
    w = w.BitsPerComponent != bits ? w.ConvertBits(bits) : w
    return c.HasAlpha ? w.AddAlphaPlane(uc_InterlaceGpu(c.ExtractA(), tff)) : w
}

# Each frame n times.
function uc_Repeat(clip c, int n)
{
    sel = "0"
    for (i = 2, n) {
        sel = sel + ",0"
    }
    return Eval("c.SelectEvery(1," + sel + ")")
}


# ---------------------------------------------------------------------------------------------------------------
# Lossless format detour. Work format: planar; RGB relabelled as YUV 4:4:4 (G in Y, B in U, R in V) so QTGMC
# and MVTools take it. Values are never converted.

function uc_ToWork(clip c)
{
    c = c.IsYUY2 ? c.ConvertToYV16() : c
    # 4:1:1 (MVTools crashes on it) -> 4:4:4 by sample duplication; uc_FromWork takes every 4th sample back
    c = c.IsYV411 ? c.ConvertToYV24(chromaresample="point") : c
    c = c.IsRGB && !c.IsPlanar ? (c.HasAlpha ? c.ConvertToPlanarRGBA() : c.ConvertToPlanarRGB()) : c
    return c.IsRGB ? CombinePlanes(c, planes=c.HasAlpha ? "YUVA" : "YUV", source_planes=c.HasAlpha ? "GBRA" : "GBR",
    \                              pixel_type=uc_PT(c.HasAlpha ? "YUVA444" : "YUV444", c.BitsPerComponent)) : c
}

function uc_FromWork(clip c, string orig_pt, bool was_rgb)
{
    c = was_rgb ? CombinePlanes(c, planes=c.HasAlpha ? "GBRA" : "GBR", source_planes=c.HasAlpha ? "YUVA" : "YUV",
    \                           pixel_type=uc_PT(c.HasAlpha ? "RGBAP" : "RGBP", c.BitsPerComponent)) : c
    return orig_pt == "YV411" ? CombinePlanes(c.ExtractY(), c.ExtractU().PointResize(c.Width / 4, c.Height),
    \                                         c.ExtractV().PointResize(c.Width / 4, c.Height), planes="YUV",
    \                                         pixel_type="YV411")
    \    : orig_pt == "YUY2" ? c.ConvertToYUY2()
    \    : orig_pt == "RGB24" ? c.ConvertToRGB24()
    \    : orig_pt == "RGB32" ? c.ConvertToRGB32()
    \    : orig_pt == "RGB48" ? c.ConvertToRGB48()
    \    : orig_pt == "RGB64" ? c.ConvertToRGB64()
    \    : c
}

# Pixel type name for a family ("YUV444", "YUVA444", "RGBP", "RGBAP") and bit depth.
function uc_PT(string family, int bits)
{
    return bits == 8 ? (family == "YUV444" ? "YV24" : family)
    \    : bits == 32 ? family + (family == "RGBP" || family == "RGBAP" ? "S" : "PS")
    \    : family + (family == "RGBP" || family == "RGBAP" ? "" : "P") + string(bits)
}

# 4:2:0 pixel type -> the matching 4:2:2 one.
function uc_PT422(string pt)
{
    return pt == "YV12" ? "YV16" : ReplaceStr(pt, "420", "422")
}


# ---------------------------------------------------------------------------------------------------------------
# Small helpers

# Integer frame property of frame `offset` (default 0), or def when the frame does not have it.
function uc_PropInt(clip c, string key, int def, int "offset")
{
    offset = Default(offset, 0)
    return propGetType(c, key, offset=offset) == 1 ? propGetInt(c, key, offset=offset) : def
}

function uc_CopyProp(clip c, clip src, string key)
{
    return propGetType(src, key) == 1 ? c.propSet(key, propGetInt(src, key)) : c.propDelete(key)
}

function uc_Order(clip c, bool tff) { return tff ? c.AssumeTFF() : c.AssumeBFF() }

function uc_Same(float a, float b) { return Abs(a - b) <= 0.000001 * Max(Abs(a), Abs(b)) }

function uc_IsInt(float x) { return x >= 1.0 && Abs(x - Round(x)) <= 0.000001 * x }

function uc_RoundMod(float x, int m) { return Max(m, Round(x / m) * m) }

function uc_Ext(string file)
{
    dot = FindStr(RevStr(file), ".")
    return dot > 0 ? RightStr(file, dot - 1) : ""
}

function uc_OneLine(string s)
{
    return ReplaceStr(ReplaceStr(s, Chr(13), " "), Chr(10), " ")
}
