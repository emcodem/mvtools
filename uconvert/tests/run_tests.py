"""UConvert test cases: plan, output format/properties and a decode check for each case.

For each case an AviSynth script is written to work/uc_tests/<name>.avs. A first run reports the chosen chain
(_uc_plan), output size, scan, rate, frame count, pixel type, colour/SAR properties and audio; a second run
decodes the first --frames frames with ffmpeg to catch run-time errors.

usage: python -I tests/run_tests.py [name-substring ...] [--frames 30] [--list] [--cpu]
--cpu: run the cases on the CPU chain (uc_gpu_default = false); default is the GPU chain.
"""
import sys, os, argparse, subprocess, re

HERE = os.path.dirname(os.path.abspath(__file__))
UC = os.path.dirname(HERE).replace("\\", "/")
sys.path.insert(0, HERE)
from ucdata import data_root
ROOT = data_root()
WORK = ROOT + "/work/uc_tests"
P = ROOT + "/probes/"

HEAD = f'''Import("{UC}/load_plugins.avs")
Import("{UC}/uconvert.avsi")
'''
if "--cpu" in sys.argv:   # every case without gpu= on the CPU chain
    sys.argv.remove("--cpu")
    HEAD += "global uc_gpu_default = false\n"

INFO = '''
c = last
fb = propGetType(c, "_FieldBased") == 1 ? string(propGetInt(c, "_FieldBased")) : "-"
function pv(clip c, string k) { return propGetType(c, k) == 1 ? string(propGetInt(c, k)) : "-" }
Assert(false, "INFO " + string(c.Width) + "x" + string(c.Height) + " fb=" + fb + " parity=" + (c.GetParity(0) ? "T" : "B") \\
  + " " + string(c.FrameRateNumerator) + "/" + string(c.FrameRateDenominator) + " n=" + string(c.FrameCount) \\
  + " " + c.PixelType + " sar=" + pv(c, "_SARNum") + ":" + pv(c, "_SARDen") + " mat=" + pv(c, "_Matrix") \\
  + " pri=" + pv(c, "_Primaries") + " trc=" + pv(c, "_Transfer") + " rng=" + pv(c, "_ColorRange") \\
  + " audio=" + (c.HasAudio ? string(c.AudioChannels) + "ch/" + string(c.AudioRate) + "/" + string(c.AudioBits) \\
  + "b len=" + string(c.AudioDuration) + "s vs " + string(c.FrameCount * c.FrameRateDenominator / float(c.FrameRateNumerator)) + "s" : "none") \\
  + chr(10) + "  source: " + (propGetType(c, "_uc_source") == 3 ? propGetString(c, "_uc_source") : "-") \\
  + chr(10) + "  plan:   " + (propGetType(c, "_uc_plan") == 3 ? propGetString(c, "_uc_plan") : "-"))
'''


def bc(fmt, w=1920, h=1080, fps="25", n=100, props=""):
    """Synthetic moving clip: a probe file converted to the given format (colour values irrelevant here)."""
    num, den = (fps.split("/") + ["1"])[:2]
    return (f'LWLibavVideoSource("{P}frc_probe_5994p.mov", cachefile="{ROOT}/work/frc_probe_5994p.mov.lwi")'
            f'.Trim(0, {n - 1}).AssumeFPS({num}, {den}).ConvertBits(8).ConvertToYV24()'
            f'.z_ConvertFormat(width={w}, height={h}, resample_filter="bilinear").{fmt_conv(fmt)}{props}')


def fmt_conv(fmt):
    if fmt.startswith("RGB"):
        bits = re.sub(r"\D", "", fmt) or "8"
        base = "ConvertToPlanarRGB()" if "A" not in fmt[3:] else "ConvertToPlanarRGBA()"
        if fmt in ("RGB24", "RGB32", "RGB48", "RGB64"):
            return {"RGB24": "ConvertToRGB24()", "RGB32": "ConvertToRGB32()",
                    "RGB48": "ConvertToRGB48()", "RGB64": "ConvertToRGB64()"}[fmt]
        return base + (f".ConvertBits({bits})" if bits != "8" else "")
    if fmt == "YUY2":
        return "ConvertToYUY2()"
    return f'z_ConvertFormat(pixel_type="{fmt}")'


SD = ROOT + "/sd/"
CASES = {
    # --- real files, source stage
    "mov_5994i_to_1080i25": f'UConvert("{P}frc_probe_5994i.mov", 1080, 1920, "tff", 25, 1, cachedir="{ROOT}/work", threads=8)',
    "mov_5994p_to_1080p50": f'UConvert("{P}frc_probe_5994p.mov", 1080, 1920, out_rate_nom=50, out_rate_den=1, threads=8)',
    "mov_2997p_to_1080i25": f'UConvert("{P}frc_probe_2997p.mov", 1080, 1920, "tff", 25, 1, threads=8)',
    "mov_24p_to_1080p50": f'UConvert("{P}frc_probe_24p.mov", 1080, 1920, "", 50, 1, threads=8)',
    "mov_23976p_to_1080p25": f'UConvert("{P}frc_probe_23976p.mov", 1080, 1920, "", 25, 1, threads=8)',
    "mov_1080i25_to_1080p50": f'UConvert("{ROOT}/deint/test_1080i25.mov", 1080, 1920, "", 50, 1, threads=8)',
    "mov_1080i25_to_1080p25": f'UConvert("{ROOT}/deint/test_1080i25.mov", 1080, 1920, "", 25, 1, threads=8)',
    "mov_1080i25_passthrough": f'UConvert("{ROOT}/deint/test_1080i25.mov", 1080, 1920, "tff", 25, 1, threads=8)',
    "mov_1080i25_to_bff": f'UConvert("{ROOT}/deint/test_1080i25.mov", 1080, 1920, "bff", 25, 1, threads=8)',
    "mpg_sd576i_to_1080i25": f'UConvert("{SD}scaling_test_SD_PAL_25_576i.mpg", 1080, 1920, "tff", 25, 1, cachedir="{ROOT}/work", sar_num=16, sar_den=15, threads=8)',
    "mpg_sd576i_to_576p25_same_raster": f'UConvert("{SD}scaling_test_SD_PAL_25_576i.mpg", 576, 720, "", 25, 1, cachedir="{ROOT}/work", threads=8)',
    "mxf_xdcam_50i_to_1080p50": f'UConvert("{SD}xdcam_50i.mxf", 1080, 1920, "", 50, 1, cachedir="{ROOT}/work", threads=8)',
    "mxf_forced_lwlibav": f'UConvert("{SD}xdcam_50i.mxf", 1080, 1920, "tff", 25, 1, source="lwlibav", cachedir="{ROOT}/work", threads=8)',
    # not an MXF despite the name: MxfSource fails to open -> LWLibav for video and audio
    "mxf_fallback_not_mxf": f'UConvert("{WORK}/fake_mpeg_ps.mxf", 576, 720, "tff", 25, 1, cachedir="{ROOT}/work", threads=8)',
    # 16 + 24 bit audio tracks: MxfSource(use_audio="always") refuses -> MxfSource video, LWLibav audio (merged to 24 bit)
    "mxf_fallback_mixed_audio": f'UConvert("{WORK}/mixed_audio.mxf", 576, 720, "tff", 25, 1, cachedir="{ROOT}/work", threads=8)',
    # --- aspect handling
    "aspect_4x3_pillar": bc("YV16", 1440, 1080, "25") + '.UConvertClip(1080, 1920, out_rate_nom=25, out_rate_den=1, sar_num=1, sar_den=1)',
    "aspect_4x3_crop": bc("YV16", 1440, 1080, "25") + '.UConvertClip(1080, 1920, out_rate_nom=25, out_rate_den=1, aspect_mode="crop")',
    "aspect_4x3_stretch": bc("YV16", 1440, 1080, "25") + '.UConvertClip(1080, 1920, out_rate_nom=25, out_rate_den=1, aspect_mode="stretch")',
    "aspect_hdv_anamorphic": bc("YV16", 1440, 1080, "25") + '.UConvertClip(1080, 1920, out_rate_nom=25, out_rate_den=1, sar_num=4, sar_den=3)',
    "aspect_scope_letterbox": bc("YV12", 2048, 858, "24") + '.UConvertClip(1080, 1920, "tff", 25, 1)',
    # --- awkward formats and rates
    "fmt_yuy2_59.94i_to_50p": bc("YUY2", 1920, 1080, "30000/1001") + '.AssumeTFF().UConvertClip(1080, 1920, out_rate_nom=50, out_rate_den=1, scan_in="tff")',
    "fmt_rgbp10_24p_to_i25": bc("RGBP10", 1920, 1080, "24") + '.UConvertClip(1080, 1920, "tff", 25, 1)',
    "fmt_rgb32_alpha_30p_to_50p": bc("RGB32", 1280, 720, "30") + '.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1)',
    "fmt_yuva422p10_i_to_p": bc("YUVA422P10", 1920, 1080, "25") + '.UConvertClip(1080, 1920, out_rate_nom=50, out_rate_den=1, scan_in="tff")',
    "fmt_yv411_ntsc_dv": bc("YV411", 720, 480, "30000/1001") + '.UConvertClip(576, 720, "bff", 25, 1, scan_in="bff", sar_num=10, sar_den=11, out_sar_num=16, out_sar_den=15)',
    "fmt_y8_odd_size": bc("Y8", 719, 433, "15") + '.UConvertClip(1080, 1920, out_rate_nom=50, out_rate_den=1)',
    "fmt_float_frc_keepbits": bc("YUV444PS", 1280, 720, "24") + '.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1, frc_8bit=false)',
    "fmt_420p10_keepbits_i": bc("YUV420P10", 1920, 1080, "50") + '.UConvertClip(1080, 1920, "tff", 25, 1, frc_8bit=false)',
    "fmt_hlg_props_passthrough": bc("YUV422P10", 3840, 2160, "50", 20, props='.propSet("_Matrix", 9).propSet("_Primaries", 9).propSet("_Transfer", 18).propSet("_ColorRange", 1)') + '.UConvertClip(1080, 1920, out_rate_nom=50, out_rate_den=1)',
    "rate_49p_to_100p": bc("YV12", 1280, 720, "49") + '.UConvertClip(720, 1280, out_rate_nom=100, out_rate_den=1)',
    "rate_40p_to_50p": bc("YV12", 1280, 720, "40") + '.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1)',
    "rate_50p_to_23976p": bc("YV12", 1280, 720, "50") + '.UConvertClip(720, 1280, out_rate_nom=24000, out_rate_den=1001)',
    "rate_12p_to_50p": bc("YV12", 1280, 720, "12") + '.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1)',
    "rate_25p_to_75p_repeat": bc("YV12", 1280, 720, "25") + '.UConvertClip(720, 1280, out_rate_nom=75, out_rate_den=1)',
    "rate_uhd_5994p_to_50p": bc("YV16", 3840, 2160, "60000/1001", 40) + '.UConvertClip(2160, 3840, out_rate_nom=50, out_rate_den=1)',
}


# Bit-exact checks: (UConvert expression, reference expression built from the same input `s`).
SRC_BC = lambda fmt, w=1280, h=720, fps="25", n=60: "s = " + bc(fmt, w, h, fps, n) + "\n"
EXACT = {
    "exact_passthrough_1080i25": (f's = LSMASHVideoSource("{ROOT}/deint/test_1080i25.mov").Trim(0, 49)\n',
                                  's.UConvertClip(1080, 1920, "tff", 25, 1)', 's'),
    "exact_detour_rgb32": (SRC_BC("RGB32"), 's.UConvertClip(720, 1280, out_rate_nom=25, out_rate_den=1)', 's'),
    "exact_detour_rgb64": (SRC_BC("RGB64"), 's.UConvertClip(720, 1280, out_rate_nom=25, out_rate_den=1)', 's'),
    "exact_detour_rgbap10": (SRC_BC("RGBAP10"), 's.UConvertClip(720, 1280, out_rate_nom=25, out_rate_den=1)', 's'),
    "exact_detour_yuy2": (SRC_BC("YUY2"), 's.UConvertClip(720, 1280, out_rate_nom=25, out_rate_den=1)', 's'),
    "exact_detour_yv411": (SRC_BC("YV411", 720, 480), 's.UConvertClip(480, 720, out_rate_nom=25, out_rate_den=1)', 's'),
    "exact_drop_50p_25p": (SRC_BC("YUV422P10", fps="50"), 's.UConvertClip(720, 1280, out_rate_nom=25, out_rate_den=1)',
                           's.SelectEven()'),
    "exact_repeat_25p_75p": (SRC_BC("YV12"), 's.UConvertClip(720, 1280, out_rate_nom=75, out_rate_den=1)',
                             's.SelectEvery(1, 0, 0, 0)'),
    "exact_pairs_24p_50p": (SRC_BC("YV16", fps="24"), 's.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1)',
                            's.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1).SelectEven().SelectEvery(1, 0, 0)'),
    "exact_psf_25p_25i": (SRC_BC("YV16"), 's.UConvertClip(720, 1280, "tff", 25, 1, gpu=false)',
                          's.mt_convolution(horizontal="1", vertical=uc_interlace_kernel, total=4096, Y=3, U=3, V=3)'),
    # GPU interlace (CudaConvolution + uc_interlace.json) = mt_convolution from 10 bit up (8 bit: +-1 on ~0.05 %,
    # 10-bit arithmetic)
    "exact_psf_gpu_10bit": (SRC_BC("YUV422P10"), 's.UConvertClip(720, 1280, "tff", 25, 1, gpu=true)',
                            's.mt_convolution(horizontal="1", vertical=uc_interlace_kernel, total=4096, Y=3, U=3, V=3)'),
    "exact_psf_gpu_16bit_444": (SRC_BC("YUV444P16"), 's.UConvertClip(720, 1280, "tff", 25, 1, gpu=true)',
                                's.mt_convolution(horizontal="1", vertical=uc_interlace_kernel, total=4096, Y=3, U=3, V=3)'),
    # GPU QTGMC with the exact search (uc_qtgmc_nvof = false) = CPU QTGMC
    "exact_deint_gpu_eq_cpu": (SRC_BC("YUV422P10", 1920, 1080, n=30) + "s = s.AssumeTFF()\n",
                               's.UConvertClip(1080, 1920, out_rate_nom=50, out_rate_den=1, scan_in="tff", frc_8bit=false, gpu=true)',
                               's.UConvertClip(1080, 1920, out_rate_nom=50, out_rate_den=1, scan_in="tff", frc_8bit=false, gpu=false)'),
    # first output picture (time 0) is the first input picture: colours / channel order survive the MC path
    "exact_rgb32_mc_frame0": (SRC_BC("RGB32", fps="30"), 's.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1).Trim(0, -1)',
                              's.Trim(0, -1)'),
    "exact_rgbp10_keepbits_mc_frame0": (SRC_BC("RGBP10", fps="30"),
                                        's.UConvertClip(720, 1280, out_rate_nom=50, out_rate_den=1, frc_8bit=false).Trim(0, -1)',
                                        's.Trim(0, -1)'),
}


def raw_md5(avs, frames):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", avs, "-frames:v", str(frames), "-f", "rawvideo",
                        "-fps_mode", "passthrough", "-"], capture_output=True)
    import hashlib
    return hashlib.md5(r.stdout).hexdigest(), len(r.stdout), r.stderr.decode(errors="replace")


def exact(name, frames):
    src, out, ref = EXACT[name]
    os.makedirs(WORK, exist_ok=True)
    files = []
    for tag, e in (("out", out), ("ref", ref)):
        f = f"{WORK}/{name}_{tag}.avs"
        open(f, "w").write(HEAD + src + e + "\n")
        files.append(f)
    (h1, n1, e1), (h2, n2, e2) = raw_md5(files[0], frames), raw_md5(files[1], frames)
    ok = h1 == h2 and n1 > 0
    print(f"{name}: {'identical' if ok else 'DIFFERENT'} ({n1} / {n2} bytes)" + ("" if ok else "\n  " + (e1 + e2).strip()[:600]))
    return ok


def run(name, expr, frames):
    os.makedirs(WORK, exist_ok=True)
    avs = f"{WORK}/{name}.avs"
    # ffmpeg cannot read float clips: the decode check reads them as 16 bit
    open(avs, "w").write(HEAD + expr + "\nBitsPerComponent() == 32 ? ConvertBits(16) : last\n")
    info_avs = f"{WORK}/{name}_info.avs"
    open(info_avs, "w").write(HEAD + expr + "\n" + INFO)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", info_avs, "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    m = re.search(r"INFO (.*?)\n\(", r.stderr, re.S)
    if not m:
        print(f"FAIL {name}\n  " + r.stderr.strip().replace("\n", "\n  ")[:1500])
        return False
    print(f"{name}: {m.group(1).strip()}")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", avs, "-frames:v", str(frames), "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    err = "\n".join(l for l in r.stderr.replace("\r", "\n").splitlines()
                    if "rror" in l or "xception" in l or "failed" in l.lower())
    if r.returncode or err.strip():
        print("  DECODE: " + err.strip()[:800])
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    sel = [n for n in list(CASES) + list(EXACT) if not a.names or any(s in n for s in a.names)]
    if a.list:
        print("\n".join(sel))
        return
    bad = [n for n in sel if not (exact(n, a.frames) if n in EXACT else run(n, CASES[n], a.frames))]
    print(f"\n{len(sel) - len(bad)}/{len(sel)} ok" + (": failed " + ", ".join(bad) if bad else ""))


if __name__ == "__main__":
    main()
