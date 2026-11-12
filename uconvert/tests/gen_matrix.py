"""Generate the UConvert input matrix: ~120 short files in many containers, codecs, pixel formats, sizes, rates, scan
types and audio layouts (0..16 channels, one or more streams), plus manifest.json describing each file and the
target UConvert has to convert it to. run_matrix.py checks the results.

Every audio channel carries its own sine tone (channel j of the file, counted over all streams in stream order:
CH_FREQ[j]), so a lost, swapped or mixed channel is detectable after conversion. Lossy codecs get layouts without LFE
(their encoders low-pass the LFE channel).

usage: python -I tests/gen_matrix.py [--out <dir>] [--only name-substring] [--force]
"""
import os, sys, json, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ucdata import data_root
ROOT = data_root()
OUT = ROOT + "/work/uc_matrix"
DUR = 2.0
CH_FREQ = [220 + 160 * j for j in range(16)]          # 220 .. 2620 Hz, 160 Hz apart
LAYOUT = {1: "mono", 2: "stereo", 3: "3.0", 4: "quad", 5: "5.0", 6: "6.0", 7: "7.0", 8: "octagonal",
          16: "hexadecagonal"}

# ---- video specs: name, container, codec args, pix_fmt, w, h, fps (num, den), scan ('p' | 'tff' | 'bff'), sar
V = [
    ("mxf_xdcam422_1080i25", "mxf", ["-c:v", "mpeg2video", "-b:v", "50M", "-g", "12", "-bf", "2"], "yuv422p", 1920, 1080, (25, 1), "tff", (1, 1)),
    ("mxf_xdcam420_1080i2997", "mxf", ["-c:v", "mpeg2video", "-b:v", "35M", "-g", "15"], "yuv420p", 1920, 1080, (30000, 1001), "tff", (1, 1)),
    ("mxf_mpeg2_720p50", "mxf", ["-c:v", "mpeg2video", "-b:v", "35M"], "yuv422p", 1280, 720, (50, 1), "p", (1, 1)),
    ("mxf_dnxhd_1080p25", "mxf", ["-c:v", "dnxhd", "-b:v", "120M"], "yuv422p", 1920, 1080, (25, 1), "p", (1, 1)),
    ("mxf_imx_576i25", "mxf", ["-c:v", "mpeg2video", "-b:v", "30M", "-g", "1"], "yuv422p", 720, 576, (25, 1), "tff", (64, 45)),
    ("mov_prores422hq_1080i25", "mov", ["-c:v", "prores_ks", "-profile:v", "3"], "yuv422p10le", 1920, 1080, (25, 1), "tff", (1, 1)),
    ("mov_prores422_1080p2398", "mov", ["-c:v", "prores_ks", "-profile:v", "2"], "yuv422p10le", 1920, 1080, (24000, 1001), "p", (1, 1)),
    ("mov_prores4444_alpha_720p30", "mov", ["-c:v", "prores_ks", "-profile:v", "4"], "yuva444p10le", 1280, 720, (30, 1), "p", (1, 1)),
    ("mov_prores_uhd_p50", "mov", ["-c:v", "prores_ks", "-profile:v", "2"], "yuv422p10le", 3840, 2160, (50, 1), "p", (1, 1)),
    ("mov_prores_1080i2997_bff", "mov", ["-c:v", "prores_ks", "-profile:v", "3"], "yuv422p10le", 1920, 1080, (30000, 1001), "bff", (1, 1)),
    ("mov_qtrle_rgb24_1080p25", "mov", ["-c:v", "qtrle"], "rgb24", 1920, 1080, (25, 1), "p", (1, 1)),
    ("mov_qtrle_argb_640x360p24", "mov", ["-c:v", "qtrle"], "argb", 640, 360, (24, 1), "p", (1, 1)),
    ("mov_dv_ntsc_480i", "mov", ["-c:v", "dvvideo"], "yuv411p", 720, 480, (30000, 1001), "bff", (10, 11)),
    ("mov_dv_pal_576i", "mov", ["-c:v", "dvvideo"], "yuv420p", 720, 576, (25, 1), "bff", (16, 15)),
    ("mov_h264_1080p5994", "mov", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv420p", 1920, 1080, (60000, 1001), "p", (1, 1)),
    ("mp4_h264_720p25", "mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv420p", 1280, 720, (25, 1), "p", (1, 1)),
    ("mp4_h264_1080i25_tff", "mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-x264opts", "tff=1"], "yuv420p", 1920, 1080, (25, 1), "tff", (1, 1)),
    ("mp4_h264_high422_10bit_1080p50", "mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv422p10le", 1920, 1080, (50, 1), "p", (1, 1)),
    ("mp4_hevc_main10_uhd_p2997", "mp4", ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "22", "-x265-params", "log-level=error"], "yuv420p10le", 3840, 2160, (30000, 1001), "p", (1, 1)),
    ("mp4_h264_vertical_1080x1920p30", "mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv420p", 1080, 1920, (30, 1), "p", (1, 1)),
    ("mp4_h264_640x480p15", "mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv420p", 640, 480, (15, 1), "p", (1, 1)),
    ("mkv_ffv1_yuv444p16_1080p25", "mkv", ["-c:v", "ffv1", "-level", "3"], "yuv444p16le", 1920, 1080, (25, 1), "p", (1, 1)),
    ("mkv_ffv1_gray_1280x720p50", "mkv", ["-c:v", "ffv1"], "gray", 1280, 720, (50, 1), "p", (1, 1)),
    ("mkv_ffv1_gray10_720x576p25", "mkv", ["-c:v", "ffv1"], "gray10le", 720, 576, (25, 1), "p", (1, 1)),
    ("mkv_ffv1_yuv420p12_1440x1080i25", "mkv", ["-c:v", "ffv1", "-level", "3"], "yuv420p12le", 1440, 1080, (25, 1), "tff", (4, 3)),
    ("mkv_ffv1_gbrp10_1920x1080p24", "mkv", ["-c:v", "ffv1", "-level", "3"], "gbrp10le", 1920, 1080, (24, 1), "p", (1, 1)),
    ("mkv_ffv1_yuva420p_854x480p30", "mkv", ["-c:v", "ffv1"], "yuva420p", 854, 480, (30, 1), "p", (1, 1)),
    ("mkv_h264_2048x858p24", "mkv", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv420p", 2048, 858, (24, 1), "p", (1, 1)),
    ("mkv_h264_odd_718x404p25", "mkv", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv444p", 718, 404, (25, 1), "p", (1, 1)),
    ("mkv_vp9_1280x720p60", "webm", ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "4M"], "yuv420p", 1280, 720, (60, 1), "p", (1, 1)),
    ("ts_h264_1080i2997", "ts", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-x264opts", "tff=1"], "yuv420p", 1920, 1080, (30000, 1001), "tff", (1, 1)),
    ("ts_mpeg2_1080i25", "ts", ["-c:v", "mpeg2video", "-b:v", "25M"], "yuv420p", 1920, 1080, (25, 1), "tff", (1, 1)),
    ("ts_h264_720p5994", "ts", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"], "yuv420p", 1280, 720, (60000, 1001), "p", (1, 1)),
    ("mpg_mpeg2_576i25_169", "mpg", ["-c:v", "mpeg2video", "-b:v", "8M"], "yuv420p", 720, 576, (25, 1), "tff", (64, 45)),
    ("mpg_mpeg2_480i2997", "mpg", ["-c:v", "mpeg2video", "-b:v", "8M"], "yuv420p", 720, 480, (30000, 1001), "bff", (8, 9)),
    ("avi_utvideo_yuv422_1280x720p25", "avi", ["-c:v", "utvideo"], "yuv422p", 1280, 720, (25, 1), "p", (1, 1)),
    ("avi_ffvhuff_rgb24_800x600p20", "avi", ["-c:v", "ffvhuff"], "rgb24", 800, 600, (20, 1), "p", (1, 1)),
    ("mxf_xdcam422_1080p2997", "mxf", ["-c:v", "mpeg2video", "-b:v", "50M", "-g", "15"], "yuv422p", 1920, 1080, (30000, 1001), "p", (1, 1)),
]

# ---- audio specs: list of streams (channels, codec, sample rate); [] = no audio. Optional "dur" offset (s).
A = [
    {"streams": [(2, "pcm_s24le", 48000)]},
    {"streams": [(1, "pcm_s16le", 48000)]},
    {"streams": [(1, "pcm_s24le", 48000)] * 2},
    {"streams": [(1, "pcm_s24le", 48000)] * 4},
    {"streams": [(1, "pcm_s24le", 48000)] * 8},
    {"streams": [(1, "pcm_s24le", 48000)] * 16},
    {"streams": [(2, "pcm_s24le", 48000)] * 4},
    {"streams": [(2, "pcm_s24le", 48000)] * 8},
    {"streams": [(4, "pcm_s24le", 48000)]},
    {"streams": [(6, "pcm_s24le", 48000)]},
    {"streams": [(8, "pcm_s24le", 48000)]},
    {"streams": [(16, "pcm_s24le", 48000)]},
    {"streams": [(8, "pcm_s16le", 48000), (8, "pcm_s16le", 48000)]},
    {"streams": [(2, "pcm_s16le", 48000), (2, "pcm_s24le", 48000)]},          # mixed bit depth -> 24 bit
    {"streams": [(2, "pcm_s24le", 48000), (2, "pcm_s24le", 44100)]},          # mixed rates -> 48 kHz
    {"streams": [(2, "pcm_s16le", 44100)]},                                   # 44.1 kHz kept
    {"streams": [(2, "pcm_s32le", 96000)]},
    {"streams": [(2, "pcm_f32le", 48000)]},
    {"streams": [(1, "pcm_s24le", 48000), (2, "pcm_s24le", 48000), (5, "pcm_s24le", 48000)]},
    {"streams": [(2, "pcm_s24le", 48000)], "dur": +0.6},                      # audio longer than video -> trimmed
    {"streams": [(2, "pcm_s24le", 48000)], "dur": -0.6},                      # shorter -> padded with silence
    {"streams": []},
    {"streams": [(2, "aac", 48000)]},
    {"streams": [(6, "aac", 48000)]},
    {"streams": [(8, "aac", 48000)]},
    {"streams": [(2, "aac", 48000)] * 4},
    {"streams": [(2, "mp2", 48000)]},
    {"streams": [(5, "ac3", 48000)]},
    {"streams": [(2, "opus", 48000)]},
    {"streams": [(6, "flac", 48000)]},
    {"streams": [(1, "aac", 44100)]},
]

LOSSY = {"aac", "mp2", "ac3", "opus"}
# what each container takes (codec -> max channels per stream); mxf: mono PCM tracks at 48 kHz, 16/24 bit only
CONT = {
    "mxf": {"pcm_s16le": 1, "pcm_s24le": 1},
    "mov": {"pcm_s16le": 16, "pcm_s24le": 16, "pcm_s32le": 16, "pcm_f32le": 16, "aac": 8},
    "mp4": {"aac": 8},
    "mkv": {"pcm_s16le": 16, "pcm_s24le": 16, "pcm_s32le": 16, "pcm_f32le": 16, "aac": 8, "flac": 8, "opus": 8, "ac3": 6, "mp2": 2},
    "webm": {"opus": 8},
    "ts": {"aac": 8, "mp2": 2, "ac3": 6},
    "mpg": {"mp2": 2, "ac3": 6},
    "avi": {"pcm_s16le": 16, "pcm_s24le": 16, "mp2": 2, "ac3": 6},
}

# ---- targets: out_h, out_w, out_interlace, num, den
T = [(1080, 1920, "tff", 25, 1), (1080, 1920, "", 50, 1), (720, 1280, "", 50, 1), (576, 720, "tff", 25, 1),
     (1080, 1920, "tff", 30000, 1001), (1080, 1920, "", 25, 1), (480, 720, "bff", 30000, 1001),
     (1080, 1920, "", 60000, 1001), (720, 1280, "", 25, 1), (1080, 1920, "bff", 25, 1), (720, 1280, "", 24000, 1001),
     (2160, 3840, "", 50, 1)]


def fits(cont, a):
    ok = CONT[cont]
    sr_ok = cont != "mxf" or all(s[2] == 48000 for s in a["streams"])
    return sr_ok and all(s[1] in ok and s[0] <= ok[s[1]] for s in a["streams"]) and sum(s[0] for s in a["streams"]) <= 16


def adapt(cont, a):
    """Container-compatible version of audio spec a: same channel layout, codec swapped where needed (mxf: split to
    mono 24-bit tracks), or None."""
    if fits(cont, a):
        return a
    streams = a["streams"]
    if cont == "mxf":
        mono = [(1, "pcm_s24le" if s[1] != "pcm_s16le" else "pcm_s16le", 48000) for s in streams for _ in range(s[0])]
        b = dict(a, streams=mono)
    else:
        pref = [c for c in ("pcm_s24le", "flac", "aac", "opus", "ac3", "mp2") if c in CONT[cont]]
        b = dict(a, streams=[(s[0], next((c for c in pref if CONT[cont][c] >= s[0]), pref[0]), s[2] if cont != "webm" else 48000) for s in streams])
    return b if fits(cont, b) else None


def atag(a):
    """Short audio tag: runs of identical streams as <n>x<ch>ch<codec>, e.g. 16x1ch-s24le, 2ch-aac+2ch-aac44k."""
    runs = []
    for s in a["streams"]:
        if runs and runs[-1][1] == s:
            runs[-1][0] += 1
        else:
            runs.append([1, s])
    t = "+".join((f"{n}x" if n > 1 else "") + f"{s[0]}ch-{s[1].replace('pcm_', '')}" + ("" if s[2] == 48000 else f"{s[2] // 1000}k")
                 for n, s in runs)
    return (t or "noaudio") + {0: "", 1: "-long", -1: "-short"}[(a.get("dur", 0) > 0) - (a.get("dur", 0) < 0)]


def cases():
    out, n = [], 0
    for i in range(120):
        v = V[i % len(V)]
        a = None
        for k in range(len(A)):
            a = adapt(v[1], A[(i * 7 + k) % len(A)])
            if a is not None:
                break
        t = T[(i * 5 + i // len(V)) % len(T)]
        if i % 12 == 0:   # same format as the input: must be a passthrough (rates/sizes the targets list lacks)
            t = (v[5], v[4], "" if v[7] == "p" else v[7], v[6][0], v[6][1])
        if v[5] * v[4] < 3000000 and t[0] == 2160 and i % 3:   # UHD output only now and then from small inputs
            t = T[0]
        name = f"{i:03d}_{v[0]}_{atag(a)}"
        out.append({"name": name, "video": v, "audio": a, "target": t})
    return out


def build(c, path):
    v, a = c["video"], c["audio"]
    _, cont, vargs, pix, w, h, (fn, fd), scan, (sn, sd) = v
    rate = f"{fn}/{fd}"
    dur = DUR if w * h <= 3000000 else 1.0
    if scan == "p":
        vf = f"testsrc2=size={w}x{h}:rate={rate}:duration={dur},setsar={sn}/{sd}"
        il = []
    else:
        vf = (f"testsrc2=size={w}x{h}:rate={2 * fn}/{fd}:duration={dur},"
              f"tinterlace=mode={'interleave_top' if scan == 'tff' else 'interleave_bottom'},"
              f"setfield={scan},setsar={sn}/{sd}")
        il = ["-flags", "+ilme+ildct", "-field_order", "tt" if scan == "tff" else "bb"]
    vf += f",format={pix}"
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", vf]
    maps, codec_args, j = ["-map", "0:v"], [], 0
    adur = dur + a.get("dur", 0.0)
    for si, (ch, codec, sr) in enumerate(a["streams"]):
        expr = "|".join(f"0.5*sin(2*PI*{CH_FREQ[j + k]}*t)" for k in range(ch))
        j += ch
        args += ["-f", "lavfi", "-i", f"aevalsrc={expr}:s={sr}:c={LAYOUT.get(ch, f'{ch}c')}:d={adur}"]
        maps += ["-map", f"{si + 1}:a"]
        codec_args += [f"-c:a:{si}", "libopus" if codec == "opus" else codec]
        if codec in LOSSY:
            codec_args += [f"-b:a:{si}", f"{64 * ch}k" if codec != "mp2" else "256k"]
    extra = []
    if cont == "mxf":
        extra = ["-f", "mxf"]
    if cont in ("mov", "mp4"):
        extra = ["-movflags", "+write_colr"]
    return args + maps + vargs + il + codec_args + extra + [path]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out + "/in", exist_ok=True)
    man, bad = [], []
    for c in cases():
        if a.only and a.only not in c["name"]:
            continue
        path = f"{a.out}/in/{c['name']}.{c['video'][1]}"
        c["file"] = path
        c["dur"] = DUR if c["video"][4] * c["video"][5] <= 3000000 else 1.0
        c["chan_freq"] = CH_FREQ[:sum(s[0] for s in c["audio"]["streams"])]
        if a.force or not os.path.isfile(path):
            cmd = build(c, path)
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode or not os.path.isfile(path):
                bad.append(c["name"])
                print(f"GEN FAIL {c['name']}: {r.stderr.strip()[:400]}")
                continue
            print(f"made {c['name']}")
        man.append(c)
    json.dump(man, open(a.out + "/manifest.json", "w"), indent=1)
    print(f"\n{len(man)} files in {a.out}/in, manifest.json written" + (f"; {len(bad)} could not be made: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
