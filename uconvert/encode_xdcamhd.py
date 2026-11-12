"""Convert a file with UConvert and encode it as XDCAM HD422 1080i25 (MPEG-2 422P@HL 50 Mbit/s CBR, MXF OP1a).

AviSynth: UConvert(file, 1080, 1920, "tff", 25, 1) from the redistributable folder (dist/UConvert). XDCAM HD422 is
8-bit 4:2:2, so 4:2:0 sources are taken to 4:2:2 in AviSynth before UConvert (per field if interlaced, so the
chroma of the two fields is not mixed); 10 -> 8 bit happens in the encoder. Colour tags are written as BT.709.
Audio: the source channels on tracks 1..n (max 8), silence on the rest, 8 mono PCM 24-bit 48 kHz tracks.
Timecode 00:00:00:00.

usage: python -I encode_xdcamhd.py <input> <output.mxf> [--cachedir DIR] [--threads 12] [--uconvert "extra args"]
"""
import os, sys, argparse, subprocess, time, re

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
AVSI = HERE + "/dist/UConvert/uconvert.avsi"


def avs_path(p):
    return os.path.abspath(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cachedir", default=os.environ.get("TEMP", "."))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--uconvert", default="", help='extra UConvertClip arguments, e.g. \'sar_num=16, sar_den=15\'')
    ap.add_argument("--pre", default="", help="AviSynth expression applied to the source clip c first, e.g. "
                    "'c.FreezeFrame(7152, 7309, 7151)' to hold the last good frame over undecodable source frames")
    a = ap.parse_args()
    src, out = avs_path(a.input), avs_path(a.output)
    extra = (", " + a.uconvert) if a.uconvert else ""
    avs = avs_path(os.path.join(a.cachedir, os.path.splitext(os.path.basename(out))[0] + ".avs"))
    script = f'''Import("{AVSI}")
c = uc_Source("{src}", cachedir="{avs_path(a.cachedir)}")
{("c = " + a.pre) if a.pre else ""}
# after --pre: Prefetch reads ahead, it must not touch frames that --pre replaces
c = c.Prefetch(2)
scan = uc_ScanOf(c)
# XDCAM HD422 is 4:2:2: 4:2:0 -> 4:2:2 before UConvert (per field for interlaced material)
c = c.Is420 ? to422(c, scan) : c
c.UConvertClip(1080, 1920, "tff", 25, 1, threads={a.threads}{extra})

function to422(clip c, string scan) {{
    pt = uc_PT422(c.PixelType)
    if (scan == "p") {{ return c.z_ConvertFormat(pixel_type=pt, resample_filter="spline36") }}
    f = (scan == "tff" ? c.AssumeTFF() : c.AssumeBFF()).SeparateFields().AssumeFrameBased()
    f = f.z_ConvertFormat(pixel_type=pt, resample_filter="spline36").AssumeFieldBased()
    f = scan == "tff" ? f.AssumeTFF() : f.AssumeBFF()
    return f.Weave().propSet("_FieldBased", scan == "tff" ? 2 : 1)
}}
'''
    open(avs, "w").write(script)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=channels", "-of", "csv=p=0", avs],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    nch = int((r.stdout.strip() or "0").splitlines()[0])
    expected = duration(avs)
    if nch > 8:
        print(f"warning: {nch} source channels, only the first 8 are written")
    used = min(nch, 8)
    fc = ["[0:v]setparams=field_mode=tff:range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709,format=yuv422p[v]"]
    if used:
        fc.append(f"[0:a]aresample=48000,asplit={used}" + "".join(f"[s{i}]" for i in range(used)) if used > 1 else "[0:a]aresample=48000[s0]")
        for i in range(used):
            fc.append(f"[s{i}]pan=mono|c0=c{i}[a{i}]")
    if used < 8:
        fc.append(f"anullsrc=r=48000:cl=mono,asplit={8 - used}" + "".join(f"[a{i}]" for i in range(used, 8)) if 8 - used > 1
                  else f"anullsrc=r=48000:cl=mono[a{used}]")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-stats", "-y", "-i", avs, "-filter_complex", ";".join(fc),
           "-map", "[v]"] + sum([["-map", f"[a{i}]"] for i in range(8)], []) + [
           "-shortest", "-c:v", "mpeg2video", "-pix_fmt", "yuv422p", "-profile:v", "0", "-level:v", "2",
           "-b:v", "50000000", "-minrate", "50000000", "-maxrate", "50000000", "-bufsize", "17825792",
           "-rc_init_occupancy", "17825792", "-g", "12", "-bf", "2", "-sc_threshold", "1000000000",
           "-flags", "+ildct+ilme", "-alternate_scan", "1", "-intra_vlc", "1", "-non_linear_quant", "1",
           "-dc", "10", "-qmin", "1", "-qmax", "12", "-c:a", "pcm_s24le", "-ar", "48000",
           "-timecode", "00:00:00:00", "-f", "mxf", out]
    t = time.time()
    r = subprocess.run(cmd)
    # an AviSynth error during the encode looks like end of file to ffmpeg (exit code 0, short file)
    got = duration(out) if r.returncode == 0 else 0.0
    ok = r.returncode == 0 and abs(got - expected) <= 0.1
    print(f"{'done' if ok else 'FAILED'} in {time.time() - t:.0f} s: {out}\n  script: {avs}\n"
          f"  duration {got:.2f} s, expected {expected:.2f} s")
    print(f"  audio: {nch} source channel(s) on track(s) 1-{used}, silence on the rest" if used
          else "  audio: none in source, 8 silent tracks")
    sys.exit(0 if ok else 1)


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


if __name__ == "__main__":
    main()
