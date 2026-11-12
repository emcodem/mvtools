"""Run UConvert over the generated input matrix (gen_matrix.py) and check every case against the spec:

1. input sanity (ffprobe): the generated file is what the manifest says (scan, size, audio layout), so a failure
   below is UConvert's, not the generator's;
2. chain: the stages in _uc_plan against an independent model of the decision tables in DESIGN.md
   (deint / scale / frc none-drop-repeat-mc / pictures x2 / interlace / passthrough), and the scaled picture keeps
   the input's display aspect inside the output raster;
3. output: size, scan / field order, frame rate, frame count, pixel format = input's, SAR;
4. audio: channel count (all streams merged in order, up to 16), sample rate (kept; 48 kHz when the streams differ),
   length = video length, and per channel the tone gen_matrix.py put on that channel: right frequency, level,
   no crosstalk from the other channels, silence where the source audio ended;
5. the whole output decodes without errors.

usage: python -I tests/run_matrix.py [name-substring ...] [-j 3] [--cpu] [--dir work/uc_matrix]
"""
import os, sys, json, re, argparse, subprocess, threading
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UC = os.path.dirname(HERE).replace("\\", "/")
sys.path.insert(0, HERE)
from ucdata import data_root
ROOT = data_root()

INFO = r'''
function pv(clip c, string k) { return propGetType(c, k) == 1 ? string(propGetInt(c, k)) : "-" }
function ps(clip c, string k) { return propGetType(c, k) == 3 ? propGetString(c, k) : "-" }
c = last
Assert(false, "INFO~~w=" + string(c.Width) + "~~h=" + string(c.Height) + "~~fb=" + pv(c, "_FieldBased") \
  + "~~parity=" + (c.GetParity(0) ? "T" : "B") + "~~fn=" + string(c.FrameRateNumerator) + "~~fd=" + string(c.FrameRateDenominator) \
  + "~~n=" + string(c.FrameCount) + "~~pt=" + c.PixelType + "~~sarn=" + pv(c, "_SARNum") + "~~sard=" + pv(c, "_SARDen") \
  + "~~ach=" + (c.HasAudio ? string(c.AudioChannels) : "0") + "~~arate=" + (c.HasAudio ? string(c.AudioRate) : "0") \
  + "~~alen=" + (c.HasAudio ? string(c.AudioLength) : "0") + "~~scan=" + SCAN + "~~plan=" + ps(c, "_uc_plan") \
  + "~~source=" + ps(c, "_uc_source") + "~~END")
'''
_print = threading.Lock()


def avs_info(path, body):
    open(path, "w").write(body)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "null", "-"], capture_output=True,
                       text=True, errors="replace")
    m = re.search(r"INFO~~(.*?)~~END", r.stderr, re.S)
    if not m:
        return None, r.stderr.strip()
    return dict(kv.split("=", 1) for kv in m.group(1).replace("\n", " ").split("~~")), ""


def probe(file):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", file], capture_output=True, text=True)
    return json.loads(r.stdout or "{}").get("streams", [])


def is_int(x):
    return x >= 1 and abs(x - round(x)) <= 1e-6 * x


def same(a, b):
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b))


def expected_chain(c, src, plan):
    """Stage tokens the plan must contain, from the spec (not from uconvert.avsi)."""
    v, t = c["video"], c["target"]
    w, h, (fn, fd), scan = v[4], v[5], v[6], v[7]
    oh, ow, oi, on, od = t
    fps, ofps = Fraction(fn, fd), Fraction(on, od)
    r = 2 * fps if scan != "p" else fps
    m = 2 * ofps if oi else ofps
    half = r < Fraction(3, 4) * m and not is_int(float(m / r))
    p = m / 2 if half else m
    if same(float(r), float(p)):
        frc = None
    elif r > p and is_int(float(r / p)):
        frc = f"frc=drop 1/{round(r / p)}"
    elif p > r and is_int(float(p / r)):
        frc = f"frc=repeat x{round(p / r)}"
    else:
        frc = f"frc=mc {p.numerator}/{p.denominator}"
    raster = (w, h) == (ow, oh)
    if raster and fps == ofps and scan == (oi or "p"):
        return ["passthrough"]
    tok = []
    if scan != "p":
        tok.append(f"deint {scan}")
    # scaling runs before the FRC when it makes the picture smaller (less to convert), after it otherwise; the scaled
    # picture size itself is checked in check_scale (aspect, fills the raster in one dimension)
    m_sc = re.search(r"scale=(\d+)x(\d+)", plan)
    down = (int(m_sc.group(1)) * int(m_sc.group(2)) if m_sc else ow * oh) < w * h
    # frame repetition runs after the scaling (identical frames are scaled once)
    pre = not raster and (down or (frc or "").startswith("frc=repeat"))
    if pre:
        tok.append("scale")
    if frc:
        tok.append(frc)
    if not raster and not pre:
        tok.append("scale")
    if half:
        tok.append("pictures x2")
    if oi:
        tok.append(f"interlace={oi}")
    return tok


def plan_tokens(plan):
    """Normalised stage tokens of a _uc_plan string (GPU/CPU names folded: qtgmc-gpu -> deint, splat -> mc)."""
    out = []
    for part in plan.split(" | ")[1:-1]:
        part = part.strip()
        if part.startswith("deint="):
            out.append("deint " + part.split()[-1])
        elif part.startswith("scale="):
            out.append("scale")
        elif part.startswith("frc="):
            part = re.sub(r"^frc=(splat|mc) ", "frc=mc ", part)
            m = re.match(r"frc=mc (\d+)/(\d+)$", part)
            if m:   # rates are printed unreduced (60000/2002)
                f = Fraction(int(m.group(1)), int(m.group(2)))
                part = f"frc=mc {f.numerator}/{f.denominator}"
            out.append(part)
        elif part.startswith("interlace="):
            out.append(part.split()[0])
        else:
            out.append(part)
    return out


def check_scale(plan, c, sar, errs):
    """sar: the file's real sample aspect (ffprobe; some encoders, e.g. DV, set their own)."""
    m = re.search(r"scale=(\d+)x(\d+) in (\d+)x(\d+)", plan)
    if not m:
        return
    sw, sh, ow, oh = map(int, m.groups())
    v = c["video"]
    dar = v[4] * sar[0] / (v[5] * sar[1])
    if sw > ow or sh > oh or (sw != ow and sh != oh):
        errs.append(f"scaled picture {sw}x{sh} does not fill {ow}x{oh} in one dimension")
    tol = 4.0 / min(sw, sh) + 0.002
    if abs(sw / sh - dar) / dar > tol:
        errs.append(f"aspect: scaled {sw}x{sh} = {sw / sh:.4f}, input display aspect {dar:.4f}")


def tones(x, rate, freqs):
    """Amplitude (linear, peak) at each of freqs for every channel of x (samples x channels). Flat-top window: the
    amplitude is right to ~0.01 dB wherever the tone falls between bins (a Hann window loses up to 1.4 dB)."""
    n = x.shape[0]
    k = np.arange(n) * 2 * np.pi / (n - 1)
    win = (0.21557895 - 0.41663158 * np.cos(k) + 0.277263158 * np.cos(2 * k) - 0.083578947 * np.cos(3 * k)
           + 0.006947368 * np.cos(4 * k))
    spec = np.abs(np.fft.rfft(x * win[:, None], axis=0)) * 2 / win.sum()
    bins = np.fft.rfftfreq(n, 1 / rate)
    out = np.zeros((x.shape[1], len(freqs)))
    for k, f in enumerate(freqs):
        sel = np.abs(bins - f) <= max(3, 2 * bins[1])
        out[:, k] = spec[sel].max(axis=0)
    peak_f = bins[np.argmax(spec[1:], axis=0) + 1]
    return out, peak_f


def decode_raw(args, nch):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error"] + args + ["-c:a", "pcm_f32le", "-f", "f32le", "-"],
                       capture_output=True)
    if r.returncode or not r.stdout:
        return None, r.stderr.decode(errors="replace").strip()[:300]
    return np.frombuffer(r.stdout, np.float32).reshape(-1, nch).astype(np.float64), ""


def reference_order(c, notes):
    """Tone per channel as ffmpeg's own decoder delivers each source stream (the order a pass-through must keep).
    Codecs may re-map a layout (e.g. AAC 6.0 / 7.1 channel configurations): noted, not an error."""
    freqs, j, ref = c["chan_freq"], 0, []
    for si, (ch, codec, sr) in enumerate(c["audio"]["streams"]):
        x, e = decode_raw(["-i", c["file"], "-map", f"0:a:{si}"], ch)
        if x is None:
            return None, f"reference decode of stream {si}: {e}"
        seg = x[int(0.2 * sr):max(int(0.2 * sr) + 2048, int((c['dur'] + c['audio'].get('dur', 0.0) - 0.2) * sr))]
        amp, _ = tones(seg, sr, freqs[j:j + ch])
        order = [freqs[j + int(np.argmax(amp[k]))] if amp[k].max() > 0.05 else 0 for k in range(ch)]
        if order != freqs[j:j + ch]:
            notes.append(f"stream {si} ({ch}ch {codec}): decoder channel order {order} (generated {freqs[j:j + ch]})")
        ref += order
        j += ch
    return ref, ""


def check_audio(c, info, avs_audio, errs, notes):
    a = c["audio"]["streams"]
    nch = int(info["ach"])
    freqs, e = reference_order(c, notes) if a else ([], "")
    if freqs is None:
        errs.append(e)
        return
    if nch != len(freqs):
        errs.append(f"audio: {nch} channels, expected {len(freqs)}")
        return
    if not freqs:
        return
    rates = {s[2] for s in a}
    exp_rate = rates.pop() if len(rates) == 1 else 48000
    rate = int(info["arate"])
    if rate != exp_rate:
        errs.append(f"audio: {rate} Hz, expected {exp_rate}")
    vdur = int(info["n"]) * int(info["fd"]) / int(info["fn"])
    adur = int(info["alen"]) / rate
    if abs(adur - vdur) > 1.5 / rate + 1e-6:
        errs.append(f"audio length {adur:.5f} s != video length {vdur:.5f} s")
    x, e = decode_raw(["-i", avs_audio, "-vn"], nch)
    if x is None:
        errs.append("audio decode: " + e)
        return
    src_end = c["dur"] + c["audio"].get("dur", 0.0)          # where the source audio stops
    lossy = any(s[1] in ("aac", "mp2", "ac3", "opus") for s in a)
    t0 = 0.15
    t1 = max(t0 + 0.1, min(src_end, vdur) - 0.15)
    seg = x[int(t0 * rate):int(t1 * rate)]
    amp, peak_f = tones(seg, rate, [f or 1 for f in freqs])
    for ch in range(nch):
        if freqs[ch] == 0:   # silent in the reference decode too
            if np.abs(seg[:, ch]).max() > 1e-3:
                errs.append(f"audio ch {ch}: not silent, the source decoder gives silence")
            continue
        want = amp[ch, ch]
        other = max([amp[ch, k] for k in range(nch) if k != ch and freqs[k] not in (0, freqs[ch])], default=0.0)
        lvl = 20 * np.log10(want + 1e-12) - 20 * np.log10(0.5)
        if abs(peak_f[ch] - freqs[ch]) > max(3, rate / max(len(seg), 1)):
            got = int(np.argmax(amp[ch]))
            hint = f" (= channel {got}'s tone)" if amp[ch, got] > 0.05 else ""
            errs.append(f"audio ch {ch}: strongest {peak_f[ch]:.0f} Hz{hint}, expected {freqs[ch]} Hz")
        elif abs(lvl) > (3.0 if lossy else 0.5):
            errs.append(f"audio ch {ch}: level {lvl:+.1f} dB against the source")
        elif other > want * (10 ** (-30 / 20)):
            errs.append(f"audio ch {ch}: crosstalk {20 * np.log10(other / want):.0f} dB")
    if src_end < vdur - 0.1:   # padded part must be silent
        tail = x[int((src_end + 0.05) * rate):]
        if tail.size and np.sqrt((tail ** 2).mean()) > 1e-4:
            errs.append(f"audio after the source's end ({src_end:.2f} s) is not silent")


def run_case(c, work, gpu_line, cache):
    name, f = c["name"], c["file"]
    errs, notes = [], []
    head = f'Import("{UC}/load_plugins.avs")\nImport("{UC}/uconvert.avsi")\n{gpu_line}'
    t = c["target"]
    call = (f'UConvert("{f}", {t[0]}, {t[1]}, "{t[2]}", {t[3]}, {t[4]}, cachedir="{cache}")')

    # 1. input sanity
    st = probe(f)
    vs = [s for s in st if s.get("codec_type") == "video"]
    aus = [s for s in st if s.get("codec_type") == "audio"]
    v = c["video"]
    if not vs or (vs[0].get("width"), vs[0].get("height")) != (v[4], v[5]):
        errs.append("GEN: video stream missing or wrong size")
    if [s.get("channels") for s in aus] != [s[0] for s in c["audio"]["streams"]]:
        errs.append(f"GEN: audio streams {[s.get('channels') for s in aus]} != {[s[0] for s in c['audio']['streams']]}")
    fo = vs[0].get("field_order", "unknown") if vs else "unknown"
    if errs:
        return name, errs, notes
    m = re.match(r"(\d+):(\d+)$", vs[0].get("sample_aspect_ratio", ""))
    sar = (int(m.group(1)), int(m.group(2))) if m and int(m.group(1)) > 0 else (1, 1)   # what the file really carries

    # input as the source stage sees it
    src, e = avs_info(f"{work}/{name}_src.avs", head.replace("SCAN", "") +
                      f'c = uc_Source("{f}", cachedir="{cache}")\nglobal SCAN = uc_ScanOf(c)\nc\n' + INFO)
    if not src:
        return name, ["source stage: " + e[:500]], notes
    if src["scan"] != v[7]:
        errs.append(f"input scan detected as '{src['scan']}', file is '{v[7]}' (ffprobe field_order {fo})")
    src_sar = (int(src["sarn"]), int(src["sard"])) if src["sarn"].isdigit() and src["sard"].isdigit() and int(src["sarn"]) > 0 else (1, 1)
    if Fraction(*src_sar) != Fraction(*sar) and abs(src_sar[0] / src_sar[1] / (sar[0] / sar[1]) - 1) < 0.03:
        notes.append(f"input SAR {src_sar[0]}:{src_sar[1]} vs stream {sar[0]}:{sar[1]}: two readings of the same aspect "
                     "(e.g. DV 4:3 generic 8:9 / ITU 10:11), accepted")
        sar = src_sar
    elif Fraction(*src_sar) != Fraction(*sar):
        errs.append(f"input SAR detected as {src_sar[0]}:{src_sar[1]}, file carries {sar[0]}:{sar[1]} (source: {src['source']})")

    # 2./3. conversion
    out, e = avs_info(f"{work}/{name}_info.avs", head + call + '\nglobal SCAN = "-"\n' + INFO)
    if not out:
        return name, errs + ["UConvert: " + e[:600]], notes
    plan = out["plan"]
    notes.append(plan)
    exp = expected_chain(c, src, plan)
    got = plan_tokens(plan)
    if not errs and got != exp:   # a wrong scan detection explains a different chain already
        errs.append(f"chain {got} != expected {exp}")
    check_scale(plan, c, sar, errs)
    oh, ow, oi, on, od = t
    if (int(out["w"]), int(out["h"])) != (ow, oh):
        errs.append(f"size {out['w']}x{out['h']} != {ow}x{oh}")
    fb_exp = {"tff": "2", "bff": "1", "": "0"}[oi]
    if out["fb"] != fb_exp or (oi and out["parity"] != ("T" if oi == "tff" else "B")):
        errs.append(f"scan: _FieldBased {out['fb']} parity {out['parity']}, expected {oi or 'progressive'}")
    if Fraction(int(out["fn"]), int(out["fd"])) != Fraction(on, od):
        errs.append(f"rate {out['fn']}/{out['fd']} != {on}/{od}")
    if out["pt"] != src["pt"]:
        errs.append(f"pixel type {out['pt']} != input {src['pt']}")
    n_exp = int(src["n"]) * Fraction(int(src["fd"]), int(src["fn"])) * Fraction(on, od)
    if abs(int(out["n"]) - n_exp) > 2:
        errs.append(f"frame count {out['n']}, expected about {float(n_exp):.1f}")
    raster = (v[4], v[5]) == (ow, oh)
    src_sar = (src["sarn"], src["sard"]) if src["sarn"] not in ("-", "0") and src["sard"] not in ("-", "0") else ("1", "1")
    sar_exp = src_sar if raster else ("1", "1")
    if (out["sarn"], out["sard"]) != sar_exp:
        errs.append(f"SAR {out['sarn']}:{out['sard']}, expected {sar_exp[0]}:{sar_exp[1]}")

    # 4. audio
    avs_audio = f"{work}/{name}_audio.avs"
    open(avs_audio, "w").write(head + call + "\nKillVideo()\n")
    check_audio(c, out, avs_audio, errs, notes)

    # 5. full decode
    avs = f"{work}/{name}.avs"
    open(avs, "w").write(head + call + "\nBitsPerComponent() == 32 ? ConvertBits(16) : last\n")
    bad = decode_errors(avs)
    if bad:
        # the same error with the source filter alone (no UConvert) is upstream: ffmpeg's AviSynth reader or the
        # source plugin (e.g. no YUVA444P12 in the reader, short last PCM packet with many channels)
        src_avs = f"{work}/{name}_srconly.avs"
        open(src_avs, "w").write(head + f'uc_Source("{f}", cachedir="{cache}")\n')
        sig = lambda l: re.sub(r"\[[^\]]*@ [0-9a-fx]+\]|\S+\.avs\.?|\d+", "", l).strip()
        up = {sig(l) for l in decode_errors(src_avs)}
        own = [l for l in bad if sig(l) not in up]
        if own:
            errs.append("decode: " + " / ".join(own)[:400])
        else:
            notes.append("upstream (same with the source filter alone): " + bad[0][:200])
    return name, errs, notes


def decode_errors(avs):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", avs, "-f", "null", "-"], capture_output=True,
                       text=True, errors="replace")
    bad = [l.strip() for l in r.stderr.splitlines() if "rror" in l or "xception" in l or "Invalid" in l or "unknown" in l]
    return bad if (r.returncode or bad) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="CPU chain (uc_gpu_default = false)")
    ap.add_argument("--dir", default=ROOT + "/work/uc_matrix")
    a = ap.parse_args()
    man = json.load(open(a.dir + "/manifest.json"))
    sel = [c for c in man if not a.names or any(s in c["name"] for s in a.names)]
    work = a.dir + ("/run_cpu" if a.cpu else "/run_gpu")
    cache = a.dir + "/cache"
    os.makedirs(work, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    gpu_line = "global uc_gpu_default = false\n" if a.cpu else ""
    results = []

    def one(c):
        try:
            res = run_case(c, work, gpu_line, cache)
        except Exception as ex:  # a checker bug must not hide the other cases
            res = (c["name"], [f"checker exception: {ex!r}"], [])
        with _print:
            name, errs, notes = res
            print(("PASS " if not errs else "FAIL ") + name + ("" if not errs else "\n    " + "\n    ".join(errs))
                  + ("" if not notes else "\n    plan: " + "\n    note: ".join(notes)), flush=True)
        return res

    with ThreadPoolExecutor(max_workers=a.j) as ex:
        results = list(ex.map(one, sel))
    bad = [r for r in results if r[1]]
    # report.json keeps the last result of every case: a partial rerun updates its cases, the others stay
    rp = work + "/report.json"
    old = {r["name"]: r for r in json.load(open(rp))} if os.path.isfile(rp) else {}
    for n, e, p in results:
        old[n] = {"name": n, "errors": e, "plan": (p[0] if p else ""), "notes": p[1:]}
    json.dump([old[k] for k in sorted(old)], open(rp, "w"), indent=1)
    print(f"\n{len(results) - len(bad)}/{len(results)} passed ({'CPU' if a.cpu else 'GPU'} chain); report: {work}/report.json")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
