"""Bit-exact probe for the reference 50p UHD -> 1080i25: uncompressed v210 UHD 50p, 200 frames (4 s).
usage: python -I gen_probe.py <out.mov>      (writes <out>.md5.txt = per-frame md5 of the yuv422p10le source)"""
import sys, subprocess, hashlib
import numpy as np
OUT = sys.argv[1]
W, H, CW = 3840, 2160, 1920
rng = np.random.default_rng(20261014)

def flat_tiles(f):        # frames 0-49: 120x120 flat tiles, every luma/chroma code 4..1019 several times
    T = 120; nx, ny = W // T, H // T; t = np.arange(nx * ny).reshape(ny, nx)
    ly = (f * 577 + t) % 1024; lu = (f * 389 + 7 * t) % 1024; lv = (f * 211 + 13 * t) % 1024
    Y = np.kron(ly, np.ones((T, T), int)); U = np.kron(lu, np.ones((T, T // 2), int)); V = np.kron(lv, np.ones((T, T // 2), int))
    return Y, U, V

BG = [512, 200, 800, 64, 940, 360, 650]
def impulses(f):          # frames 50-99: isolated +-impulses on flat background, all 4x4 UHD phases
    i = f - 50; bg = BG[i % len(BG)]; py, px = (i // 2) % 4, (i // 8) % 4 if i < 32 else (i % 4)
    Y = np.full((H, W), bg); U = np.full((H, CW), 512); V = np.full((H, CW), 512)
    ys, xs = np.arange(32 + py, H - 32, 64), np.arange(32 + px, W - 32, 64)
    s = np.where((np.arange(len(ys))[:, None] + np.arange(len(xs))[None, :]) % 2 == 0, 1, -1)
    Y[np.ix_(ys, xs)] = np.clip(bg + 400 * s, 4, 1019)
    cx = np.arange(32 + px, CW - 32, 64); sc = np.where((np.arange(len(ys))[:, None] + np.arange(len(cx))[None, :]) % 2 == 0, 1, -1)
    U[np.ix_(ys, cx)] = 512 + 300 * sc; V[np.ix_(ys, cx)] = 512 - 300 * sc
    return Y, U, V

def stripes(f):           # frames 100-125: vertical stripes (H pass only); 126-149: horizontal stripes (V pass only)
    if f <= 125:
        return [np.broadcast_to(rng.integers(4, 1020, w), (H, w)) for w in (W, CW, CW)]
    return [np.broadcast_to(rng.integers(4, 1020, H)[:, None], (H, w)) for w in (W, CW, CW)]

LV = [64, 128, 256, 400, 512, 640, 768, 900, 940, 1000, 20]
def noise(f):             # frames 150-174: iid uniform 4..1019 full frame (also edge mode); 175-199: +-8 around levels
    if f < 175:
        return [rng.integers(4, 1020, (H, w)) for w in (W, CW, CW)]
    lv = LV[(f - 175) % len(LV)]
    return [np.clip(lv + rng.integers(-8, 9, (H, w)), 0, 1023) for w in (W, CW, CW)]

def frame(f):
    g = flat_tiles if f < 50 else impulses if f < 100 else stripes if f < 150 else noise
    # ffmpeg's v210 encoder clips to 4..1019, so the probe uses only those codes
    return b"".join(np.ascontiguousarray(np.clip(p, 4, 1019), dtype="<u2").tobytes() for p in g(f))

cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
       "-f", "rawvideo", "-pix_fmt", "yuv422p10le", "-s", f"{W}x{H}", "-r", "50", "-i", "-",
       "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
       # tag via setparams: output -color_* options make this ffmpeg build insert a colour conversion
       "-vf", "setparams=field_mode=prog:range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
       "-c:v", "v210", "-c:a", "pcm_s24le", "-timecode", "00:00:00:00", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
md5 = []
for f in range(200):
    b = frame(f); md5.append(hashlib.md5(b).hexdigest()); p.stdin.write(b)
p.stdin.close(); rc = p.wait()
open(OUT + ".md5.txt", "w").write("\n".join(md5) + "\n")
print("ffmpeg rc", rc, file=sys.stderr)
