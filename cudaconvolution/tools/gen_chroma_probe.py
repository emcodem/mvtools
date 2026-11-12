"""Chroma probe for the reference UHD -> 1080i: identifies the chroma H filter and its edge kernel.
Uncompressed v210 UHD 50p, 100 frames (2 s), BT.709 tags. All chroma content is constant down each column inside
120-line bands (18 per frame), so away from band boundaries (> 8 UHD lines) the V pass is the identity and the
output shows the chroma H pass alone.
  frames  0-39  interior single-column impulses (U +A, V other amplitudes), amplitudes +-10..+-400, backgrounds
                512/200/800/64/940, column position mod 4 = f mod 4; luma flat 512
  frames 40-59  edge impulses: chroma column k and 1919-k, k = 0..23, +-300, backgrounds 512/200; luma flat 512
  frames 60-89  random chroma columns 4..1019 per band (dense data for LS / LP / MILP incl. edges); luma flat 512
  frames 90-99  as 60-89 but luma random columns too (crosstalk check: does chroma depend on luma?)
usage: python -I gen_chroma_probe.py <out.mov>   (writes <out>.md5.txt = per-frame md5 of the yuv422p10le source)"""
import sys, subprocess, hashlib
import numpy as np
OUT = sys.argv[1]
W, H, CW, BAND = 3840, 2160, 1920, 120
NB = H // BAND
rng = np.random.default_rng(20261015)
AMP = [400, -400, 200, -200, 100, -100, 50, -50, 25, -25, 300, -300, 150, -150, 75, -75, 10, -10]
BG = [512, 200, 800, 64, 940]

def bands(rows):          # rows: NB x width values -> H x width, constant down each band
    return np.repeat(np.asarray(rows), BAND, axis=0)

def interior(f):
    bg = BG[(f // 4) % len(BG)]; p = f % 4; xs = np.arange(32 + p, CW - 32, 64)
    U = np.full((NB, CW), bg); V = np.full((NB, CW), bg)
    for b in range(NB):
        U[b, xs] = bg + AMP[b]; V[b, xs] = bg + AMP[(b + 7) % NB]
    return np.full((H, W), 512), bands(U), bands(V)

def edges(f):
    g = f - 40; U = np.empty((NB, CW), int); V = np.empty((NB, CW), int)
    for b in range(NB):
        k = (g * NB + b) % 24; bg = 512 if (b // 2) % 2 == 0 else 200; a = 300 if (g + b) % 2 == 0 else -300
        U[b] = bg; V[b] = bg
        U[b, k] = bg + a; U[b, CW - 1 - k] = bg + a; V[b, k] = bg - a; V[b, CW - 1 - k] = bg - a
    return np.full((H, W), 512), bands(U), bands(V)

def random_cols(f):
    U = rng.integers(4, 1020, (NB, CW)); V = rng.integers(4, 1020, (NB, CW))
    Y = bands(rng.integers(4, 1020, (NB, W))) if f >= 90 else np.full((H, W), 512)
    return Y, bands(U), bands(V)

def frame(f):
    g = interior if f < 40 else edges if f < 60 else random_cols
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
for f in range(100):
    b = frame(f); md5.append(hashlib.md5(b).hexdigest()); p.stdin.write(b)
p.stdin.close(); rc = p.wait()
open(OUT + ".md5.txt", "w").write("\n".join(md5) + "\n")
print("ffmpeg rc", rc, file=sys.stderr)
