"""Which cascade explains the fitted chroma H filter? Each hypothesis: 4:2:2 chroma (UHD) -> 4:4:4 upsampling u ->
luma H filter (2:1, integer taps of mode uhd) -> 4:4:4 -> 4:2:2 downsampling d. The combined linear kernel (from unit
impulses, float64) is compared with the max-margin fit (tools/chroma_fit.py).  usage: python -I tools/chroma_cascade.py"""
import json, os, pathlib, itertools
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from preset_io import measured
P = measured()
HY = np.array(P["taps"]["luma_h"]["taps"]) / 16384.0           # offsets -15..15
fit = np.load(os.path.join(os.environ["CUDACONVOLUTION_DATA"], "chroma_probe", "chroma_taps_margin_free.npy"))  # offsets -14..14
N = 256                                                        # UHD chroma samples (test signal length)

def upsample(c, kind):
    """4:2:2 -> 4:4:4 on the UHD luma grid: even positions = chroma samples (co-sited), odd = interpolated"""
    o = np.zeros(2 * len(c)); o[0::2] = c
    cp = np.pad(c, 8, mode="edge")
    def tap(k):  # interpolated value between c[j] and c[j+1] with symmetric kernel k (weights for c[j-n+1..j+n])
        n = len(k) // 2; return sum(k[i] * cp[8 - n + 1 + i: 8 - n + 1 + i + len(c)] for i in range(len(k)))
    if kind == "hold": o[1::2] = c
    elif kind == "hold_next": o[1::2] = cp[9:9 + len(c)]
    elif kind == "linear": o[1::2] = tap([0.5, 0.5])
    elif kind == "cubic": o[1::2] = tap([-1/16, 9/16, 9/16, -1/16])
    elif kind == "halfband": o[1::2] = tap(2 * HY[1::2][::-1][len(HY[1::2]) // 2 - 7: len(HY[1::2]) // 2 + 9][:16] if False else 2 * HY[0::2][8:24] if False else (2 * HY[1::2]))
    else: raise ValueError(kind)
    return o

def luma_h(x):  # 2:1, output n centred on input 2n, replicate borders
    xp = np.pad(x, 15, mode="edge"); n = len(x) // 2
    return np.array([np.dot(HY, xp[2 * i: 2 * i + 31]) for i in range(n)])

def downsample(y, kind):
    """HD 4:4:4 (1920 grid) -> 4:2:2: output x centred on y[2x]"""
    yp = np.pad(y, 2, mode="edge"); m = len(y) // 2; e = yp[2:2 + 2 * m:2]
    if kind == "pick": return e
    if kind == "avg_right": return 0.5 * (e + yp[3:3 + 2 * m:2])
    if kind == "avg_left": return 0.5 * (e + yp[1:1 + 2 * m:2])
    if kind == "121": return 0.25 * yp[1:1 + 2 * m:2] + 0.5 * e + 0.25 * yp[3:3 + 2 * m:2]
    raise ValueError(kind)

def kernel(u, d):
    """combined kernel: response of output x0 (interior) to UHD chroma impulses at 2*x0 + j"""
    x0 = N // 4; k = np.zeros(29)
    for i, j in enumerate(range(-14, 15)):
        c = np.zeros(N); c[2 * x0 + j] = 1.0
        k[i] = downsample(luma_h(upsample(c, u)), d)[x0]
    return k

res = []
for u, d in itertools.product(["hold", "hold_next", "linear", "cubic"], ["pick", "avg_right", "avg_left", "121"]):
    k = kernel(u, d); res.append((np.abs(k - fit).max(), u, d, k))
res.sort(key=lambda r: r[0])
for err, u, d, k in res[:6]:
    print(f"up {u:9s} down {d:9s}  max |kernel - fit| {err:.2e}  sum {k.sum():.6f}")
best = res[0]
print("best kernel:", " ".join(f"{v:.6f}" for v in best[3]))
print("fit        :", " ".join(f"{v:.6f}" for v in fit))
