"""Cross-check of the embedded self-test vector "balanced/h2v2 YUV422P10" with the python reference model
(independent of the C++ engines). Reproduces cudaconvolution::selftest_fill.  usage: python -I tools/selftest_check.py"""
import sys, os, json, hashlib, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ref_model import frame
def fill(w, h, f, pl, bits):
    x = (0x9E3779B9 ^ (((f * 3 + pl + 1) * 0x85EBCA6B) & 0xFFFFFFFF)) & 0xFFFFFFFF; out = np.empty(w * h, np.int64)
    for i in range(w * h):
        x ^= (x << 13) & 0xFFFFFFFF; x ^= x >> 17; x ^= (x << 5) & 0xFFFFFFFF
        out[i] = (x >> 8) & ((1 << bits) - 1)
    return out.reshape(h, w)
W, H = 256, 96   # case_size() for factor 2
src = [[fill(W if pl == 0 else W // 2, H, f, pl, 10) for pl in range(3)] for f in range(2)]
out = frame(src[0], src[1], out_clip=(0, 1023))  # self-test vectors use the default: no output clipping
md5 = hashlib.md5(b"".join(np.ascontiguousarray(p, dtype="<u2").tobytes() for p in out)).hexdigest()
emb = json.loads((pathlib.Path(__file__).resolve().parent.parent / "selftest" / "vectors.json").read_text())["balanced/h2v2 YUV422P10"]
print("python model", md5); print("embedded    ", emb); print("MATCH" if md5 == emb else "DIFFERENT"); sys.exit(md5 != emb)
