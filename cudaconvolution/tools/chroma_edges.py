"""Edge kernels of the chroma H pass (mode uhd): for each of the outermost output chroma columns, fit a kernel on the
source columns that exist (left edge: source 0..2x+R; right edge: 2x-R..1919), so any special border handling of the
reference is absorbed. Data: chroma probe frames 40-99 (edge impulses, random columns).
Optionally also the noise frames of the main probe (CUDACONVOLUTION_DATA/pin.yuv, pref.yuv): there the known V filter is folded
into the features (the H and V passes are linear), so every output row is a sample.
usage: CUDACONVOLUTION_DATA=<folder> python -I tools/chroma_edges.py [ncols] [R] [main-probe frames, e.g. 150-174:2]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import chroma_fit as cf
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
R = int(sys.argv[2]) if len(sys.argv) > 2 else 16
CW = 1920

def samples(x, side, frames):
    """X: source chroma columns (absolute index list), a: reference output at output column x (left) / 959-x (right)"""
    if side == "left": cols = np.arange(0, 2 * x + R + 1); ox = x
    else: ox = 959 - x; cols = np.arange(2 * ox - R, CW)
    Xs, As = [], []
    for f in frames:
        k, p = f // 2, f % 2
        for pl in (1, 2):
            src = cf.chroma(cf.cin[f], cf.W, cf.H, pl); out = cf.chroma(cf.cref[k], 1920, 1080, pl)
            for b in range(cf.H // cf.BAND):
                y = (cf.BAND * b + 60) // 2; y += (y - p) % 2
                Xs.append(src[2 * y, cols].astype(np.int64) - 512); As.append(int(out[y, ox]))
    X = np.array(Xs, dtype=np.float64); a = np.array(As)
    keep = (a > 1) & (a < 1020)
    return X[keep], a[keep], cols

MAIN = sys.argv[3] if len(sys.argv) > 3 else ""
VT = np.array(cf.json_preset()["taps"]["v"]["taps"]) / 16384.0
HC = np.array(cf.json_preset()["taps"]["chroma_h"]["taps"], dtype=np.float64)

def main_samples(x, side, frames):
    """noise frames of the main probe: features = V filter applied to the source columns (replicated rows)"""
    pin = np.memmap(os.path.join(os.environ["CUDACONVOLUTION_DATA"], "pin.yuv"), np.uint16, "r").reshape(200, -1)
    pref = np.memmap(os.path.join(os.environ["CUDACONVOLUTION_DATA"], "pref.yuv"), np.uint16, "r").reshape(100, -1)
    if side == "left": cols = np.arange(0, 2 * x + R + 1); ox = x
    else: ox = 959 - x; cols = np.arange(2 * ox - R, CW)
    Xs, As = [], []
    for f in frames:
        k, p = f // 2, f % 2
        for pl in (1, 2):
            src = cf.chroma(pin[f], cf.W, cf.H, pl)[:, cols].astype(np.float64) - 512
            out = cf.chroma(pref[k], 1920, 1080, pl)[:, ox].astype(np.int64)
            ys = np.arange(p, 1080, 2)
            feat = sum(c * src[np.clip(2 * ys + v, 0, cf.H - 1)] for c, v in zip(VT, range(-8, 9)))
            # rows whose H intermediates (estimated with the interior kernel on replicated sources) come near the
            # intermediate clip [-512, 508] are not linear: drop them
            if side == "left": full = cf.chroma(pin[f], cf.W, cf.H, pl)[:, :2 * ox + 15].astype(np.float64) - 512
            else: full = cf.chroma(pin[f], cf.W, cf.H, pl)[:, 2 * ox - 14:].astype(np.float64) - 512
            idx = np.clip(np.arange(2 * ox - 14, 2 * ox + 15) - (0 if side == "left" else 2 * ox - 14), 0, full.shape[1] - 1)
            hmid = full[:, idx] @ HC
            near = np.zeros(len(ys), bool)
            for v in range(-8, 9): near |= np.abs(hmid[np.clip(2 * ys + v, 0, cf.H - 1)] + 2) > 500
            Xs.append(feat[~near]); As.append(out[ys][~near])
    X = np.concatenate(Xs); a = np.concatenate(As); keep = (a > 1) & (a < 1020)
    return X[keep], a[keep]

def parse_frames(spec):
    a, b = spec.split(":")[0].split("-"); step = int(spec.split(":")[1]) if ":" in spec else 1
    return range(int(a), int(b) + 1, step)

if __name__ == "__main__":
    result = {"left": [], "right": []}
    for side in ("left", "right"):
        for x in range(NC):
            X, a, cols = samples(x, side, range(40, 100))
            if MAIN:
                Xm, am = main_samples(x, side, parse_frames(MAIN)); X = np.concatenate([X, Xm]); a = np.concatenate([a, am])
            t, m = cf.lp_margin(X, a)
            lo, hi = cf.bounds(a); o = X @ t; ex = np.mean((o >= lo) & (o <= hi)) * 100
            # interior kernel (replicate) for comparison: how wrong is it here?
            print(f"{side:5s} output col {x if side == 'left' else 959 - x:3d}: {len(a)} samples, margin {m:+.2e}, exact {ex:.3f} %,"
                  f" source cols {cols[0]}..{cols[-1]}, taps sum {t.sum():.6f}", flush=True)
            result[side].append({"x": int(x), "src_first": int(cols[0] if side == "left" else cols[0] - CW), "taps": [float(v) for v in np.float32(t)]})
    json.dump(result, open(os.path.join(cf.S, "chroma_edge_kernels.json"), "w"), indent=1)
