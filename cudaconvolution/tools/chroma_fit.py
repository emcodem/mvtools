"""Chroma H analysis on the chroma probe (tools/gen_chroma_probe.py) against the reference output.
Needs CUDACONVOLUTION_DATA/chroma_probe/{cin.yuv (probe, 100 UHD frames), cref.yuv (reference output, 50 HD frames)}.
Inside a 120-line band all chroma columns are constant, so the V pass is the identity there and each output row is the
chroma H pass of one source row.  usage: python -I tools/chroma_fit.py [L]"""
import os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from preset_io import measured
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, identity
L = int(sys.argv[1]) if len(sys.argv) > 1 else 14
S = os.path.join(os.environ["CUDACONVOLUTION_DATA"], "chroma_probe")
W, H, CW, BAND = 3840, 2160, 1920, 120
cin = np.memmap(os.path.join(S, "cin.yuv"), np.uint16, "r").reshape(100, -1)
cref = np.memmap(os.path.join(S, "cref.yuv"), np.uint16, "r").reshape(50, -1)
def chroma(buf, w, h, pl):  # pl 1 = U, 2 = V
    y = w * h; c = w // 2 * h
    return buf[y + (pl - 1) * c: y + pl * c].reshape(h, w // 2)
T = np.arange(-L, L + 1)

def json_preset():
    import json, pathlib
    return measured()

def rows(frames, cols):
    """samples (X, a, meta) from the band centres: X[:, j] = source chroma at column 2x + T[j], a = reference output"""
    Xs, As, Ms = [], [], []
    for f in frames:
        k, p = f // 2, f % 2
        for pl in (1, 2):
            src = chroma(cin[f], W, H, pl); out = chroma(cref[k], 1920, 1080, pl)
            for b in range(H // BAND):
                y = (BAND * b + 60) // 2; y += (y - p) % 2          # output row of parity p near the band centre
                s = src[2 * y].astype(np.int64); a = out[y].astype(np.int64)
                Xs.append(np.stack([s[2 * cols + j] for j in T], 1)); As.append(a[cols])
                Ms.append(np.stack([np.full(len(cols), f), np.full(len(cols), pl), cols], 1))
    X = np.concatenate(Xs) - 512; a = np.concatenate(As); M = np.concatenate(Ms)
    keep = (a > 1) & (a < 1020)
    return X[keep].astype(np.float64), a[keep], M[keep]

def bounds(a):
    """interval of o with 512 + trunc(o + 0.5) == a"""
    lo = np.where(a < 512, a - 513.5, np.where(a > 512, a - 512.5, -1.5))
    hi = np.where(a < 512, a - 512.5, np.where(a > 512, a - 511.5, 0.5))
    return lo, hi

def lp(X, a, eps=1e-6):
    n, d = X.shape; lo, hi = bounds(a); I = identity(n, format="csr")
    Aub = vstack([hstack([csr_matrix(-X), -I]), hstack([csr_matrix(X), -I])]).tocsr()
    r = linprog(np.r_[np.zeros(d), np.ones(n)], A_ub=Aub, b_ub=np.r_[-lo - eps, hi - eps],
                bounds=[(None, None)] * d + [(0, None)] * n, method="highs")
    return r.x[:d]

def exact(t, X, a):
    o = X @ t; return 512 + np.trunc(o + 0.5) == a

if __name__ == "__main__":
    interior = np.arange(L // 2 + 8, 960 - L // 2 - 8)
    X, a, M = rows(range(60, 90), interior)
    rng = np.random.default_rng(1); sel = rng.choice(len(a), 60000, replace=False)
    t = lp(X[sel], a[sel])
    for it in range(6):                                   # cutting planes: add violated samples, refit
        ok = exact(t, X, a); bad = np.flatnonzero(~ok)
        print(f"iter {it}: fit on {len(sel)} samples, exact {ok.mean() * 100:.5f} % of {len(a)} ({len(bad)} wrong), sum {t.sum():.8f}", flush=True)
        if len(bad) == 0: break
        sel = np.union1d(sel, bad[:20000]); t = lp(X[sel], a[sel])
    np.save(os.path.join(S, f"chroma_taps_L{L}.npy"), t)
    print("taps:", " ".join(f"{v:.9f}" for v in t))
    Xc, ac, _ = rows(range(90, 100), interior)
    print(f"random luma frames (crosstalk check): exact {exact(t, Xc, ac).mean() * 100:.5f} % of {len(ac)}")

def lp_margin(X, a, sym=False):
    """taps maximising the minimum distance m of X t to the rounding-interval borders (robust centre of the
    feasible set); sym: t_j = t_-j. Returns (t, m); m < 0 means infeasible."""
    lo, hi = bounds(a); n = X.shape[0]
    if sym:  # parameters: t_0..t_L (t_k for |j| = k)
        Xs = np.stack([X[:, L]] + [X[:, L - k] + X[:, L + k] for k in range(1, L + 1)], 1)
    else:
        Xs = X
    d = Xs.shape[1]; one = np.ones((n, 1))
    A = vstack([hstack([csr_matrix(-Xs), csr_matrix(one)]), hstack([csr_matrix(Xs), csr_matrix(one)])]).tocsr()
    r = linprog(np.r_[np.zeros(d), -1.0], A_ub=A, b_ub=np.r_[-lo, hi], bounds=[(None, None)] * d + [(None, 1.0)],
                method="highs")
    t = r.x[:d]
    if sym: t = np.r_[t[1:][::-1], t[0], t[1:]]
    return t, r.x[d]

def fit_margin(X, a, sym, n0=60000, rounds=8):
    rng = np.random.default_rng(2); sel = rng.choice(len(a), n0, replace=False)
    for it in range(rounds):
        t, m = lp_margin(X[sel], a[sel], sym)
        lo, hi = bounds(a); o = X @ t; slack = np.minimum(o - lo, hi - o)
        bad = np.flatnonzero(slack < m - 1e-9)
        print(f"  {'sym ' if sym else 'free'} iter {it}: margin {m:.3e} on {len(sel)} samples; all samples: min slack {slack.min():.3e},"
              f" exact {np.mean(slack >= 0) * 100:.5f} %", flush=True)
        if len(bad) == 0: break
        sel = np.union1d(sel, bad[np.argsort(slack[bad])][:5000])
    return t, m
