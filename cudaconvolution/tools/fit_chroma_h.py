"""Provisional chroma H taps for mode "uhd" of preset balanced: LP fit (min sum of rounding-interval violations) on the vertical
stripe items of the probe (frames 100-125: V pass is the identity there), U and V together, support +-L chroma px.
usage: CUDACONVOLUTION_DATA=<folder> python -I tools/fit_chroma_h.py [L]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, identity
from pio import inp, ref
L = int(sys.argv[1]) if len(sys.argv) > 1 else 14
T = np.arange(-L, L + 1)
def samples(pl, rows):
    Xs, As = [], []
    for f in range(100, 126):
        k, p = f // 2, f % 2
        s = inp(f, pl)[100].astype(np.float64); A = ref(k, pl)[p::2]
        for row in rows:
            a = A[row].astype(np.float64); xs = np.arange(20, len(a) - 20)
            Xs.append(np.stack([s[2 * xs + j] for j in T], 1)); As.append(a[xs])
    X = np.concatenate(Xs); a = np.concatenate(As); m = (a > 1) & (a < 1020)
    return X[m] - 512, a[m]
def bounds(a):   # interval of o (before 512 + trunc(o + .5)) giving output a
    lo = np.where(a < 512, a - 512 - 1.5, a - 512 - .5)
    return lo, lo + 1
def lp(X, a, eps=1e-7):
    n, d = X.shape; lo, hi = bounds(a); I = identity(n, format="csr")
    Aub = vstack([hstack([csr_matrix(-X), -I]), hstack([csr_matrix(X), -I])]).tocsr()
    r = linprog(np.r_[np.zeros(d), np.ones(n)], A_ub=Aub, b_ub=np.r_[-lo, hi - eps],
                bounds=[(None, None)] * d + [(0, None)] * n, method="highs")
    return r.x[:d], r.x[d:]
def exact(t, X, a):
    o = X @ t; return np.mean(512 + np.trunc(o + .5) == a)
X, a = map(np.concatenate, zip(*(samples(pl, [30]) for pl in (1, 2))))
Xv, av = map(np.concatenate, zip(*(samples(pl, [200, 400]) for pl in (1, 2))))       # held-out rows
t, e = lp(X, a)
print(f"L={L}: fit samples {len(a)}, violations {(e > 1e-6).sum()}, sum {t.sum():.6f}")
print(f"float taps: exact fit {exact(t, X, a):.6f}, held-out {exact(t, Xv, av):.6f}")
for nb in (14, 16, 20):
    D = 2 ** nb; k = np.round(t * D).astype(int); k[L] += D - k.sum()       # unity DC via the centre tap
    print(f"2^{nb}: held-out exact {exact(k / D, Xv, av):.6f}  taps {k.tolist()}")
