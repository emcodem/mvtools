"""Luma of tools/ref_model.py vs the reference output on the probe.  usage: python -I tools/p_verify.py <edge> <k0> <k1>
(expected: replicate 0 100 -> 0 mismatches in 207,360,000 samples)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pio import inp, ref
from ref_model import field, HY, HYR
edge = sys.argv[1]; ks = range(int(sys.argv[2]), int(sys.argv[3]))
tot = {"interior": [0, 0], "border": [0, 0]}; byk = []
B = 24
for k in ks:
    A = ref(k, 0).astype(np.int64); bad_k = 0
    for p in (0, 1):
        out = field(inp(2*k+p, 0), p, HY, HYR, edge); a = A[p::2]
        inter = np.zeros(a.shape, bool); inter[B//2:-B//2, B:-B] = True
        d = out != a
        for nm, m in (("interior", inter), ("border", ~inter)): tot[nm][0] += int(d[m].sum()); tot[nm][1] += int(m.sum())
        bad_k += int(d[inter].sum())
    byk.append(bad_k)
print(edge, {k: f"{v[0]} / {v[1]}" for k, v in tot.items()}); print("interior mismatches per output frame:", byk)
