"""Compare cmp.ps1 outputs of two DLL folders: python -I cmp.py <out\\tag> [a b]"""
import array
import os
import sys

root = sys.argv[1]
a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('ours', 'dtl')
HDR = 80  # MVAnalysisData at the start of a vector frame: struct padding is uninitialised


def md5s(p):
    return [l.rsplit(',', 1)[1].strip() for l in open(p) if not l.startswith('#')]


for name in sorted(os.listdir(os.path.join(root, a))):
    pa, pb = os.path.join(root, a, name), os.path.join(root, b, name)
    if name.endswith('.md5'):
        x, y = md5s(pa), md5s(pb)
        diff = [i for i in range(max(len(x), len(y))) if i >= len(x) or i >= len(y) or x[i] != y[i]]
        print('%-22s frames %d/%d  differ: %s' % (name, len(x), len(y), diff if diff else 'none'))
    elif name.endswith('.raw'):
        x, y = open(pa, 'rb').read(), open(pb, 'rb').read()
        if len(x) != len(y):
            print('%-22s size %d vs %d' % (name, len(x), len(y)))
            continue
        fs = len(x) // 4
        res = []
        for f in range(4):
            u = array.array('i', x[f * fs + HDR:(f + 1) * fs])
            v = array.array('i', y[f * fs + HDR:(f + 1) * fs])
            nd = sum(1 for i in range(len(u)) if u[i] != v[i])
            hd = sum(1 for i in range(HDR) if x[f * fs + i] != y[f * fs + i])
            res.append('%d ints (hdr bytes %d)' % (nd, hd))
        print('%-22s frame %d B  differ per frame [bv10 fv10 bv50 fv50]: %s' % (name, fs, ', '.join(res)))
