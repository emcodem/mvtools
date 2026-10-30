"""Per-block maps from MRecalculate vector frames (479x269 blocks, 16x16 / overlap 8) written as PPM, plus statistics.
python viz.py <ours vec.raw> <dtl smooth=2 vec.raw> <out dir>"""
import array
import colorsys
import json
import math
import os
import sys

BX, BY = 479, 269
HDR = 64  # int32 words before [total, validity, size]
NAMES = ['bv10', 'fv10', 'bv50', 'fv50']


def frames(path):
    b = open(path, 'rb').read()
    fs = len(b) // 4
    out = []
    for f in range(4):
        a = array.array('i', b[f * fs:(f + 1) * fs])
        assert a[HDR + 2] == BX * BY * 3 + 1
        p = a[HDR + 3:]
        out.append([(p[3 * i], p[3 * i + 1], p[3 * i + 2]) for i in range(BX * BY)])
    return out


def ppm(path, px):
    with open(path, 'wb') as f:
        f.write(b'P6 %d %d 255\n' % (BX, BY))
        f.write(bytes(px))


def flow_color(vx, vy, vmax):
    # optical-flow colouring: hue = direction, saturation = length (white = no motion)
    m = math.hypot(vx, vy)
    if m == 0:
        return (255, 255, 255)
    h = (math.atan2(-vy, vx) / (2 * math.pi)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, min(1.0, m / vmax), 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


a, b = frames(sys.argv[1]), frames(sys.argv[2])
out = sys.argv[3]
stats = {}
for k, name in enumerate(NAMES):
    A, B = a[k], b[k]
    mags = sorted(math.hypot(v[0], v[1]) for v in A)
    vmax = max(4.0, mags[int(len(mags) * 0.99)])
    for tag, V in (('pinterf', A), ('dtl_sm2', B)):
        px = bytearray()
        for v in V:
            px += bytes(flow_color(v[0], v[1], vmax))
        ppm(os.path.join(out, '%s_%s.ppm' % (name, tag)), px)
    # change map: grey = unchanged block (by SAD lightness), colour = changed vector, brighter = larger change
    px = bytearray()
    changed = sad_better = sad_worse = 0
    dmax = 0
    for va, vb in zip(A, B):
        d = math.hypot(va[0] - vb[0], va[1] - vb[1])
        if d:
            changed += 1
            dmax = max(dmax, d)
            sad_better += vb[2] < va[2]
            sad_worse += vb[2] > va[2]
    for va, vb in zip(A, B):
        d = math.hypot(va[0] - vb[0], va[1] - vb[1])
        if d == 0:
            g = 40
            px += bytes((g, g, g))
        else:
            t = min(1.0, math.log1p(d) / math.log1p(max(dmax, 2)))
            r, g, bb = colorsys.hsv_to_rgb(0.12 - 0.12 * t, 0.9, 0.55 + 0.45 * t)
            px += bytes((int(r * 255), int(g * 255), int(bb * 255)))
    ppm(os.path.join(out, '%s_change.ppm' % name), px)
    stats[name] = {'blocks': BX * BY, 'changed': changed, 'sad_lower': sad_better, 'sad_higher': sad_worse,
                   'max_change_qpel': dmax, 'vmax_colour': vmax,
                   'sad_sum_pinterf': sum(v[2] for v in A), 'sad_sum_dtl': sum(v[2] for v in B)}
json.dump(stats, open(os.path.join(out, 'stats.json'), 'w'), indent=1)
print(json.dumps(stats, indent=1))
