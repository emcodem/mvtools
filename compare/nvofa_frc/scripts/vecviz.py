"""Vector maps of the crop region for output frames 41 / 42: exact search | NVOFA medium | NVOFA fast.

python -I vecviz.py <work folder>    (needs <work>/vec_{recalc,analyse}_{exact,medium,fast}.raw, see render_vec.ps1)

Raw files: the vector clips of avs/frc_vec_recalc.avs / frc_vec_analyse.avs, 4 frames each: bv 61, fv 62 (used for
output frame 41 = source 61.25), bv 62, fv 63 (output frame 42 = source 62.5). Layout per frame: 64 header words, then
[total, validity], then per level (coarsest first) [size, (vx, vy, sad) * blocks]; level 0 is the last one. Vectors
in 1/2 pixel (pel 2).

Per block (drawn as its centre tile, step x step pixels):
- flow map: hue = direction, saturation = length (white = no motion), one scale for the three variants;
- outliers (white outline): the vector differs from the median of its 8 neighbours by more than OUTLIER_PX pixels;
- SAD map: grey, log scale, one scale for the three variants.
Writes PPMs to <work>/vecviz and prints per-cell statistics for the cells named in the visual judgement (cells.json).
"""
import array
import colorsys
import json
import math
import os
import sys

HDR = 64
X0, Y0, CW, CH = 2880, 240, 960, 540   # crop of the comparison (cells of 120 px: A..H, 1..5)
CELL = 120
OUTLIER_PX = 3.0
PEL = 2
KINDS = {'recalc': (16, 8, 479, 269), 'analyse': (32, 16, 239, 134)}   # size, step, blocks x, blocks y
VARIANTS = ['exact', 'medium', 'fast']
FRAMES = [('41', 'bv', 'bv 61 -> 62'), ('41', 'fv', 'fv 62 -> 61'), ('42', 'bv', 'bv 62 -> 63'), ('42', 'fv', 'fv 63 -> 62')]
# cells from the visual judgement (frame -> variant -> cells)
JUDGED = {'41': {'exact': ['B1', 'B2', 'C2', 'D1', 'D2', 'D3', 'E3', 'H1'], 'fast': ['B2']},
          '42': {'exact': ['D2'], 'fast': ['B4']}}


def level0(path, nbx, nby):
    b = open(path, 'rb').read()
    fs = len(b) // 4
    out = []
    for f in range(4):
        a = array.array('i', b[f * fs:(f + 1) * fs])
        p = HDR + 2
        last = None
        while p < len(a) and a[p] > 0:
            last = p
            p += a[p]
        assert a[last] == nbx * nby * 3 + 1, (path, a[last])
        q = a[last + 1:last + 1 + nbx * nby * 3]
        out.append([(q[3 * i], q[3 * i + 1], q[3 * i + 2]) for i in range(nbx * nby)])
    return out


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def flow_color(vx, vy, vmax):
    m = math.hypot(vx, vy)
    if m == 0:
        return (255, 255, 255)
    h = (math.atan2(-vy, vx) / (2 * math.pi)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, min(1.0, m / vmax), 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def ppm(path, w, h, px):
    with open(path, 'wb') as f:
        f.write(b'P6 %d %d 255\n' % (w, h))
        f.write(bytes(px))


def cell_of(x, y):
    return '%s%d' % (chr(ord('A') + x // CELL), y // CELL + 1)


work = sys.argv[1]
od = os.path.join(work, 'vecviz')
os.makedirs(od, exist_ok=True)
stats = {}
for kind, (size, step, nbx, nby) in KINDS.items():
    data = {v: level0(os.path.join(work, 'vec_%s_%s.raw' % (kind, v)), nbx, nby) for v in VARIANTS}
    off = (size - step) // 2   # centre tile of block b: [b * step + off, b * step + off + step)
    # blocks whose centre tile lies in the crop
    bxs = [bx for bx in range(nbx) if X0 <= bx * step + off < X0 + CW]
    bys = [by for by in range(nby) if Y0 <= by * step + off < Y0 + CH]
    for fi, (frame, d, label) in enumerate(FRAMES):
        crop_vec = {v: [data[v][fi][by * nbx + bx] for by in bys for bx in bxs] for v in VARIANTS}
        mags = sorted(math.hypot(x, y) for x, y, _ in crop_vec['exact'])
        vmax = max(8.0, mags[int(len(mags) * 0.98)])
        sads = sorted(s for v in VARIANTS for _, _, s in crop_vec[v])
        smax = max(1, sads[int(len(sads) * 0.99)])
        for v in VARIANTS:
            V = data[v][fi]
            flow = bytearray(CW * CH * 3)
            sadm = bytearray(CW * CH * 3)
            per_cell = {}
            for by in bys:
                for bx in bxs:
                    x, y, s = V[by * nbx + bx]
                    nb = [V[(by + j) * nbx + bx + i] for j in (-1, 0, 1) for i in (-1, 0, 1)
                          if (i or j) and 0 <= bx + i < nbx and 0 <= by + j < nby]
                    dev = math.hypot(x - median([n[0] for n in nb]), y - median([n[1] for n in nb])) / PEL
                    out = dev > OUTLIER_PX
                    tx, ty = bx * step + off - X0, by * step + off - Y0
                    c = cell_of(tx, ty)
                    pc = per_cell.setdefault(c, [0, 0, 0])
                    pc[0] += 1
                    pc[1] += out
                    pc[2] += s
                    fc = flow_color(x, y, vmax)
                    g = int(255 * min(1.0, math.log1p(s) / math.log1p(smax)))
                    for yy in range(ty, min(ty + step, CH)):
                        for xx in range(tx, min(tx + step, CW)):
                            edge = out and (yy in (ty, ty + step - 1) or xx in (tx, tx + step - 1))
                            k = 3 * (yy * CW + xx)
                            flow[k:k + 3] = bytes((255, 255, 255) if edge else (fc if not out else tuple(int(q * 0.8) for q in fc)))
                            sadm[k:k + 3] = bytes((255, 60, 200) if edge else (g, g, g))
            ppm(os.path.join(od, '%s_%s_%s_%s_flow.ppm' % (kind, frame, d, v)), CW, CH, flow)
            ppm(os.path.join(od, '%s_%s_%s_%s_sad.ppm' % (kind, frame, d, v)), CW, CH, sadm)
            tot = [sum(p[i] for p in per_cell.values()) for i in range(3)]
            stats.setdefault(kind, {}).setdefault(frame + ' ' + d, {})[v] = {
                'outliers': tot[1], 'blocks': tot[0], 'mean_sad': round(tot[2] / tot[0]),
                'cells': {c: {'outliers': p[1], 'blocks': p[0], 'mean_sad': round(p[2] / p[0])} for c, p in sorted(per_cell.items())}}
        stats[kind][frame + ' ' + d]['scale'] = {'vmax_halfpel': vmax, 'sad_max': smax}

json.dump(stats, open(os.path.join(od, 'stats.json'), 'w'), indent=1)

# summary: whole crop, then the judged cells
for kind in KINDS:
    print('== %s (level 0)' % kind)
    for key, sv in stats[kind].items():
        print('  %-6s ' % key + '  '.join('%s: %4d/%d outliers, mean SAD %6d' % (v, sv[v]['outliers'], sv[v]['blocks'], sv[v]['mean_sad'])
                                            for v in VARIANTS))
    for frame, judged in JUDGED.items():
        cells = sorted({c for cs in judged.values() for c in cs})
        for d in ('bv', 'fv'):
            sv = stats[kind][frame + ' ' + d]
            row = []
            for c in cells:
                row.append('%s ' % c + '/'.join('%d' % sv[v]['cells'][c]['outliers'] for v in VARIANTS))
            print('  frame %s %s, outliers exact/medium/fast per judged cell: %s' % (frame, d, ', '.join(row)))

# ---- figures: per kind / frame / direction one PNG, row 1 flow maps, row 2 SAD maps, exact | medium | fast, with the
# cell grid (A1..H5) of the slow-motion clip; written to ../img/vec_<kind>_<frame>_<dir>.png
import subprocess

FONT = r'C\:/Windows/Fonts/consola.ttf'
HEAD = 48
img = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')
labels = ','.join("drawtext=fontfile='%s':text='%s%d':x=%d:y=%d:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=3"
                  % (FONT, chr(ord('A') + c), r + 1, c * CELL + 4, HEAD + r * CELL + 4)
                  for r in range((CH + CELL - 1) // CELL) for c in range(CW // CELL))
for kind in KINDS:
    for frame, d, label in FRAMES:
        ins, chains = [], []
        for row, what in enumerate(('flow', 'sad')):
            for i, v in enumerate(VARIANTS):
                n = row * len(VARIANTS) + i
                ins += ['-i', os.path.join(od, '%s_%s_%s_%s_%s.ppm' % (kind, frame, d, v, what))]
                sv = stats[kind][frame + ' ' + d][v]
                title = '%s  %s  out %s  %s' % ({'exact': 'exact search', 'medium': 'NVOFA medium', 'fast': 'NVOFA fast'}[v],
                                                 'vectors' if what == 'flow' else 'SAD', frame, label)
                info = '%d outliers  mean SAD %d' % (sv['outliers'], sv['mean_sad'])
                chains.append("[%d:v]pad=%d:%d:0:%d:color=black,drawgrid=x=0:y=%d:w=%d:h=%d:t=1:c=yellow@0.6,%s,"
                              "drawbox=x=0:y=0:w=%d:h=%d:color=black:t=fill,"
                              "drawtext=fontfile='%s':text='%s':x=10:y=4:fontsize=19:fontcolor=white,"
                              "drawtext=fontfile='%s':text='%s':x=10:y=26:fontsize=17:fontcolor=yellow[p%d]"
                              % (n, CW, CH + HEAD, HEAD, HEAD, CELL, CELL, labels, CW, HEAD, FONT, title, FONT, info, n))
        graph = ';'.join(chains) + ';[p0][p1][p2]hstack=3[r0];[p3][p4][p5]hstack=3[r1];[r0][r1]vstack[out]'
        gf = os.path.join(od, 'graph.txt')
        open(gf, 'w').write(graph)
        out = os.path.join(img, 'vec_%s_%s_%s.png' % (kind, frame, d))
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error'] + ins +
                       ['-/filter_complex', gf, '-map', '[out]', '-frames:v', '1', out], check=True)
        print(os.path.normpath(out))
