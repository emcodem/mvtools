"""Pre-screen for the visual check: finds the worst spots (output frame + 120 px cell of the full UHD frame) of each
variant from its vectors, plus spots where a lambda variant is worse than uc_FRC, and cuts 2x2 crops of them.

<venv python with numpy> prescreen.py <work folder> [--top 6]

Variants: uc_FRC (avs/frc_flow.avs), MAnalyse plevel=0 lambda=4000 (frc_flow_pl0_l4000.avs), MAnalyse lambda=1000
(frc_flow_l1000.avs), NVOFA medium (frc_flow.avs with MVCUDA_NVOFA=medium).

Per interpolated output frame k (MFlowFps frame k + 8 = source position 1.25 (k + 8), between source frames n and n + 1;
MFlowFps uses bv n and fv n + 1) and per block of the MRecalculate vectors (16/8, 479 x 269):
- outlier: the vector differs from the median of its 8 neighbours by more than 3 px (bv n and fv n + 1);
- temporal jump: the block's change since the previous source frame differs from the median change of its 8 neighbours
  by more than 4 px (flicker of single blocks; a pan that speeds up moves all blocks alike and does not count);
- SAD of both vectors.
Per cell (120 x 120 px, columns A..AF, rows 1..18; block assigned by its centre): score = outlier share + jump share
(0..2), mean SAD. Spots: per variant the highest scores (neighbouring frames / cells of one spot merged), and
regressions: lambda variant score - uc_FRC score > 0.25 with SAD at least 1.2x uc_FRC's.
Output: <work>/prescreen/spots.json, ../img/spot_<nn>.jpg (2x2, full-frame grid, cell labels), printed list.
"""
import argparse
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--top', type=int, default=6)
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', 'img')
OD = os.path.join(a.work, 'prescreen')
os.makedirs(OD, exist_ok=True)
VARIANTS = [('ucfrc', 'uc_FRC (lambda 16000)', 'frc_flow.avs', ''),
            ('pl0_l4000', 'plevel 0 lambda 4000', 'frc_flow_pl0_l4000.avs', ''),
            ('l1000', 'lambda 1000', 'frc_flow_l1000.avs', ''),
            ('medium', 'NVOFA medium', 'frc_flow.avs', 'medium')]
NBX, NBY, STEP, BLK = 479, 269, 8, 16
HDR = 64
WORDS = HDR + 3 + NBX * NBY * 3
CELL, NCX, NCY = 120, 32, 18
OUT_FRAMES = 112
PEL = 2
cx_of = np.minimum((np.arange(NBX) * STEP + BLK // 2) // CELL, NCX - 1)
cy_of = np.minimum((np.arange(NBY) * STEP + BLK // 2) // CELL, NCY - 1)
cell_index = (cy_of[:, None] * NCX + cx_of[None, :]).ravel()
cell_count = np.bincount(cell_index, minlength=NCX * NCY).astype(np.float64)


def cell_name(cx, cy):
    col = chr(ord('A') + cx) if cx < 26 else 'A' + chr(ord('A') + cx - 26)
    return '%s%d' % (col, cy + 1)


def nb_median(f):
    """median of the 8 neighbours of every block (edges replicated), f: (NBY, NBX)"""
    p = np.pad(f, 1, mode='edge')
    sh = [p[1 + j:1 + j + NBY, 1 + i:1 + i + NBX] for j in (-1, 0, 1) for i in (-1, 0, 1) if i or j]
    return np.median(np.stack(sh), axis=0)


def outliers(vx, vy, thr_px):
    return np.hypot(vx - nb_median(vx), vy - nb_median(vy)) > thr_px * PEL


def per_cell(mask_or_val):
    return np.bincount(cell_index, weights=mask_or_val.ravel().astype(np.float64), minlength=NCX * NCY) / cell_count


def vec_avs(flow_avs, out_path):
    lines = [l for l in open(os.path.join(HERE, 'avs', flow_avs)).read().splitlines()
             if not l.startswith('out =') and not l.startswith('return')]
    open(out_path, 'w').write('\n'.join(lines) + '\nreturn Interleave(bv, fv)\n')


def output_frames():
    """interpolated output frames k -> left source frame n"""
    res = {}
    for k in range(OUT_FRAMES):
        pos4 = 5 * (k + 8)          # 4 * source position
        if pos4 % 4:
            res[k] = pos4 // 4
    return res


ks = output_frames()
need = {}
for k, n in ks.items():
    need.setdefault(n, []).append(k)

metrics = {}
for key, label, flow, nvofa in VARIANTS:
    cache = os.path.join(OD, 'metrics_%s.npz' % key)
    if os.path.exists(cache):
        metrics[key] = dict(np.load(cache))
        continue
    avs = os.path.join(OD, 'vec_%s.avs' % key)
    vec_avs(flow, avs)
    env = dict(os.environ)
    env.pop('MVCUDA_NVOFA', None)
    if nvofa:
        env['MVCUDA_NVOFA'] = nvofa
    pr = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', avs, '-f', 'rawvideo', '-'],
                          stdout=subprocess.PIPE, env=env)
    score = np.full((OUT_FRAMES, NCY, NCX), np.nan)
    outl = np.full((OUT_FRAMES, NCY, NCX), np.nan)
    jump = np.full((OUT_FRAMES, NCY, NCX), np.nan)
    sad = np.full((OUT_FRAMES, NCY, NCX), np.nan)
    hist = {}   # source frame -> (bv, fv) arrays (3, NBY, NBX)
    n = 0
    last_n = max(need) + 1
    while n <= last_n:
        pair = []
        for _ in range(2):
            b = pr.stdout.read(WORDS * 4)
            if len(b) < WORDS * 4:
                break
            w = np.frombuffer(b, dtype=np.int32)
            assert w[HDR + 2] == NBX * NBY * 3 + 1
            pair.append(w[HDR + 3:].reshape(NBY, NBX, 3).transpose(2, 0, 1).astype(np.int64))
        if len(pair) < 2:
            break
        hist[n] = pair
        hist.pop(n - 3, None)
        m = n - 1   # output frames between m and m + 1 are complete now (bv m, fv m + 1, previous frames for jumps)
        if m in need and m - 1 in hist:
            bv, fv = hist[m][0], hist[m + 1][1]
            bvp, fvp = hist[m - 1][0], hist[m][1]
            o = (outliers(bv[0], bv[1], 3.0).astype(np.float64) + outliers(fv[0], fv[1], 3.0)) / 2
            j = (outliers(bv[0] - bvp[0], bv[1] - bvp[1], 4.0).astype(np.float64) + outliers(fv[0] - fvp[0], fv[1] - fvp[1], 4.0)) / 2
            oc, jc = per_cell(o).reshape(NCY, NCX), per_cell(j).reshape(NCY, NCX)
            sc = per_cell((bv[2] + fv[2]) / 2.0).reshape(NCY, NCX)
            for k in need[m]:
                outl[k], jump[k], sad[k], score[k] = oc, jc, sc, oc + jc
        n += 1
    pr.stdout.close()
    pr.wait()
    metrics[key] = {'score': score, 'outl': outl, 'jump': jump, 'sad': sad}
    np.savez(cache, **metrics[key])
    print('scanned %s' % key, flush=True)


def pick(score, filt=None, top=6):
    """highest cells, merging spots closer than 3 frames and 2 cells"""
    s = score.copy()
    if filt is not None:
        s[~filt] = -1
    s = np.nan_to_num(s, nan=-1)
    out = []
    order = np.argsort(s, axis=None)[::-1]
    for idx in order:
        if len(out) >= top or s.flat[idx] <= 0:
            break
        k, cy, cx = np.unravel_index(idx, s.shape)
        if any(abs(k - k2) < 3 and abs(cy - y2) <= 2 and abs(cx - x2) <= 2 for k2, y2, x2 in out):
            continue
        out.append((int(k), int(cy), int(cx)))
    return out


spots = []
for key, label, _, _ in VARIANTS:
    for k, cy, cx in pick(metrics[key]['score'], top=a.top):
        spots.append({'why': 'worst of ' + label, 'variant': key, 'k': k, 'cy': cy, 'cx': cx})
base = metrics['ucfrc']
for key in ('pl0_l4000', 'l1000'):
    m = metrics[key]
    filt = (m['score'] - base['score'] > 0.25) & (m['sad'] >= 1.2 * base['sad'])
    for k, cy, cx in pick(m['score'] - base['score'], filt, top=a.top):
        spots.append({'why': 'regression of %s vs uc_FRC' % dict((v[0], v[1]) for v in VARIANTS)[key],
                      'variant': key, 'k': k, 'cy': cy, 'cx': cx})
# one entry per (frame, cell): merge reasons
merged = {}
for s in spots:
    merged.setdefault((s['k'], s['cy'], s['cx']), []).append(s['why'])
spots = [{'k': k, 'cy': cy, 'cx': cx, 'cell': cell_name(cx, cy), 'why': w} for (k, cy, cx), w in sorted(merged.items())]
for i, s in enumerate(spots):
    s['id'] = i + 1
    s['values'] = {key: {'score': round(float(metrics[key]['score'][s['k'], s['cy'], s['cx']]), 2),
                         'outliers': round(float(metrics[key]['outl'][s['k'], s['cy'], s['cx']]), 2),
                         'jumps': round(float(metrics[key]['jump'][s['k'], s['cy'], s['cx']]), 2),
                         'sad': int(metrics[key]['sad'][s['k'], s['cy'], s['cx']])} for key, _, _, _ in VARIANTS}
json.dump(spots, open(os.path.join(OD, 'spots.json'), 'w'), indent=1)

# ---- crops: 960 x 540 window (8 x 4.5 cells, aligned to the cell grid) around each spot, 4 variants 2 x 2
FONT = 'C\\:/Windows/Fonts/consola.ttf'
frames = sorted({s['k'] for s in spots})
for key, label, flow, nvofa in VARIANTS:
    env = dict(os.environ)
    env.pop('MVCUDA_NVOFA', None)
    if nvofa:
        env['MVCUDA_NVOFA'] = nvofa
    sel = '+'.join('eq(n,%d)' % k for k in frames)
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error', '-i', os.path.join(HERE, 'avs', flow),
                    '-vf', "select='%s'" % sel, '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
                    os.path.join(OD, key + '_%03d.png')], check=True, env=env)
    # select numbers the written frames 1.. in order
    for i, k in enumerate(frames):
        src, dst = os.path.join(OD, key + '_%03d.png' % (i + 1)), os.path.join(OD, '%s_k%03d.png' % (key, k))
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)

HEAD = 48
for s in spots:
    x0 = min(max((s['cx'] - 3) * CELL, 0), 3840 - 960)
    y0 = min(max((s['cy'] - 2) * CELL, 0), 2160 - 540)
    lab = []
    for r in range(5):
        for c in range(8):
            if y0 + r * CELL >= 2160:
                continue
            lab.append("drawtext=fontfile='%s':text='%s':x=%d:y=%d:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=3"
                       % (FONT, cell_name(x0 // CELL + c, y0 // CELL + r), c * CELL + 4, HEAD + r * CELL + 4))
    # the flagged cell: orange frame
    fx, fy = s['cx'] * CELL - x0, s['cy'] * CELL - y0 + HEAD
    chains, ins = [], []
    for i, (key, label, _, _) in enumerate(VARIANTS):
        ins += ['-i', os.path.join(OD, '%s_k%03d.png' % (key, s['k']))]
        v = s['values'][key]
        info = 'score %.2f (outl %.2f jump %.2f)  SAD %d' % (v['score'], v['outliers'], v['jumps'], v['sad'])
        chains.append("[%d:v]crop=960:540:%d:%d,pad=960:%d:0:%d:color=black,drawgrid=x=0:y=%d:w=%d:h=%d:t=1:c=yellow@0.6,%s,"
                      "drawbox=x=%d:y=%d:w=%d:h=%d:color=0xff8000:t=3,drawbox=x=0:y=0:w=960:h=%d:color=black:t=fill,"
                      "drawtext=fontfile='%s':text='%s   frame %d   cell %s':x=10:y=4:fontsize=20:fontcolor=white,"
                      "drawtext=fontfile='%s':text='%s':x=10:y=27:fontsize=17:fontcolor=yellow[p%d]"
                      % (i, x0, y0, 540 + HEAD, HEAD, HEAD, CELL, CELL, ','.join(lab), fx, fy, CELL, CELL, HEAD,
                         FONT, label, s['k'], s['cell'], FONT, info, i))
    graph = ';'.join(chains) + ';[p0][p1]hstack=2[r0];[p2][p3]hstack=2[r1];[r0][r1]vstack=2[out]'
    gf = os.path.join(OD, 'graph.txt')
    open(gf, 'w').write(graph)
    s['image'] = 'img/spot_%02d.jpg' % s['id']
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error'] + ins +
                   ['-/filter_complex', gf, '-map', '[out]', '-frames:v', '1', '-q:v', '2', os.path.join(IMG, 'spot_%02d.jpg' % s['id'])], check=True)
json.dump(spots, open(os.path.join(OD, 'spots.json'), 'w'), indent=1)

print('%-3s %-5s %-5s %-60s %s' % ('id', 'frame', 'cell', 'flagged as', 'score uc_FRC / pl0 l4000 / l1000 / NVOFA medium'))
for s in spots:
    print('%-3d %-5d %-5s %-60s %s' % (s['id'], s['k'], s['cell'], '; '.join(s['why'])[:60],
                                      ' / '.join('%.2f' % s['values'][k]['score'] for k, _, _, _ in VARIANTS)))
