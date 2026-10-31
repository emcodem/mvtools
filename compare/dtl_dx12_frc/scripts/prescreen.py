"""Pre-screen for the visual check of the DTL DX12_ME comparison (adapted from compare/nvencme_frc/scripts/prescreen.py):
scores every variant from its MRecalculate vectors, prints a summary, picks spots and cuts 2x2 crops.

<python with numpy> prescreen.py <work folder> [--panels dtl5_8,dx8] [--top 5] [--metrics-only] [--keys a,b,..]

The vector clip is read as raw frames through ffmpeg (both DLLs format it the same way, ClipFnc::format_vector_clip);
the header of every frame is checked, not assumed: headerSize 256 bytes, nBlkX x nBlkY 479 x 269 (16/8 at UHD),
pel 2 (MRecalculate's super), data size nBlkX * nBlkY * 3 + 1 words.

Score per interpolated output frame k and 120 px cell (columns A..AF, rows 1..18), as in nvofa_frc: share of blocks
whose vector differs from the median of its 8 neighbours by > 3 px (bv n and fv n + 1), plus share of blocks whose change
since the previous source frame differs from its neighbours' by > 4 px; mean SAD.
Spots: the worst cells of each panel variant, and its regressions against NVOFA medium (score at least 0.25 higher
and SAD at least 1.2x). Crops: exact | NVOFA medium on top, the two panels below.
Output: <work>/prescreen/spots.json, ../img/spot_<nn>.jpg, ../spots_check.csv (semicolon, UTF-8 BOM).
"""
import argparse
import json
import os
import subprocess
import threading

import numpy as np

import variants as V

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--panels', default='dtl5_8,dx8')
ap.add_argument('--top', type=int, default=5)
ap.add_argument('--metrics-only', action='store_true')
ap.add_argument('--keys', default=','.join(v[0] for v in V.ALL))
a = ap.parse_args()
KEYS = a.keys.split(',')

IMG = os.path.join(V.HERE, '..', 'img')
OD = V.run_dir(os.path.join(os.path.abspath(a.work), 'prescreen'))
os.makedirs(IMG, exist_ok=True)
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
    p = np.pad(f, 1, mode='edge')
    sh = [p[1 + j:1 + j + NBY, 1 + i:1 + i + NBX] for j in (-1, 0, 1) for i in (-1, 0, 1) if i or j]
    return np.median(np.stack(sh), axis=0)


def outliers(vx, vy, thr_px):
    return np.hypot(vx - nb_median(vx), vy - nb_median(vy)) > thr_px * PEL


def per_cell(v):
    return np.bincount(cell_index, weights=v.ravel().astype(np.float64), minlength=NCX * NCY) / cell_count


def output_frames():
    res = {}
    for k in range(OUT_FRAMES):
        pos4 = 5 * (k + 8)
        if pos4 % 4:
            res[k] = pos4 // 4
    return res


ks = output_frames()
need = {}
for k, n in ks.items():
    need.setdefault(n, []).append(k)

def check_header(w, key):
    """the layout this reader assumes, checked on every vector frame (MVAnalysisData field order is the same in both
    sources: headerSize, nMagicKey, nVersion, nBlkSizeX, nBlkSizeY, nPel, nLvCount, nDeltaFrame, isBackward, nFlags,
    nWidth, nHeight, nOverlapX, nOverlapY, nBlkX, nBlkY)"""
    got = (w[0], w[3], w[4], w[5], w[12], w[13], w[14], w[15], w[HDR + 2])
    want = (HDR * 4, BLK, BLK, PEL, BLK - STEP, BLK - STEP, NBX, NBY, NBX * NBY * 3 + 1)
    if got != want:
        raise SystemExit('%s: vector clip layout %s, expected %s' % (key, got, want))


metrics = {}
for key in KEYS:
    cache = os.path.join(OD, 'metrics_%s.npz' % key)
    if os.path.exists(cache):
        metrics[key] = dict(np.load(cache))
        continue
    lines = [l for l in V.script_text(key).splitlines() if not l.startswith('out =') and not l.startswith('return')]
    avs = V.write_script(OD, 'vec_' + key, '\n'.join(lines) + '\nreturn Interleave(bv, fv)\n')
    pr = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', avs, '-f', 'rawvideo', '-'],
                          stdout=subprocess.PIPE, env=V.env_for(key), cwd=OD)
    watchdog = threading.Timer(V.TIMEOUT, pr.kill)  # a stalled GPU path must not hang the pre-screen
    watchdog.start()
    score = np.full((OUT_FRAMES, NCY, NCX), np.nan)
    outl, jump, sad = score.copy(), score.copy(), score.copy()
    hist = {}
    n = 0
    last_n = max(need) + 1
    while n <= last_n:
        pair = []
        for _ in range(2):
            b = pr.stdout.read(WORDS * 4)
            if len(b) < WORDS * 4:
                break
            w = np.frombuffer(b, dtype=np.int32)
            check_header(w, key)
            pair.append(w[HDR + 3:].reshape(NBY, NBX, 3).transpose(2, 0, 1).astype(np.int64))
        if len(pair) < 2:
            break
        hist[n] = pair
        hist.pop(n - 3, None)
        m = n - 1
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
    watchdog.cancel()
    if pr.returncode != 0 or np.isnan(score).all():
        raise SystemExit('vector dump of %s failed' % key)
    metrics[key] = {'score': score, 'outl': outl, 'jump': jump, 'sad': sad}
    np.savez(cache, **metrics[key])
    print('scanned %s' % key, flush=True)

# summary: mean over all interpolated frames, and over the tree section 1.7-2.6 s (output frames 41..62)
valid = sorted(ks)
tree = [k for k in valid if 41 <= k + 0 <= 62]
print('\n%-26s %10s %10s %10s %12s' % ('variant', 'score all', 'score tree', 'SAD all', 'cells > 0.5'))
for key in KEYS:
    label = V.BY_KEY[key][1]
    m = metrics[key]
    print('%-26s %10.4f %10.4f %10.0f %12d' % (label, np.nanmean(m['score'][valid]), np.nanmean(m['score'][tree]),
                                              np.nanmean(m['sad'][valid]), int(np.nansum(m['score'][valid] > 0.5))))
if a.metrics_only:
    raise SystemExit(0)

panels = a.panels.split(',')
assert len(panels) == 2 and all(p in metrics and p not in ('exact', 'medium') for p in panels)
SHOW = ['exact', 'medium'] + panels


def pick(score, filt=None, top=6):
    s = np.nan_to_num(score.copy(), nan=-1)
    if filt is not None:
        s[~filt] = -1
    out = []
    for idx in np.argsort(s, axis=None)[::-1]:
        if len(out) >= top or s.flat[idx] <= 0:
            break
        k, cy, cx = np.unravel_index(idx, s.shape)
        if any(abs(k - k2) < 3 and abs(cy - y2) <= 2 and abs(cx - x2) <= 2 for k2, y2, x2 in out):
            continue
        out.append((int(k), int(cy), int(cx)))
    return out


spots = []
med = metrics['medium']
for key in panels:
    label = V.BY_KEY[key][1]
    for k, cy, cx in pick(metrics[key]['score'], top=a.top):
        spots.append(((k, cy, cx), 'worst of ' + label, key, 'worst'))
    m = metrics[key]
    filt = (m['score'] - med['score'] > 0.25) & (m['sad'] >= 1.2 * med['sad'])
    for k, cy, cx in pick(m['score'] - med['score'], filt, top=a.top):
        spots.append(((k, cy, cx), 'regression of %s vs NVOFA medium' % label, key, 'regr'))
merged = {}
for pos, why, key, kind in spots:
    e = merged.setdefault(pos, {'why': [], 'keys': [], 'kinds': []})
    e['why'].append(why)
    e['keys'].append(key)
    e['kinds'].append(kind)
spots = []
for i, ((k, cy, cx), e) in enumerate(sorted(merged.items())):
    spots.append({'id': i + 1, 'k': k, 'cy': cy, 'cx': cx, 'cell': cell_name(cx, cy), 'why': e['why'], 'keys': e['keys'], 'kinds': e['kinds'],
                  'values': {key: {'score': round(float(metrics[key]['score'][k, cy, cx]), 2),
                                   'sad': int(metrics[key]['sad'][k, cy, cx])} for key in SHOW}})

frames = sorted({s['k'] for s in spots})
for key in SHOW:
    avs = V.write_script(OD, key)
    sel = '+'.join('eq(n,%d)' % k for k in frames)
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error', '-i', avs, '-vf', "select='%s'" % sel,
                    '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', os.path.join(OD, key + '_%03d.png')], check=True, env=V.env_for(key), cwd=OD, timeout=V.TIMEOUT)
    for i, k in enumerate(frames):
        src, dst = os.path.join(OD, key + '_%03d.png' % (i + 1)), os.path.join(OD, '%s_k%03d.png' % (key, k))
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)

FONT = 'C\\:/Windows/Fonts/consola.ttf'
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
    fx, fy = s['cx'] * CELL - x0, s['cy'] * CELL - y0 + HEAD
    chains, ins = [], []
    for i, key in enumerate(SHOW):
        ins += ['-i', os.path.join(OD, '%s_k%03d.png' % (key, s['k']))]
        v = s['values'][key]
        info = 'score %.2f  SAD %d' % (v['score'], v['sad'])
        chains.append("[%d:v]crop=960:540:%d:%d,pad=960:%d:0:%d:color=black,drawgrid=x=0:y=%d:w=%d:h=%d:t=1:c=yellow@0.6,%s,"
                      "drawbox=x=%d:y=%d:w=%d:h=%d:color=0xff8000:t=3,drawbox=x=0:y=0:w=960:h=%d:color=black:t=fill,"
                      "drawtext=fontfile='%s':text='%s   frame %d   cell %s':x=10:y=4:fontsize=20:fontcolor=white,"
                      "drawtext=fontfile='%s':text='%s':x=10:y=27:fontsize=17:fontcolor=yellow[p%d]"
                      % (i, x0, y0, 540 + HEAD, HEAD, HEAD, CELL, CELL, ','.join(lab), fx, fy, CELL, CELL, HEAD,
                         FONT, V.BY_KEY[key][1], s['k'], s['cell'], FONT, info, i))
    graph = ';'.join(chains) + ';[p0][p1]hstack=2[r0];[p2][p3]hstack=2[r1];[r0][r1]vstack=2[out]'
    gf = os.path.join(OD, 'graph.txt')
    open(gf, 'w').write(graph)
    s['image'] = 'img/spot_%02d.jpg' % s['id']
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error'] + ins +
                   ['-/filter_complex', gf, '-map', '[out]', '-frames:v', '1', '-q:v', '2', os.path.join(IMG, 'spot_%02d.jpg' % s['id'])], check=True)
json.dump(spots, open(os.path.join(OD, 'spots.json'), 'w'), indent=1)

# CSV for the judgement: one question per spot, priority 1 = regressions against NVOFA medium, 2 = worst cells
L = {k: V.BY_KEY[k][1] for k in SHOW}
rows = []
for s in spots:
    keys = sorted(set(s['keys']), key=panels.index)
    names = ' and '.join(L[k] for k in keys)
    if 'regr' in s['kinds']:
        prio = 1
        q = ('In and around %s: is %s (bottom row) visibly worse than NVOFA medium (top right)? '
             'Which panels show defects (blocks, wrong colour, odd smearing)?' % (s['cell'], names))
    else:
        prio = 2
        q = ('In and around %s: does %s (bottom row) show a visible defect? Are exact (top left) or NVOFA medium '
             '(top right) better there?' % (s['cell'], names))
    rows.append((prio, s, q))
rows.sort(key=lambda r: (r[0], r[1]['k']))
hdr = (['id', 'priority', 'image', 'frame', 'cell', 'flagged as', 'question'] + ['score %s' % L[k] for k in SHOW]
       + ['defect %s (y/n)' % L[k] for k in SHOW] + ['notes'])
with open(os.path.join(V.HERE, '..', 'spots_check.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
    fh.write(';'.join(hdr) + '\r\n')
    for prio, s, q in rows:
        why = '; '.join(s['why'])
        cells = [str(s['id']), str(prio), s['image'], str(s['k']), s['cell'], '"%s"' % why if ';' in why else why, q]
        cells += ['%.2f' % s['values'][k]['score'] for k in SHOW] + [''] * len(SHOW) + ['']
        fh.write(';'.join(cells) + '\r\n')

print('\n%-3s %-5s %-5s %-60s %s' % ('id', 'frame', 'cell', 'flagged as', ' / '.join(SHOW)))
for s in spots:
    print('%-3d %-5d %-5s %-60s %s' % (s['id'], s['k'], s['cell'], '; '.join(s['why'])[:60],
                                      ' / '.join('%.2f' % s['values'][k]['score'] for k in SHOW)))
