"""Skip-frame check: vectors on the even frames, MFlowInter time=50 predicts each odd frame, compared with the real one.
Independent of how coherent the field looks; colour planes included. 8-bit 4:2:2 UHD.

python skipcheck.py <work> --keys exact,medium,...
Per variant: Y PSNR (mean over frames), U+V mean abs error, and per 16x16 block (Y + U + V mean abs error): p99, max,
blocks above 12 / 24 (all frames), the 5 worst frames' worst block.
"""
import argparse
import json
import os
import subprocess

import numpy as np

import variants as V

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--keys', default=','.join(v[0] for v in V.ALL))
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
W, H = 3840, 2160
YS, CS = W * H, (W // 2) * H
FS = YS + 2 * CS
B = 16


def block_err(it, tr):
    y = np.abs(it[:YS].astype(np.int16) - tr[:YS]).reshape(H // B, B, W // B, B).mean(axis=(1, 3))
    u = np.abs(it[YS:YS + CS].astype(np.int16) - tr[YS:YS + CS]).reshape(H // B, B, W // B, B // 2).mean(axis=(1, 3))
    v = np.abs(it[YS + CS:].astype(np.int16) - tr[YS + CS:]).reshape(H // B, B, W // B, B // 2).mean(axis=(1, 3))
    return y, u + v


res = {}
for key in a.keys.split(','):
    cache = os.path.join(a.work, 'skip_%s.json' % key)
    if os.path.exists(cache):
        res[key] = json.load(open(cache))
        continue
    avs = V.write_script(a.work, 'skip_' + key, V.script_text(key, 'skip'))
    env = V.env_for(key)
    env['NVME_STATS'] = os.path.join(a.work, 'stats_skip_%s.txt' % key)
    pr = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', avs, '-pix_fmt', 'yuv422p',
                           '-f', 'rawvideo', '-'], stdout=subprocess.PIPE, env=env)
    psnr, cerr, blocks, worst = [], [], [], []
    f = 0
    while True:
        it = pr.stdout.read(FS)
        tr = pr.stdout.read(FS)
        if len(tr) < FS:
            break
        it = np.frombuffer(it, np.uint8)
        tr = np.frombuffer(tr, np.uint8)
        d = it[:YS].astype(np.float64) - tr[:YS]
        psnr.append(10 * np.log10(255 * 255 / max(np.mean(d * d), 1e-9)))
        yb, cb = block_err(it, tr)
        cerr.append(float(cb.mean()))
        tot = yb + cb
        blocks.append(tot.ravel().astype(np.float32))
        worst.append((float(tot.max()), f))
        f += 1
    pr.stdout.close()
    pr.wait()
    if pr.returncode != 0 or not psnr:
        raise SystemExit('skip render of %s failed' % key)
    allb = np.concatenate(blocks)
    res[key] = {'frames': f, 'psnr_y': float(np.mean(psnr)), 'uv_err': float(np.mean(cerr)),
                'p99': float(np.percentile(allb, 99)), 'p999': float(np.percentile(allb, 99.9)), 'max': float(allb.max()),
                'gt12': int((allb > 12).sum()), 'gt24': int((allb > 24).sum()),
                'worst5': sorted(worst, reverse=True)[:5], 'psnr_frames': psnr}
    json.dump(res[key], open(cache, 'w'))
    print('done', key, flush=True)

print('\n| variant | Y PSNR | U+V err | block p99 | p99.9 | max | blocks > 12 | > 24 | worst frames |\n|---|---:|---:|---:|---:|---:|---:|---:|---|')
for key in a.keys.split(','):
    r = res[key]
    print('| %s | %.2f | %.3f | %.2f | %.2f | %.1f | %d | %d | %s |' % (
        V.BY_KEY[key][1], r['psnr_y'], r['uv_err'], r['p99'], r['p999'], r['max'], r['gt12'], r['gt24'],
        ', '.join('%d (%.0f)' % (fr, e) for e, fr in r['worst5'])))
