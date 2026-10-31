"""Renders the comparison clips and the speed table of the NVENC ME variants.

python render.py <work folder> [--keys exact,medium,p4,p7h] [--runs 3]

Per variant: <work>/<key>_crop.mp4 (the 960x540 crop at x 2880, y 240 over the whole run, input of slowmo.py) and the
md5 of the whole output (ffmpeg -f md5). Speed: wall time of `ffmpeg -f null` over the 112 output frames (Prefetch 3),
best of --runs; the GPU may be shared with other jobs, so the numbers are indicative.
"""
import argparse
import os
import subprocess
import time

import variants as V

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--keys', default='exact,medium,p1,p4,p7,p7h')
ap.add_argument('--crop-keys', default='exact,medium,p4,p7h')
ap.add_argument('--runs', type=int, default=3)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
C = 'crop=960:540:2880:240'
FF = ['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error']

rows = []
for key in a.keys.split(','):
    avs = V.write_script(a.work, key)
    env = V.env_for(key)
    best = 1e9
    for _ in range(a.runs):
        t0 = time.perf_counter()
        subprocess.run(FF + ['-i', avs, '-f', 'null', '-'], check=True, env=env)
        best = min(best, time.perf_counter() - t0)
    md5 = subprocess.run(FF + ['-i', avs, '-f', 'md5', '-'], check=True, env=env, capture_output=True, text=True).stdout.strip()
    rows.append((V.BY_KEY[key][1], best, md5.split('=')[-1][:12].upper()))
    if key in a.crop_keys.split(','):
        subprocess.run(FF + ['-i', avs, '-vf', C + ',format=yuv420p', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow',
                             os.path.join(a.work, key + '_crop.mp4')], check=True, env=env)
    print('%-26s %6.2f s  %s' % rows[-1], flush=True)

print('\n| variant | wall (best of %d) | md5 |\n|---|---:|---|' % a.runs)
for r in rows:
    print('| %s | %.2f s | %s |' % r)
