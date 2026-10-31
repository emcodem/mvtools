"""Renders the comparison clips and the speed table of the DTL DX12_ME comparison.

python render.py <work folder> [--keys exact,medium,...] [--crop-keys ...] [--runs 3]

Per variant: md5 of the whole output (ffmpeg -f md5, twice: a second md5 that differs means the variant is not
deterministic), <work>/<key>_crop.mp4 (the 960x540 crop at x 2880, y 240 over the whole run, input of slowmo.py), and
the wall time of `ffmpeg -f null` over the 112 output frames (Prefetch 3), best of --runs (indicative: the GPU may be
shared). Every ffmpeg run has a timeout (variants.TIMEOUT); a variant that fails or times out is reported, not retried.
"""
import argparse
import os
import subprocess
import time

import variants as V

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--keys', default=','.join(v[0] for v in V.ALL))
ap.add_argument('--crop-keys', default='exact,medium,dx8,dtl5_8,dtl5_16,dtl6_8,dtl6_16,dtl_cpu')
ap.add_argument('--runs', type=int, default=3)
a = ap.parse_args()
work = V.run_dir(os.path.abspath(a.work))
C = 'crop=960:540:2880:240'
FF = ['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error', '-xerror']


def ff(args, env):
    """one ffmpeg run; an AviSynth error mid-stream can still end with exit code 0, so any message on stderr fails"""
    r = subprocess.run(FF + args, env=env, cwd=work, timeout=V.TIMEOUT, capture_output=True, text=True)
    if r.returncode or r.stderr.strip():
        raise subprocess.CalledProcessError(r.returncode or 1, 'ffmpeg', r.stdout, r.stderr)
    return r


rows = []
for key in a.keys.split(','):
    avs = V.write_script(work, key)
    env = V.env_for(key)
    try:
        frames = sum(1 for l in ff(['-i', avs, '-f', 'framecrc', '-'], env).stdout.splitlines() if l and l[0] != '#')
        if frames != 112:
            raise subprocess.CalledProcessError(1, 'ffmpeg', '', '%d frames instead of 112' % frames)
        md5 = [ff(['-i', avs, '-f', 'md5', '-'], env).stdout.strip().split('=')[-1][:12].upper()
               for _ in range(2)]
        best = 1e9
        for _ in range(a.runs):
            t0 = time.perf_counter()
            ff(['-i', avs, '-f', 'null', '-'], env)
            best = min(best, time.perf_counter() - t0)
        if key in a.crop_keys.split(','):
            ff(['-i', avs, '-vf', C + ',format=yuv420p', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow',
                os.path.join(work, key + '_crop.mp4')], env)
        rows.append((V.BY_KEY[key][1], '%.2f s' % best, md5[0] if md5[0] == md5[1] else '%s / %s (differs)' % tuple(md5)))
    except subprocess.TimeoutExpired:
        rows.append((V.BY_KEY[key][1], 'timeout', ''))
    except subprocess.CalledProcessError as e:
        rows.append((V.BY_KEY[key][1], 'failed (%d)' % e.returncode, (e.stderr or '').strip()[:80] if isinstance(e.stderr, str) else ''))
    print('%-26s %10s  %s' % rows[-1], flush=True)

print('\n| variant | wall (best of %d) | md5 |\n|---|---:|---|' % a.runs)
for r in rows:
    print('| %s | %s | %s |' % r)
