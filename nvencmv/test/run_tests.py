"""Tests of nvencmv (CudaNVENCAnalyse): engine sanity, determinism, token input, formats, every option, fallback.

python run_tests.py <bin folder with nvencmv.dll, mvtools2.dll, cudafilters.dll, mvtools_cuda.dll> <clip> [--only regex]

<clip>: any progressive video with detail that L-SMASH can open (the pictures for the synthetic shifts are cut from it),
e.g. seg1_wires_pole.mp4. Needs ffmpeg with AviSynth+ input, numpy, and LSMASHSource.dll (path in NVENCMV_LSMASH, else
LSMASHSource.dll in the bin folder). NVENC hardware (Pascal or newer) for everything but the fallback cases.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('bin')
ap.add_argument('clip')
ap.add_argument('--only', default='')
a = ap.parse_args()
BIN = os.path.abspath(a.bin).replace('\\', '/')
CLIP = os.path.abspath(a.clip).replace('\\', '/')
LS = os.environ.get('NVENCMV_LSMASH', BIN + '/LSMASHSource.dll').replace('\\', '/')
WORK = tempfile.mkdtemp(prefix='nvencmv_test_')

HEAD = ('LoadPlugin("%s")\nLoadPlugin("%s/mvtools2.dll")\nLoadPlugin("%s/cudafilters.dll")\nLoadPlugin("%s/nvencmv.dll")\n'
        'm = LWLibavVideoSource("%s")\n' % (LS, BIN, BIN, BIN, CLIP))
# two pictures, the second the first moved by (dx, dy) = (-4, -4) px; fmt: target format; chroma_only: flat luma
SHIFT = '''
t = m.Trim(40, 40).ConvertBits(8).ConvertToYV24().BilinearResize(1920, 1080)
p0 = t.Crop(16, 16, 1888, 1048)
p1 = t.Crop(12, 12, 1888, 1048)
'''


def run(body, env=None, raw=True):
    p = os.path.join(WORK, 'case.avs')
    open(p, 'w').write(HEAD + body + '\n')
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', p, '-f', 'rawvideo', '-'],
                       capture_output=True, env=e)
    err = r.stderr.decode(errors='replace')
    LAST_ERR[0] += err
    return r.returncode, r.stdout, err


def vectors(raw, frames):
    """(frames, blocks, 3) of a one-level vector clip"""
    w = np.frombuffer(raw, np.int32).reshape(frames, -1)
    words = w[0, 64]
    return w[:, 64 + 3:64 + words].reshape(frames, -1, 3), w[:, 64 + 1]


results = []


LAST_ERR = ['']


def case(name, ok, info=''):
    # a frame NVENC failed on gets MAnalyse's vectors silently (only a message): never a pass
    if 'NVENC failed' in LAST_ERR[0] and 'no GPU' not in name:
        ok, info = False, 'fallback used: ' + LAST_ERR[0].strip()[-160:]
    LAST_ERR[0] = ''
    results.append((name, ok, info))
    print('%-44s %s  %s' % (name, 'PASS' if ok else 'FAIL', info), flush=True)


def want(name):
    return not a.only or re.search(a.only, name)


def shift_share(args, chroma_only=False, fmt='YV12', expect=(-8, -8)):
    """share of blocks (pel 2) with the expected vector on frame 1 of the shifted pair"""
    if chroma_only:
        mk = ('y = BlankClip(p0, pixel_type="Y8", color_yuv=$808080)\n'
              'u0 = p0.ExtractY()\nu1 = p1.ExtractY()\nv = y\n'
              'f0 = CombinePlanes(y, u0, v, planes="YUV", source_planes="YYY", pixel_type="YV24")\n'
              'f1 = CombinePlanes(y, u1, v, planes="YUV", source_planes="YYY", pixel_type="YV24")\n')
    else:
        mk = 'f0 = p0\nf1 = p1\n'
    body = SHIFT + mk + 'c = (f0 + f1).ConvertTo%s()\nreturn CudaNVENCAnalyse(c, isb=false, blksize=16, overlap=8, pel=2%s).Trim(1, 1)' % (
        fmt, (', ' + args) if args else '')
    rc, out, err = run(body)
    if rc or not out:
        return None, err.strip()[-200:]
    v, _ = vectors(out, 1)
    return float(np.mean((v[0, :, 0] == expect[0]) & (v[0, :, 1] == expect[1]))), ''


def md5(body, env=None):
    rc, out, err = run(body, env)
    return (hashlib.md5(out).hexdigest()[:12] if rc == 0 and out else None), err.strip()[-200:]


# ---- availability and engine sanity
if want('available'):
    rc, out, err = run('return BlankClip(length=1, width=8, height=8, pixel_type="Y8", color_yuv=CudaNVENCAvailable() ? $FFFFFF : 0)')
    case('available', rc == 0 and out[:1] == b'\xff', err[-120:])
for name, args, co, lo in (('shift luma, y', 'channels="y"', False, 0.9), ('shift luma, defaults', '', False, 0.9),
                           ('shift chroma only, y (expect none)', 'channels="y"', True, None),
                           ('shift chroma only, yuv', 'channels="yuv"', True, 0.7),
                           ('shift chroma only, rgb', 'channels="rgb"', True, 0.7),
                           ('shift chroma only, yuv chroma=false', 'channels="yuv", chroma=false', True, 0.7),
                           ('shift luma, half', 'half=true', False, 0.9)):
    if not want(name):
        continue
    s, err = shift_share(args, co, 'YV24' if co else 'YV12')
    if s is None:
        case(name, False, err)
    elif lo is None:
        case(name, s < 0.05, 'right %.1f %% (NVENC ignores chroma)' % (100 * s))
    else:
        case(name, s >= lo, 'right %.1f %%' % (100 * s))
if want('hevc'):
    s, err = shift_share('codec="hevc"')
    case('hevc runs (known: zero vectors)', s is not None, err if s is None else 'right %.1f %%' % (100 * s))

# ---- determinism, session modes, token input
PAIR = 'c = m.Trim(0, 23)\nbv = CudaNVENCAnalyse(c, isb=true, blksize=32, overlap=16%s)\nfv = CudaNVENCAnalyse(c, isb=false, blksize=32, overlap=16%s)\nreturn Interleave(bv, fv).Prefetch(4)'
if want('determinism'):
    a1, e1 = md5(PAIR % ('', ''))
    a2, _ = md5(PAIR % ('', ''))
    case('determinism (defaults, Prefetch 4)', a1 is not None and a1 == a2, '%s %s %s' % (a1, a2, e1))
if want('session'):
    r, _ = md5(PAIR % (', session="reset"', ', session="reset"'))
    f, e = md5(PAIR % (', session="fresh"', ', session="fresh"'))
    case('session reset == fresh', r is not None and r == f, '%s %s %s' % (r, f, e))
if want('token'):
    h, e = md5(PAIR % ('', ''))
    t, e2 = md5(PAIR.replace('c = m.Trim(0, 23)', 'c = m.Trim(0, 23).CudaUpload()') % ('', ''))
    case('token input == host input', h is not None and h == t, '%s %s %s' % (h, t, e2))

# ---- formats and options (runs, plausible SADs, valid flags)
OPTS = [
    ('420p8', 'm.ConvertBits(8).ConvertToYV12()', ''),
    ('422p10', 'm.ConvertBits(10).ConvertToYUV422()', ''),
    ('444p8', 'm.ConvertBits(8).ConvertToYV24()', ''),
    ('y8', 'm.ConvertBits(8).ConvertToY()', ''),
    ('pel 1', 'm', 'pel=1'),
    ('delta 2', 'm', 'delta=2'),
    ('blksize 8/0 (raw cells)', 'm', 'blksize=8, overlap=0'),
    ('blksize 64/32', 'm', 'blksize=64, overlap=32'),
    ('blksize 16x8', 'm', 'blksize=16, blksizeV=8, overlap=4, overlapV=2'),
    ('qp -1, no swap, P1, no intra, p16', 'm', 'qp=-1, swap=false, preset=1, intra=false, p16=true'),
    ('P7, qp 51', 'm', 'preset=7, qp=51'),
    ('yuv + neighbours + lambda', 'm', 'channels="yuv", neighbours=true, lambda=250'),
    ('rgb + fbpenalty', 'm', 'channels="rgb", fbpenalty=1000'),
    ('session kept', 'm', 'session="kept"'),
    ('temporal hints', 'm', 'temporal=true'),
    ('hintclip (MAnalyse)', 'm', 'hintclip=MAnalyse(MSuper(c, pel=2), isb=true, blksize=16, overlap=8)'),
    ('hintclip + temporal', 'm', 'temporal=true, hintclip=MAnalyse(MSuper(c, pel=2), isb=true, blksize=16, overlap=8)'),
    ('hpad 0, vpad 0', 'm', 'hpad=0, vpad=0'),
]
for name, src, args in OPTS:
    if not want(name):
        continue
    body = 'c = %s.Trim(0, 5)\nreturn CudaNVENCAnalyse(c, isb=true, blksize=16, overlap=8%s)%s' % (src, (', ' + args) if args else '', '')
    if 'blksize' in args:
        body = 'c = %s.Trim(0, 5)\nreturn CudaNVENCAnalyse(c, isb=true%s)' % (src, ', ' + args)
    rc, out, err = run(body)
    if rc or not out:
        case(name, False, err.strip()[-200:])
        continue
    v, valid = vectors(out, 6)
    nref = 4 if 'delta=2' in args else 5  # frames that have a reference (isb=true)
    ok = all(valid[nref:] == 0) and all(valid[:nref] == 1) and np.all(v[:nref, :, 2] >= 0) and np.abs(v[:nref, :, :2]).max() < 4000
    case(name, bool(ok), 'valid %s, |v| max %d, SAD mean %.0f' % ([int(x) for x in valid], np.abs(v[:nref, :, :2]).max(),
                                                                   v[:nref, :, 2].mean()))
if want('stats'):
    sp = os.path.join(WORK, 'stats.txt').replace('\\', '/')
    rc, out, err = run('c = m.Trim(0, 5)\nreturn CudaNVENCAnalyse(c, channels="yuv", neighbours=true, stats="%s")' % sp)
    line = open(sp).read().split() if os.path.exists(sp) else []
    cnt = sum(int(x) for x in line[5:] if '=' not in x) if len(line) > 5 else 0
    tail = [x for x in line if x.startswith('frames=') or x.startswith('manalyse_frames=')]
    case('stats file', rc == 0 and cnt > 0 and tail == ['frames=5', 'manalyse_frames=0'], '%d cells counted, %s' % (cnt, ' '.join(tail)))

# ---- into MVTools: MRecalculate, MFlowFps, MDegrain2, MCompensate accept the clip
if want('chain'):
    body = ('c = m.Trim(0, 11)\nsup = MSuper(c, pel=2, levels=1)\n'
            'bv = MRecalculate(sup, CudaNVENCAnalyse(c, isb=true, blksize=16, overlap=8), blksize=8, overlap=4)\n'
            'fv = MRecalculate(sup, CudaNVENCAnalyse(c, isb=false, blksize=16, overlap=8), blksize=8, overlap=4)\n'
            'b2 = CudaNVENCAnalyse(c, isb=true, delta=2, blksize=16, overlap=8)\nf2 = CudaNVENCAnalyse(c, isb=false, delta=2, blksize=16, overlap=8)\n'
            'b1 = CudaNVENCAnalyse(c, isb=true, blksize=16, overlap=8)\nf1 = CudaNVENCAnalyse(c, isb=false, blksize=16, overlap=8)\n'
            'd = c.MDegrain2(sup, b1, f1, b2, f2)\nk = c.MCompensate(sup, b1)\nfl = c.MFlowFps(sup, bv, fv, num=50, den=1)\n')
    for name, last in (('chain: MRecalculate + MFlowFps', 'fl'), ('chain: MDegrain2 (delta 1 + 2)', 'd'), ('chain: MCompensate', 'k')):
        rc, out, err = run(body + 'return %s.Prefetch(4)' % last)
        case(name, rc == 0 and len(out) > 0, err.strip()[-200:])

# ---- fallback
if want('fallback'):
    body = 'c = m.Trim(0, 5)\nreturn CudaNVENCAnalyse(c, isb=true, blksize=16, overlap=8)'
    rc, out, err = run(body, {'MVTOOLS_GPU_DISABLE': '1'})
    case('no GPU, no fallback: error', rc != 0 or 'cuda_cpu_fallback' in err, err.strip()[-160:])
    fb, e1 = md5('global cuda_cpu_fallback = true\n' + body, {'MVTOOLS_GPU_DISABLE': '1'})
    ma, e2 = md5('c = m.Trim(0, 5)\nreturn MAnalyse(MSuper(c, pel=2, hpad=8, vpad=8, chroma=true), isb=true, delta=1, blksize=16, '
                 'blksizeV=16, overlap=8, overlapV=8, truemotion=true, chroma=true, scaleCSAD=0)', {'MVTOOLS_GPU_DISABLE': '1'})
    case('no GPU, cuda_cpu_fallback: = MAnalyse', fb is not None and fb == ma, '%s %s %s' % (fb, ma, e1 or e2))
    rc, out, err = run('global cuda_cpu_fallback = true\n' + body, {'MVTOOLS_GPU_DISABLE': '1'})
    case('no GPU, cuda_cpu_fallback: logged', 'MAnalyse on the CPU for the whole clip' in err, err.strip()[-160:])
    # per-frame fallback: every frame (5 with a reference) reported, summary at the end
    rc, out, err = run(body, {'MVTOOLS_GPU_FORCE_FALLBACK': '1'})
    LAST_ERR[0] = ''
    lines = err.count("MAnalyse's vectors for this frame")
    # every request counts (the host may ask for a frame twice): at least the 5 frames, and the summary agrees
    case('no GPU per frame: every frame logged + summary', rc == 0 and lines >= 5 and '%d of %d frames used MAnalyse' % (lines, lines) in err,
         '%d lines; %s' % (lines, err.strip().splitlines()[-1][:150] if err.strip() else ''))

bad = [r for r in results if not r[1]]
print('\n%d/%d passed' % (len(results) - len(bad), len(results)))
sys.exit(1 if bad else 0)
