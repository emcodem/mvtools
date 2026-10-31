"""Why the exact search misses the pan at output frame 41 (source 61/62): MAnalyse with one parameter changed at a time
against the uc_FRC settings, level-0 vectors compared with NVOFA medium's (the reference for the true motion here).

python -I why_pan.py <work folder> [name filter]   (needs <work>/vec_analyse_medium.raw from render_vec.ps1)

Per variant, bv 61 -> 62 and fv 62 -> 61: 'match' = share of blocks within 16 px of NVOFA medium's vector of the same
block, the median vector and the mean SAD (MAnalyse's own SAD, luma + chroma), in the crop and over the whole frame.
"""
import array
import os
import subprocess
import sys

work = sys.argv[1]
HDR = 64
X0, Y0, CW, CH = 2880, 240, 960, 540
NBX, NBY = 239, 134
BASE = 'blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true'
VARIANTS = [
    ('base (uc_FRC)', ''),
    ('truemotion=false', 'truemotion=false'),
    ('lambda=0', 'lambda=0'),
    ('pnew=0', 'pnew=0'),
    ('plevel=0', 'plevel=0'),
    ('pzero=0', 'pzero=0'),
    ('global=false', 'global=false'),
    ('lsad=0', 'lsad=0'),
    ('searchparam=4', 'searchparam=4'),
    ('searchparam=8', 'searchparam=8'),
    ('levels=-1', 'levels=-1'),
    ('levels=-2', 'levels=-2'),
    ('truemotion=false, searchparam=4', 'truemotion=false, searchparam=4'),
    # compromise values (uc_FRC's lambda is truemotion's default 1000 * 32 * 32 / 64 = 16000)
    ('lambda=8000', 'lambda=8000'),
    ('lambda=4000', 'lambda=4000'),
    ('lambda=2000', 'lambda=2000'),
    ('lambda=1000', 'lambda=1000'),
    ('plevel=0, lambda=8000', 'plevel=0, lambda=8000'),
    ('plevel=0, lambda=4000', 'plevel=0, lambda=4000'),
]

AVS = r'''LoadPlugin("C:\dev\uconvert\dlls\LSMASHSource.dll")
LoadPlugin("C:\dev\uconvert\dlls\RgTools.dll")
LoadPlugin("C:\dev\mvtools\profile\build_nvofa\mvtools2.dll")
SEGFILE = "C:\dev\mvtools\profile\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, threads=4, cachefile=SEGFILE + ".lwi")
supf = MSuper(m.RemoveGrain(22), pel=2, hpad=16, vpad=16, rfilter=4, gpu=true)
ba = MAnalyse(supf, isb=true, %s, gpu=true)
fa = MAnalyse(supf, isb=false, %s, gpu=true)
return Interleave(ba.Trim(61, -1), fa.Trim(62, -1))
'''


def level0(path, frames):
    b = open(path, 'rb').read()
    fs = len(b) // frames
    out = []
    for f in range(frames):
        a = array.array('i', b[f * fs:(f + 1) * fs])
        p, last = HDR + 2, None
        while p < len(a) and a[p] > 0:
            last = p
            p += a[p]
        assert a[last] == NBX * NBY * 3 + 1
        out.append(a[last + 1:last + 1 + NBX * NBY * 3])
    return out


def in_crop(i):
    x, y = (i % NBX) * 16 + 16, (i // NBX) * 16 + 16
    return X0 <= x < X0 + CW and Y0 <= y < Y0 + CH


ref = level0(os.path.join(work, 'vec_analyse_medium.raw'), 4)[:2]   # bv 61, fv 62
crop = [i for i in range(NBX * NBY) if in_crop(i)]
every = list(range(NBX * NBY))


def stats(q, r, sel):
    ok = sum(1 for i in sel if (q[3 * i] - r[3 * i]) ** 2 + (q[3 * i + 1] - r[3 * i + 1]) ** 2 <= (16 * 2) ** 2)
    xs = sorted(q[3 * i] / 2 for i in sel)
    ys = sorted(q[3 * i + 1] / 2 for i in sel)
    n = len(sel)
    return 100.0 * ok / n, xs[n // 2], ys[n // 2], sum(q[3 * i + 2] for i in sel) / n


only = sys.argv[2] if len(sys.argv) > 2 else ''   # optional: run only the variants whose name contains this text
print('%-34s %-3s | %-34s | %-22s' % ('variant', 'dir', 'crop: match  median vector   SAD', 'frame: match    SAD'))
for name, extra in VARIANTS:
    if only and only not in name:
        continue
    params = BASE + (', ' + extra if extra else '')
    # later keywords win in AviSynth? No: a repeated named argument is an error, so replace instead of appending
    p = dict(kv.split('=') for kv in BASE.replace(' ', '').split(','))
    for kv in extra.replace(' ', '').split(','):
        if kv:
            k, v = kv.split('=')
            p[k] = v
    params = ', '.join('%s=%s' % kv for kv in p.items())
    avs = os.path.join(work, 'why_pan.avs')
    open(avs, 'w').write(AVS % (params, params))
    raw = os.path.join(work, 'why_pan.raw')
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error', '-i', avs, '-f', 'rawvideo', raw], check=True)
    got = level0(raw, 2)
    for k, d in enumerate(('bv', 'fv')):
        c = stats(got[k], ref[k], crop)
        f = stats(got[k], ref[k], every)
        print('%-34s %-3s | %5.1f%%  (%5.0f, %5.0f)  %6.0f | %5.1f%%  %6.0f' % (name if k == 0 else '', d, c[0], c[1], c[2], c[3], f[0], f[3]))
    sys.stdout.flush()
