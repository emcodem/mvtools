"""Code differences between two renders of the same clip: two filter calls, each through its own plugin set.

  python -I fastcmp.py <src clip> [--a CALL] [--b CALL] [--dlls-a DIR] [--dlls-b DIR] [--frames N] [--rate num/den]
                       [--per-frame]

CALL is an AviSynth+ expression on the source clip c with the output rate num, den (default --a
'CudaSplatFps(c, num, den, exact=false)', --b 'SplatFps(c, num, den, exact=true)'); DIR holds splatfps.dll (and
mvtools_cuda.dll); both default to the directory in the environment variable SPLATFPS_DLLS. Both renders run
sequentially from frame 0 (state carried across frames is exercised) and are compared frame by frame in the output
format, 10-bit 4:2:2 codes: per frame and plane (Y, U, V) the number of codes that differ, the largest difference and
a histogram of the differences (1, 2, 3-4, 5-8, 9-16, > 16). Summary: totals, the worst frame, and how many frames stay
within the fast-variant bounds (at most 500 luma and 500 chroma codes differing at 1920 x 1080, scaled with the
pixel count, none by more than 2). Only output codes are compared here, so flips at rounding ties count too (the stage
checks leave them out); on content with little motion the count can exceed the bound while every difference is 1.
--frames: the first N output frames (default 48). Exit code 1 when a frame is outside the bounds.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avs_util as AU  # noqa: E402
import stage_inputs as SI  # noqa: E402

BINS = (1, 2, 3, 5, 9, 17)  # histogram edges: 1, 2, 3-4, 5-8, 9-16, > 16
LABELS = ('1', '2', '3-4', '5-8', '9-16', '>16')


def start(script):
    """ffmpeg rendering the script as raw 10-bit 4:2:2 to a pipe; returns (process, temporary script path)"""
    f = tempfile.NamedTemporaryFile('w', suffix='.avs', delete=False)
    f.write(script)
    f.close()
    p = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-i', f.name, '-f', 'rawvideo', '-pix_fmt',
                          'yuv422p10le', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         creationflags=AU.NOWIN)
    return p, f.name


def read_frame(p, n):
    b = p.stdout.read(n)
    return b if len(b) == n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--a', default='CudaSplatFps(c, num, den, exact=false)')
    ap.add_argument('--b', default='SplatFps(c, num, den, exact=true)')
    ap.add_argument('--dlls-a', default=os.environ.get('SPLATFPS_DLLS'))
    ap.add_argument('--dlls-b', default=os.environ.get('SPLATFPS_DLLS'))
    ap.add_argument('--frames', type=int, default=48)
    ap.add_argument('--rate', default='60000/1001')
    ap.add_argument('--per-frame', action='store_true')
    args = ap.parse_args()
    if not args.dlls_a or not args.dlls_b:
        sys.exit('give --dlls-a / --dlls-b (directories with splatfps.dll) or set SPLATFPS_DLLS')
    _, _, w, h, pix = AU.probe(args.src)
    if pix != 'yuv422p10le':
        sys.exit(f'{args.src}: a 10-bit 4:2:2 source is needed, not {pix}')
    r = Fraction(args.rate)
    procs = []
    for call, ddir in ((args.a, args.dlls_a), (args.b, args.dlls_b)):
        script = (AU.source_script(args.src, [os.path.join(ddir, 'splatfps.dll')], var='c')
                  + f'num = {r.numerator}\nden = {r.denominator}\n{call}.Trim(0, {args.frames - 1})\n')
        procs.append(start(script))
    ny, nc = w * h, 2 * (w // 2) * h
    size = 2 * (ny + nc)
    lim, MAX_DIFF = SI.max_codes(w, h), SI.MAX_DIFF
    tot = np.zeros((3, len(BINS)), np.int64)
    worst, n_in, k = (0, -1), 0, 0
    try:
        while True:
            fa, fb = read_frame(procs[0][0], size), read_frame(procs[1][0], size)
            if fa is None or fb is None:
                break
            a, b = np.frombuffer(fa, np.uint16).astype(np.int32), np.frombuffer(fb, np.uint16).astype(np.int32)
            d = np.abs(a - b)
            planes = (d[:ny], d[ny:ny + nc // 2], d[ny + nc // 2:])
            hist = np.stack([np.histogram(p, bins=BINS + (1 << 16,))[0] for p in planes])
            tot += hist
            cnt = [int(np.count_nonzero(p)) for p in planes]
            mx = [int(p.max()) for p in planes]
            ok = cnt[0] <= lim and cnt[1] + cnt[2] <= lim and max(mx) <= MAX_DIFF
            n_in += ok
            worst = max(worst, (sum(cnt), k))
            if args.per_frame or not ok:
                print(f'frame {k}: ' + ', '.join(f'{nm} {c} (max {m})' for nm, c, m in zip('YUV', cnt, mx))
                      + ('' if ok else '  OUTSIDE BOUNDS'), flush=True)
            k += 1
    finally:
        errs = []
        for p, path in procs:
            p.stdout.close()
            errs.append(p.stderr.read().decode(errors='replace').strip())
            p.wait()
            os.unlink(path)
    if k == 0:
        sys.exit('no frames: ' + ' | '.join(e[-800:] for e in errs))
    print(f'{k} frames compared ({args.a} vs {args.b})')
    for nm, row in zip('YUV', tot):
        print(f'  {nm}: {int(row.sum())} codes differ; by ' + ', '.join(f'{lb}: {int(v)}' for lb, v in zip(LABELS, row)))
    print(f'  worst frame {worst[1]} ({worst[0]} codes); {n_in} / {k} frames within the bounds '
          f'({lim} luma, {lim} chroma, max {MAX_DIFF})')
    if any(errs):
        print('  ffmpeg: ' + ' | '.join(e[-300:] for e in errs if e))
    return 0 if n_in == k else 1


if __name__ == '__main__':
    sys.exit(main())
