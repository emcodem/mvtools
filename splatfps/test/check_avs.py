"""The whole filter through AviSynth+ (ffmpeg's avs input, hidden), random access: copy frames and built frames.

  python -I check_avs.py <dll dir> <splatcore_test.dll> <src clip> <rate num/den> <frames> <cpu | cuda>

<frames>: comma-separated tokens: "copies" (every copy frame of the clip: P side, N side and end-of-clip
placeholders), "builtK" (K built frames spread over the clip), or frame numbers. The frames are selected inside the
script and requested in a shuffled order (random access). The frame plan is the library's own (sc_plan, default
copy threshold).
- cpu: SplatFps (fast path). Copy frames must be the source picture code for code; built frames must equal
  CudaSplatFps(exact=true).
- cuda: CudaSplatFps(exact=true). Copy frames must be the source picture; built frames must equal
  SplatFps(exact=true) (the exact CPU path as a whole filter).
10-bit 4:2:2 sources; the default settings of the filters.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avs_util as AU  # noqa: E402
import stage_inputs as SI  # noqa: E402


def main():
    ddir, capi, src, rate, which, variant = sys.argv[1:7]
    from fractions import Fraction
    rout = Fraction(rate)
    n, rin, w, h, pix = AU.probe(src)
    if pix != 'yuv422p10le':
        sys.exit(f'{src}: a 10-bit 4:2:2 source is needed, not {pix}')
    lib = SI.CpuLib(capi, w, h)
    plan = lib.plan(n, rin, rout)
    frames = []
    for tok in which.split(','):
        if tok == 'copies':
            frames += [k for k, p, d, kind in plan if kind != 'built']
        elif tok.startswith('built'):
            built = [k for k, p, d, kind in plan if kind == 'built']
            cnt = int(tok[5:])
            frames += [built[(i * (len(built) - 1)) // max(1, cnt - 1)] for i in range(cnt)]
        else:
            frames.append(int(tok))
    frames = sorted(set(frames))
    random.Random(1).shuffle(frames)
    r = f'{rout.numerator}, {rout.denominator}'
    if variant == 'cpu':
        test, other = f'SplatFps({r}, exact=false)', f'CudaSplatFps({r}, exact=true)'
    else:
        test, other = f'CudaSplatFps({r}, exact=true)', f'SplatFps({r}, exact=true)'
    head = AU.source_script(src, [os.path.join(ddir, 'splatfps.dll')])
    src_script = AU.source_script(src, var='s')

    def src_index(k):
        # the source picture a copy frame passes on (as the filter picks it)
        _, p, d, kind = plan[k]
        return p if kind == 'copyP' else (p + 1 if kind == 'copyN' else min(n - 1, p + (1 if 2 * d > 1 else 0)))

    bad = 0
    for i in range(0, len(frames), 32):
        chunk = frames[i:i + 32]
        Y, C = AU.avs_frames(head + f's = {test}\n', chunk, w, h)
        if len(Y) != len(chunk):
            sys.exit(f'got {len(Y)} frames for {len(chunk)}')
        copies = [k for k in chunk if plan[k][3] != 'built']
        built = [k for k in chunk if plan[k][3] == 'built']
        want = {}
        if copies:
            sY, sC = AU.avs_frames(src_script, [src_index(k) for k in copies], w, h)
            want.update({k: (sY[m], sC[m]) for m, k in enumerate(copies)})
        if built:
            oY, oC = AU.avs_frames(head + f's = {other}\n', built, w, h)
            want.update({k: (oY[m], oC[m]) for m, k in enumerate(built)})
        for j, k in enumerate(chunk):
            rY, rC = want[k]
            dy, dc = np.count_nonzero(Y[j] != rY), np.count_nonzero(C[j] != rC)
            if dy or dc:
                bad += 1
                print(f'frame {k} ({plan[k][3]}): luma {dy}, chroma {dc} samples differ', flush=True)
    kinds = {}
    for k in frames:
        kinds[plan[k][3]] = kinds.get(plan[k][3], 0) + 1
    print(f'{test}: {len(frames) - bad} / {len(frames)} frames identical (copies with the source, built frames with '
          f'{other}) {kinds}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
