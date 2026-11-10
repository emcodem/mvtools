"""Parity of the fast CPU path (SplatFps(exact=false), splat_fast) with the exact CPU path.

  python -I check_fast.py <splatcore_test.dll> <splatfps.dll> <src clip> [--start S] [--count N] [--frames F]
                          [--rate num/den] [--modes 2,1]

Stage level (splatcore_test, sc_set_fast): on stage inputs the exact path derives from N pictures of the source clip
(stage_inputs.py), every stage of the fast path against the exact path: ME pictures A, B, phase-correlation
candidates of every pair, refinement of both sides, match errors, assignment (both sides and forward side only),
local fallback, refinement to local fallback from the candidates (fields), the whole chain from luma and the picture
build at two phases. Mode 2 is the fast path with the AVX2 kernels (when the CPU has AVX2
and FMA), mode 1 the fast path without them.
Whole filter (AviSynth+ through ffmpeg, hidden): the first F source pictures of the clip (default 10),
SplatFps(exact=false) against SplatFps(exact=true), every output code.

Bounds (exit code 1 when exceeded): final vectors of the chain from luma equal (bit for bit) on at least 99.9 % of the
ME pixels; per built frame (stage build and whole filter) at most 500 luma and 500 chroma codes differing, none by more
than 2. The fast path is designed to give the exact bits (see splat_fast.h): the expected result is 0 differences.
"""
import argparse
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avs_util as AU  # noqa: E402
import stage_inputs as SI  # noqa: E402

MAX_CODES, MAX_DIFF, MIN_VEC = 500, 2, 0.999


def stage_results(lib, I):
    """every stage's outputs on the generated inputs, for the current mode"""
    A, r = I['A'], {}
    for c, y in I['y'].items():
        r[f'ME A {c}'] = lib.me(y, 0)
        r[f'ME B {c}'] = lib.me(y, 1)
    for c in I['pair']:
        r[f'pcorr {c},{c + 1}'] = lib.pcorr(I['B'][c], I['B'][c + 1])
    for c, z in I['ctr'].items():
        single = z['single']
        r[f'refine fwd {c}'] = lib.refine(A[c], A[c + 1], z['cand_f'], False)
        if not single:
            r[f'refine bwd {c}'] = lib.refine(A[c], A[c - 1], z['cand_b'], True)
        r[f'errors fwd {c}'] = np.stack([lib.error(A[c], A[c + 1], v) for v in z['rf']])
        if not single:
            r[f'errors bwd {c}'] = np.stack([lib.error(A[c], A[c - 1], v) for v in z['rb']])
        r[f'assign {c}'] = lib.assign(z['rf'], z['ef'], z.get('rb'), z.get('eb'))
        r[f'fallback {c}'] = np.stack([lib.disorder_weight(f[0], f[1]) for f in z['asg']])
        r[f'fields {c}'] = lib.fields(None if single else A[c - 1], A[c], A[c + 1], z['cand_f'], z['cand_b'])
        r[f'chain {c}'] = lib.fields_luma(None if single else I['y'][c - 1], I['y'][c], I['y'][c + 1])
    for (p, d) in I['build']:
        r[f'build {p}+{d}'] = lib.build(I['P'][p], I['P'][p + 1], I['ctr'][p]['fields'][0], I['ctr'][p + 1]['fields'][1],
                                        d)
    return r


def flat(v):
    """a stage result (array, scalar or tuple of them) as one comparable float32 array"""
    if isinstance(v, tuple):
        return np.concatenate([np.ravel(np.asarray(x, SI.f32)) for x in v])
    return np.ravel(np.asarray(v, SI.f32))


def compare_frame(Ya, Ca, Yb, Cb):
    dY, dC = np.abs(Ya.astype(np.int64) - Yb), np.abs(Ca.astype(np.int64) - Cb)
    nY, nC = int(np.count_nonzero(dY)), int(np.count_nonzero(dC))
    mx = int(max(dY.max(initial=0), dC.max(initial=0)))
    return nY, nC, mx, nY <= MAX_CODES and nC <= MAX_CODES and mx <= MAX_DIFF


def check_stages(lib, I, modes):
    bad = []
    lib.mode(0)
    ref = stage_results(lib, I)
    # the generated inputs come from the same exact path: the results must reproduce them
    c = max(I['ctr'])
    if SI.nbits(flat(ref[f'fields {c}']), flat(I['ctr'][c]['fields'])):
        bad.append('the exact path gave other fields on the second run')
    for m in modes:
        simd = lib.mode(m)
        got = stage_results(lib, I)
        lib.mode(0)
        diff = {k: SI.nbits(flat(ref[k]), flat(got[k])) for k in ref}
        notes = []
        for k, n in diff.items():
            if not n:
                continue
            notes.append(f'{k}: {n} values')
            if k.startswith('chain'):
                # final vectors: fraction of ME pixels whose fwd and bwd (vx, vy) are equal
                (aF, aB), (bF, bB) = ref[k], got[k]
                eq = min(np.mean(np.all(aF[:2] == bF[:2], axis=0)), np.mean(np.all(aB[:2] == bB[:2], axis=0)))
                notes[-1] += f' (vectors equal {100 * eq:.3f} %)'
                if eq < MIN_VEC:
                    bad.append(f'mode {m} {k}')
            elif k.startswith('build'):
                nY, nC, mx, ok = compare_frame(*SI.out_codes(ref[k]), *SI.out_codes(got[k]))
                notes[-1] += f' (codes: luma {nY}, chroma {nC}, max {mx})'
                if not ok:
                    bad.append(f'mode {m} {k}')
        label = 'fast, AVX2' if simd else 'fast, portable'
        same = sum(1 for n in diff.values() if n == 0)
        print(f'stages [mode {m}, {label}]: {same} / {len(ref)} stage outputs identical to the exact path'
              + ('' if not notes else '; differ: ' + '; '.join(notes[:10])), flush=True)
    return bad


def check_filter(splat, src, frames, rate):
    _, _, w, h, _ = AU.probe(src)
    head = AU.source_script(src, [splat]) + f'Trim(0, {frames - 1})\n'
    Ye, Ce = AU.raw_frames(head + f'SplatFps({rate.numerator}, {rate.denominator}, exact=true)\n', w, h)
    Yf, Cf = AU.raw_frames(head + f'SplatFps({rate.numerator}, {rate.denominator}, exact=false)\n', w, h)
    if len(Ye) != len(Yf) or not len(Ye):
        return [f'whole filter: {len(Ye)} / {len(Yf)} frames']
    bad, worst, ndiff_frames = [], (0, 0, 0), 0
    for k in range(len(Ye)):
        nY, nC, mx, ok = compare_frame(Ye[k], Ce[k], Yf[k], Cf[k])
        if nY or nC:
            ndiff_frames += 1
            worst = max(worst, (nY, nC, mx))
        if not ok:
            bad.append(f'whole filter frame {k}: luma {nY}, chroma {nC}, max {mx}')
    print(f'whole filter, source 0..{frames - 1} at {rate}: {len(Ye) - ndiff_frames} / {len(Ye)} output frames '
          f'identical to exact=true' + (f'; worst luma {worst[0]}, chroma {worst[1]}, max {worst[2]}' if ndiff_frames else ''),
          flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capi')
    ap.add_argument('splat')
    ap.add_argument('src')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--rate', default='60000/1001')
    ap.add_argument('--modes', default='2,1')
    args = ap.parse_args()
    lib = SI.CpuLib(args.capi)
    I = SI.make_inputs(lib, args.src, args.start, args.count)
    bad = check_stages(lib, I, [int(m) for m in args.modes.split(',')])
    if args.frames > 0:
        bad += check_filter(args.splat, args.src, args.frames, Fraction(args.rate))
    for b in bad[:20]:
        print('  FAIL', b)
    print(f'fast path within bounds: {"yes" if not bad else "no, " + str(len(bad)) + " failures"}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
