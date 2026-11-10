"""Parity of the CUDA stages and the CUDA picture build with the exact CPU path.

  python -I check_cuda.py <mvtools_cuda.dll> <splatcore_test.dll> <src clip> [--start S] [--count N]

On stage inputs the exact CPU path derives from N pictures of the source clip (stage_inputs.py), CUDA exact and CUDA
fast run: ME pictures A, B, match errors of every refined candidate and the zero vector, local fallback (the weights
of the assigned fields), temporal fallback (adjusted phase and its terms, at two phases, two nonzero strengths, with
the match errors as they are and scaled up so the phase moves) and the picture build (two phases per pair), each
compared with the exact CPU result. CUDA exact must be bit-identical everywhere (exit code 1 otherwise; the phase must
move in at least one temporal-fallback case). CUDA fast (FMA, single-precision
exp, splat sums in any order) is reported: the number of values that differ and, for the build, the output codes
(10 bit, 4:2:2) that differ (count, max); it is not judged here.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_inputs as SI  # noqa: E402

STRENGTHS = (1.0, 0.5)
# the temporal fallback also runs on the fields with their match errors scaled up: a clean clip's errors stay below the fallback's
# threshold, where the adjusted phase is the phase itself and the smooth step is not exercised
ERR_SCALES = (1, 24)


def detail(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return f'max {np.abs(a - b).max():.3g}'


def stages(gpu, cpu, I):
    """(name, values that differ, detail) per stage comparison"""
    res = []

    def add(name, got, want):
        n = SI.nbits(got, want)
        res.append((name, n, detail(got, want) if n > 0 else ''))

    A = I['A']
    for c, y in I['y'].items():
        add(f'ME A {c}', gpu.me(y, 0), A[c])
        add(f'ME B {c}', gpu.me(y, 1), I['B'][c])
    for c, z in I['ctr'].items():
        add(f'errors fwd {c}', gpu.errors(A[c], A[c + 1], z['rf']), z['ef'])
        if not z['single']:
            add(f'errors bwd {c}', gpu.errors(A[c], A[c - 1], z['rb']), z['eb'])
        add(f'fallback weights {c}', gpu.disorder_weight(z['asg']), np.stack(z['ch3']))
    for (p, d), st, k in pull_cases(I):
        fP, fN = pull_fields(I, p, k)
        gd, ginfo = gpu.phase_pull(fP, fN, d, st)
        cd, cinfo = cpu.phase_pull(fP, fN, d, st)
        add(f'pull d {p}+{d} s{st} e{k}', np.float64(gd), np.float64(cd))
        add(f'pull terms {p}+{d} s{st} e{k}', ginfo, cinfo)
    return res


def pull_cases(I):
    return [(pd, st, k) for pd in sorted(I['build']) for st in STRENGTHS for k in ERR_SCALES]


def pull_fields(I, p, k):
    """forward field of p and backward field of p + 1, the match errors (channel 2) scaled by k"""
    fP, fN = I['ctr'][p]['fields'][0].copy(), I['ctr'][p + 1]['fields'][1].copy()
    fP[2] *= SI.f32(k)
    fN[2] *= SI.f32(k)
    return fP, fN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cuda')
    ap.add_argument('capi')
    ap.add_argument('src')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=5)
    args = ap.parse_args()
    cpu = SI.CpuLib(args.capi)
    I = SI.make_inputs(cpu, args.src, args.start, args.count)
    bad = []
    gpus = [SI.CudaLib(args.cuda, cpu, ex) for ex in (True, False)]
    print('mvtools_cuda API', gpus[0].d.mvcuda_api_version())
    # the temporal fallback must move the phase somewhere, else the formula is not exercised
    cases = pull_cases(I)
    moved = [c for c in cases if cpu.phase_pull(*pull_fields(I, c[0][0], c[2]), c[0][1], c[1])[0] != c[0][1]]
    print(f'temporal fallback on the CPU: the phase moves in {len(moved)} / {len(cases)} cases (pair, phase, strength, error scale)',
          flush=True)
    if not moved:
        bad.append('temporal fallback: the phase never moves, the temporal fallback is not exercised')
    for g in gpus:
        res = stages(g, cpu, I)
        d = [r for r in res if r[1] != 0]
        print(f'stages [{g.name}]: {len(res) - len(d)} / {len(res)} identical to the exact CPU path'
              + ('' if not d else '; differ: ' + ', '.join(f'{k} {n} ({x})' for k, n, x in d[:10])), flush=True)
        if g.exact:
            bad += [k for k, *_ in d]
    for (p, d), ref in sorted(I['build'].items()):
        P, N = I['P'][p], I['P'][p + 1]
        MEP, MEN = I['ctr'][p]['fields'][0], I['ctr'][p + 1]['fields'][1]
        rY, rC = SI.out_codes(ref)
        line = f'build {p}+{d}:'
        for g in gpus:
            best = 1e9
            for _ in range(3):  # the GPU may be shared: the fastest of 3 runs
                t = time.perf_counter()
                out = g.build(P, N, MEP, MEN, d)
                best = min(best, time.perf_counter() - t)
            Y, Cc = SI.out_codes(out)
            dY, dC = np.abs(Y - rY), np.abs(Cc - rC)
            nf = SI.nbits(out, ref)
            line += (f" | {g.name}: float diff {nf}, luma codes {int(np.count_nonzero(dY))} (max {int(dY.max())}), "
                     f"chroma {int(np.count_nonzero(dC))} (max {int(dC.max())}), {best * 1000:.0f} ms")
            if g.exact and nf:
                bad.append(f'build {p}+{d}')
        print(line, flush=True)
    for b in bad[:20]:
        print('  FAIL', b)
    print(f'CUDA exact identical with the exact CPU path: {"yes" if not bad else "no, " + str(len(bad)) + " differ"}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
