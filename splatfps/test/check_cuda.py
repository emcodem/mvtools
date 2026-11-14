"""Parity of the CUDA stages and the CUDA picture build with the exact CPU path.

  python -I check_cuda.py <mvtools_cuda.dll> <splatcore_test.dll> <src clip> [--start S] [--count N] [--args SETTINGS]

On stage inputs the exact CPU path derives from N pictures of the source clip (stage_inputs.py), CUDA exact and CUDA
fast run: ME pictures A, B, match errors of every refined candidate and the zero vector, local fallback (the weights
of the assigned fields), temporal fallback (adjusted phase and its terms, at two phases, two nonzero strengths, with
the match errors as they are and scaled up so the phase moves), the two-build blend (one pyramid reduce step with the
main and the second taps, two levels; the error map; the main and the second build; the mix on the CPU's parts), the
projection fields of both sides (splat prep) with fallback_edge off and on, and the picture build (two phases per
pair; also with fallback_edge on, on the fields as they are and with their vectors pushed out of the picture and a
local-fallback weight everywhere).
--args: named settings as the filters take them ("name=value name=v1,v2,v3"), applied to every comparison (default:
the defaults of the clip's size).

CUDA exact must be bit-identical everywhere, also run to run (the build three times) - exit code 1 otherwise. CUDA fast
(FMA, single-precision exp, splat sums in any order): stage values that differ are reported; built pictures (build,
build parts, mix) must stay within the bounds of stage_inputs.picture_diff (10 bit, 4:2:2: 500 luma and 500 chroma
codes differing at 1920 x 1080, scaled with the pixel count, codes on a rounding tie of the exact value not counted),
none by more than 2.
A CUDA entry missing from the library counts as a failure (named in the output).
CPU self-checks (exit code 1 when one fails, so every case is exercised): the temporal fallback moves the phase
somewhere; the mix of the CPU's main and second build is the CPU's whole build; the blend changes the picture; and
fallback_edge changes the projection fields and the build in at least one case.
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
# fallback_edge cases: the margin, how far (ME pixels) the vectors are pushed so that cells near the edges leave, the
# local-fallback weight they get
EDGE_ARGS = 'fallback_edge=1 fallback_edge_margin=1'
EDGE_PUSH, EDGE_FALLBACK = 6.0, 0.5


def detail(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return f'max {np.abs(a - b).max():.3g}'


def code_diff(got, ref):
    """built pictures: (luma codes differing, chroma codes differing, max difference, the same counts without rounding
    ties, within the fast bounds) - stage_inputs.picture_diff"""
    return SI.picture_diff(got, ref)


def with_args(cpu, extra):
    """the current named settings plus extra"""
    return (cpu.args + ' ' + extra).strip()


def pushed(F, dx):
    """a field with its vectors moved by dx ME pixels horizontally and (dx != 0) a local-fallback weight of at least
    EDGE_FALLBACK everywhere: real fields rarely carry a fallback weight at the picture edges, where fallback_edge acts"""
    G = F.copy()
    if dx:
        G[0] += SI.f32(dx)
        G[3] = np.maximum(G[3], SI.f32(EDGE_FALLBACK))
    return G


class Results:
    """stage comparisons of one backend: (name, values differing, detail); failures and missing entries"""

    def __init__(self, gpu):
        self.gpu, self.rows, self.bad, self.missing = gpu, [], [], []

    def run(self, name, fn, want, picture=False):
        try:
            got = fn()
        except SI.Missing as e:
            self.missing.append(str(e))
            self.bad.append(f'{name}: {e} missing')
            return None
        n = SI.nbits(got, want)
        det = detail(got, want) if n > 0 else ''
        if picture and n:
            nY, nC, mx, cY, cC, ok = code_diff(got, want)
            det += f'; codes luma {nY}, chroma {nC} (without rounding ties {cY}, {cC}), max {mx}'
            if not self.gpu.exact and not ok:
                self.bad.append(f'{name} (fast bounds)')
        if self.gpu.exact and n:
            self.bad.append(name)
        self.rows.append((name, n, det))
        return got


def stages(gpu, cpu, I, blend):
    R = Results(gpu)
    A = I['A']
    for c, y in I['y'].items():
        R.run(f'ME A {c}', lambda: gpu.me(y, 0), A[c])
        R.run(f'ME B {c}', lambda: gpu.me(y, 1), I['B'][c])
    for c, z in I['ctr'].items():
        R.run(f'errors fwd {c}', lambda: gpu.errors(A[c], A[c + 1], z['rf']), z['ef'])
        if not z['single']:
            R.run(f'errors bwd {c}', lambda: gpu.errors(A[c], A[c - 1], z['rb']), z['eb'])
        R.run(f'fallback weights {c}', lambda: gpu.disorder_weight(z['asg']), np.stack(z['ch3']))
    for (p, d), st, k in pull_cases(I):
        fP, fN = pull_fields(I, p, k)
        cd, cinfo = cpu.phase_pull(fP, fN, d, st)
        res = {}
        R.run(f'pull d {p}+{d} s{st} e{k}', lambda: np.float64(res.setdefault('r', gpu.phase_pull(fP, fN, d, st))[0]),
              np.float64(cd))
        if 'r' in res:
            R.run(f'pull terms {p}+{d} s{st} e{k}', lambda: res['r'][1], cinfo)
    first_d = min(dd for _, dd in I['build'])
    for (p, d) in sorted(I['build']):
        P, N = I['P'][p], I['P'][p + 1]
        MEP, MEN = I['ctr'][p]['fields'][0], I['ctr'][p + 1]['fields'][1]
        if blend:
            if d == first_d:
                # one reduce step of a source picture's luma and of the result, both tap sets (per source, not per phase)
                for second in (False, True):
                    lv, tag = P[0], 'second' if second else 'main'
                    for lev in (1, 2):
                        want = cpu.pyr_reduce(lv, second)
                        R.run(f'pyr reduce {tag} {p} level {lev}', lambda: gpu.pyr_reduce(lv, second), want)
                        lv = want
                e = cpu.blend_error_map(MEP, MEN)
                R.run(f'blend error map {p}', lambda: gpu.blend_error_map(MEP, MEN), e)
            e = cpu.blend_error_map(MEP, MEN)
            parts = [cpu.build_part(P, N, MEP, MEN, d, k) for k in (0, 1)]
            for k in (0, 1):
                R.run(f'build part {k} {p}+{d}', lambda: gpu.build_part(P, N, MEP, MEN, d, k), parts[k], True)
            R.run(f'blend mix {p}+{d}', lambda: gpu.blend_mix(parts[0], parts[1], e), I['build'][(p, d)], True)
        base = cpu.args
        for edge in (False, True):
            if edge:
                cpu.set_args(with_args(cpu, EDGE_ARGS))
            for side, F in ((0, MEP), (1, MEN)):
                for push in (0.0, EDGE_PUSH):
                    G = pushed(F, -push if side else push)
                    want = cpu.border_weight(G, d, side)
                    R.run(f'prep {"N" if side else "P"} {p}+{d}{" edge" if edge else ""}{" pushed" if push else ""}',
                          lambda: gpu.border_weight(G, d, side), want)
            cpu.set_args(base)
    return R


def pull_cases(I):
    return [(pd, st, k) for pd in sorted(I['build']) for st in STRENGTHS for k in ERR_SCALES]


def pull_fields(I, p, k):
    """forward field of p and backward field of p + 1, the match errors (channel 2) scaled by k"""
    fP, fN = I['ctr'][p]['fields'][0].copy(), I['ctr'][p + 1]['fields'][1].copy()
    fP[2] *= SI.f32(k)
    fN[2] *= SI.f32(k)
    return fP, fN


def cpu_checks(cpu, I, blend):
    """the self-checks of the module text: a list of failures"""
    bad = []
    cases = pull_cases(I)
    moved = [c for c in cases if cpu.phase_pull(*pull_fields(I, c[0][0], c[2]), c[0][1], c[1])[0] != c[0][1]]
    print(f'temporal fallback on the CPU: the phase moves in {len(moved)} / {len(cases)} cases (pair, phase, strength, '
          f'error scale)', flush=True)
    if not moved:
        bad.append('temporal fallback: the phase never moves, the temporal fallback is not exercised')
    if blend:
        same, changed = 0, 0
        for (p, d), ref in sorted(I['build'].items()):
            P, N = I['P'][p], I['P'][p + 1]
            MEP, MEN = I['ctr'][p]['fields'][0], I['ctr'][p + 1]['fields'][1]
            main, second = cpu.build_part(P, N, MEP, MEN, d, 0), cpu.build_part(P, N, MEP, MEN, d, 1)
            same += SI.nbits(cpu.blend_mix(main, second, cpu.blend_error_map(MEP, MEN)), ref) == 0
            changed += SI.nbits(main, ref) != 0
        n = len(I['build'])
        print(f'two-build blend on the CPU: mix of the parts == whole build in {same} / {n}, the blend changes the '
              f'picture in {changed} / {n}', flush=True)
        if same != n:
            bad.append('two-build blend: the mix of the CPU parts is not the CPU build')
        if not changed:
            bad.append('two-build blend: the blend never changes the picture, not exercised')
    else:
        print('two-build blend: off in these settings (blend_sigma 0), its stages are not compared', flush=True)
    base, prep, built = cpu.args, 0, 0
    for (p, d) in sorted(I['build']):
        P, N = I['P'][p], I['P'][p + 1]
        MEP, MEN = I['ctr'][p]['fields'][0], I['ctr'][p + 1]['fields'][1]
        for push in (0.0, EDGE_PUSH):
            out = []
            for edge in (False, True):
                cpu.set_args(with_args(cpu, EDGE_ARGS) if edge else base)
                out.append((cpu.border_weight(pushed(MEP, push), d, 0), cpu.border_weight(pushed(MEN, -push), d, 1),
                            cpu.build(P, N, pushed(MEP, push), pushed(MEN, -push), d)))
            cpu.set_args(base)
            prep += any(SI.nbits(a, b) for a, b in zip(out[0][:2], out[1][:2]))
            built += SI.nbits(out[0][2], out[1][2]) != 0
    print(f'fallback_edge on the CPU: changes the projection fields in {prep}, the build in {built} of '
          f'{2 * len(I["build"])} cases', flush=True)
    if not prep or not built:
        bad.append('fallback_edge: no effect on the projection fields or the build, not exercised')
    return bad


def builds(gpus, cpu, I):
    """the whole build (two phases per pair), and with fallback_edge on (fields as they are and pushed out)"""
    bad, base = [], cpu.args
    cases = [(pd, '', 0.0) for pd in sorted(I['build'])]
    cases += [(pd, EDGE_ARGS, push) for pd in sorted(I['build']) for push in (0.0, EDGE_PUSH)]
    for (p, d), extra, push in cases:
        cpu.set_args(with_args(cpu, extra) if extra else base)
        P, N = I['P'][p], I['P'][p + 1]
        MEP, MEN = pushed(I['ctr'][p]['fields'][0], push), pushed(I['ctr'][p + 1]['fields'][1], -push)
        ref = I['build'][(p, d)] if not extra else cpu.build(P, N, MEP, MEN, d)
        tag = f'build {p}+{d}' + (' edge' if extra else '') + (' pushed' if push else '')
        line = tag + ':'
        for g in gpus:
            best, outs = 1e9, []
            for _ in range(3):  # the GPU may be shared: the fastest of 3 runs; exact must repeat its bits
                t = time.perf_counter()
                outs.append(g.build(P, N, MEP, MEN, d))
                best = min(best, time.perf_counter() - t)
            out = outs[0]
            nY, nC, mx, cY, cC, ok = code_diff(out, ref)
            nf = SI.nbits(out, ref)
            line += (f' | {g.name}: float diff {nf}, luma codes {nY}, chroma {nC} (without rounding ties {cY}, {cC}; '
                     f'max {mx}), {best * 1000:.0f} ms')
            if g.exact:
                if nf:
                    bad.append(f'{g.name} {tag}')
                if any(SI.nbits(o, out) for o in outs[1:]):
                    bad.append(f'{g.name} {tag}: differs run to run')
                    line += ' RUN-TO-RUN DIFFERS'
            elif not ok:
                bad.append(f'{g.name} {tag} (fast bounds)')
        print(line, flush=True)
    cpu.set_args(base)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cuda')
    ap.add_argument('capi')
    ap.add_argument('src')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--args', default='')
    args = ap.parse_args()
    cpu = SI.CpuLib(args.capi)
    if args.args:
        cpu.set_args(args.args)
    I = SI.make_inputs(cpu, args.src, args.start, args.count)
    gpus = [SI.CudaLib(args.cuda, cpu, ex) for ex in (True, False)]
    print('mvtools_cuda API', gpus[0].d.mvcuda_api_version())
    blend = cpu.get('blend_sigma') > 0
    bad = cpu_checks(cpu, I, blend)
    missing = set()
    for g in gpus:
        R = stages(g, cpu, I, blend)
        d = [r for r in R.rows if r[1] != 0]
        print(f'stages [{g.name}]: {len(R.rows) - len(d)} / {len(R.rows) + len(R.missing)} identical to the exact CPU '
              f'path' + ('' if not d else '; differ: ' + ', '.join(f'{k} {n} ({x})' for k, n, x in d[:12]))
              + (f'; {len(R.missing)} not run (entry missing)' if R.missing else ''), flush=True)
        bad += [f'{g.name} {b}' for b in R.bad]
        missing |= set(R.missing)
    bad += builds(gpus, cpu, I)
    for m in sorted(missing):
        print('  MISSING', m)
    for b in bad[:30]:
        print('  FAIL', b)
    print(f'CUDA identical with the exact CPU path (exact) and within bounds (fast): '
          f'{"yes" if not bad else "no, " + str(len(bad)) + " failures"}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
