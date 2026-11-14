"""Parity of the CUDA motion-estimation stages with the exact CPU path: phase correlation, refinement, assignment,
refinement to local fallback from candidates (fields) and the whole chain from luma.

  python -I check_motion.py <splatcore_test.dll> <mvtools_cuda.dll> <src clip> [--start S] [--count N]
                             [--args SETTINGS] [--no-whole]

The inputs and the CPU results come from the exact CPU path on N pictures of the source clip (stage_inputs.py).
- Phase correlation on the B pictures of every pair: the candidates must be integer vectors within the largest block's range and the
  global one constant (CPU and CUDA); CUDA must give the CPU's bits: candidates, prefiltered picture, global vector.
- Refinement (both sides), assignment (both sides, and forward side only), the fields from the phase-correlation
  candidates, and the fields as for an external vector source (backward candidates pointing from the centre picture, no refinement).
- The whole chain from luma, with the GPU time per source picture.
- Whole-picture correlation (global_candidate 2 and 3) on the pairs of the clip, at the ME scales that give every FFT
  size the clip allows (512, 1024 and an ME picture larger than 1024, where 2 falls back to the vote): the surface,
  its peak and height, the candidates of the pair and the global vector, and (at the clip's own scale) the chain from
  luma; also on a synthetic pair (a picture and itself moved by 100 ME pixels, beyond the blocks' range). CPU
  self-checks: the candidates carry the whole-picture vector (2: the global field; 3: the vote, then the whole-picture
  vector) or the vote where the picture is too large; the whole-picture vector of the synthetic pair is its motion
  and (3) differs from the vote.
--args: named settings as the filters take them ("name=value ..."), applied to every comparison; --no-whole: skip the
whole-picture cases.
CUDA exact must be bit-identical with the CPU everywhere. CUDA fast (FMA, single-precision exp) is reported; its chain
from luma must give final vectors within 0.01 px of the CPU's on at least 99 % of the ME pixels, and its whole-picture
peaks and global vectors must equal the CPU's (counted as a match rate per case). A CUDA entry missing from the
library counts as a failure.
Exit code 0 when every exact comparison matches and every match rate is at least 99 %.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_inputs as SI  # noqa: E402

MAX_CAND = 64  # the largest phase-correlation block of the defaults is 128 ME pixels: peaks within +-64
# whole-picture cases: a synthetic pair, a picture and itself moved right by SHIFT ME pixels (edge repeated): beyond the
# blocks' range, within half the smallest FFT size, so the whole-picture vector differs from the blocks' vote
SHIFT = 100


def shifted(B, s):
    o = np.empty_like(B)
    o[:, s:] = B[:, :-s]
    o[:, :s] = B[:, :1]
    return o


class Report:
    def __init__(self):
        self.exact_bad = []
        self.n_exact = 0
        self.rates = []
        self.missing = set()

    def exact(self, tag, got, want, must=True):
        k = SI.nbits(np.asarray(got, SI.f32), np.asarray(want, SI.f32))
        if must:
            self.n_exact += 1
            if k:
                self.exact_bad.append(tag)
        if k < 0:
            return f'shape {np.shape(got)} != {np.shape(want)}'
        if k:
            dif = np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64))
            return f'{k} of {np.size(got)} differ (max {np.nanmax(dif):.3g})'
        return 'identical'

    def rate(self, tag, r):
        self.rates.append((tag, r))
        return f'{100 * r:.3f} %'


def vec_match(F, R, tol):
    """fraction of ME pixels whose (vx, vy) equal the CPU's within tol ME pixels per component"""
    return float(np.mean(np.all(np.abs(F[:2].astype(np.float64) - R[:2]) <= tol, axis=0)))


def pcorr_sane(v, nglob=1):
    """integer candidates, the block ones within the largest block's range, the nglob global fields constant"""
    g = v[len(v) - nglob:]
    return bool(np.all(v == np.round(v)) and np.all(np.abs(v[:len(v) - nglob]) <= MAX_CAND)
                and np.all(g == g[:, :, :1, :1]))


def check(be, I, R):
    fast = not be.exact
    A = I['A']
    print(f'[{be.name}]', flush=True)
    nglob = 2 if int(I['cpu'].get('global_candidate')) == 3 else 1
    for c, z in I['pair'].items():
        cands, hp, glob = be.pcorr(I['B'][c], I['B'][c + 1])
        sane = pcorr_sane(cands, nglob)
        R.n_exact += 1
        if not sane:
            R.exact_bad.append(f'{be.name} pair {c} pcorr sanity')
        t = f'{be.name} pair {c}'
        print(f'  pair {c},{c + 1} pcorr: {len(cands)} candidates, integer / in range / constant global '
              f'{"yes" if sane else "NO"}; vs CPU: cands {R.exact(t + " pcorr cands", cands, z["cand"], not fast)}, '
              f'hp {R.exact(t + " hp", hp, z["hp"], not fast)}, glob {R.exact(t + " glob", glob, z["glob"], not fast)}',
              flush=True)
    for c, z in I['ctr'].items():
        single = z['single']
        t = f'{be.name} centre {c}'
        line = f'  centre {c}{" (forward side only)" if single else ""}:'
        rf = be.refine(A[c], A[c + 1], z['cand_f'], False)
        line += f' refine fwd {R.exact(t + " refine fwd", rf, z["rf"], not fast)}'
        if not single:
            rb = be.refine(A[c], A[c - 1], z['cand_b'], True)
            line += f', bwd {R.exact(t + " refine bwd", rb, z["rb"], not fast)}'
        F, B = be.assign(z['rf'], z['ef'], z.get('rb'), z.get('eb'))
        line += (f'; assign fwd {R.exact(t + " assign fwd", F, z["asg"][0], not fast)}, '
                 f'bwd {R.exact(t + " assign bwd", B, z["asg"][1], not fast)}')
        prev = None if single else A[c - 1]
        F, B = be.fields(prev, A[c], A[c + 1], z['cand_f'], z['cand_b'])
        line += (f'; fields fwd {R.exact(t + " fields fwd", F, z["fields"][0], not fast)}, '
                 f'bwd {R.exact(t + " fields bwd", B, z["fields"][1], not fast)}')
        if not single:
            # as an external vector source delivers them: backward candidates pointing from c, used as they are
            ext_b = -z['cand_b']
            cF, cB = I['cpu'].fields(prev, A[c], A[c + 1], z['cand_f'], ext_b, toward_b=1, refine=0)
            F, B = be.fields(prev, A[c], A[c + 1], z['cand_f'], ext_b, toward_b=1, refine=0)
            line += (f'; fields external candidates fwd {R.exact(t + " fields ext fwd", F, cF, not fast)}, '
                     f'bwd {R.exact(t + " fields ext bwd", B, cB, not fast)}')
        print(line, flush=True)
        F, B, ms = be.fields_luma(None if single else I['y'][c - 1], I['y'][c], I['y'][c + 1])
        tm = (f'; GPU ms: ingest {ms[0]:.2f} + pair {ms[1]:.2f} + fields {ms[2]:.2f} = {ms[3]:.2f} per source picture')
        rF, rB = z['chain']
        if fast:
            # refined vectors carry FMA rounding (~1e-5 px): the decisions must agree within 0.01 px
            print(f'  centre {c} chain from luma vs CPU: final vectors equal (0.01 px) fwd '
                  f'{R.rate(t + " chain fwd", vec_match(F, rF, 0.01))}, bwd {R.rate(t + " chain bwd", vec_match(B, rB, 0.01))}'
                  f' (bit-identical: fwd {R.exact(t + " chain fwd", F, rF, False)}){tm}', flush=True)
        else:
            print(f'  centre {c} chain from luma vs CPU: fwd {R.exact(t + " chain fwd", F, rF)}, '
                  f'bwd {R.exact(t + " chain bwd", B, rB)}{tm}', flush=True)


def whole_cases(cpu):
    """(label, settings, FFT size) of the whole-picture cases: global_candidate 2 and 3 at the clip's own ME scale,
    then 2 at the other scales that reach FFT sizes not yet covered (1024, larger than 1024)"""
    base, W, H = cpu.args, cpu.W, cpu.H
    own = SI.whole_size(cpu.mw, cpu.mh)
    out = [(f'global_candidate {g}', (base + f' global_candidate={g}').strip(), own) for g in (2, 3)]
    seen = {min(own, 2048)}
    for s in (8, 4, 2):
        spec = (base + f' global_candidate=2 me_scale={s}').strip()
        try:
            cpu.set_args(spec, W, H)
        except ValueError:
            continue
        T = min(SI.whole_size(cpu.mw, cpu.mh), 2048)
        if T not in seen:
            seen.add(T)
            out.append((f'global_candidate 2, me_scale {s}', spec, T))
    cpu.set_args(base, W, H) if base else cpu.set_size(W, H)
    return out


def check_whole(cpu, gpus, I, R):
    base = cpu.args
    cases = whole_cases(cpu)
    found = set()
    for label, spec, T in cases:
        cpu.set_args(spec)
        gc = int(cpu.get('global_candidate'))
        own = spec == (base + f' global_candidate={gc}').strip()
        Bs = I['B'] if own else {c: cpu.me(y, 1) for c, y in I['y'].items()}
        print(f'[{label}] ME grid {cpu.mw} x {cpu.mh}, FFT {T if T <= 1024 else "none (picture larger than 1024)"}, '
              f'{cpu.n} candidates', flush=True)
        pairs = [(f'{c},{c + 1}', Bs[c], Bs[c + 1]) for c in I['pair']]
        pairs.append((f'{min(I["pair"])} moved {SHIFT}', Bs[min(I['pair'])], shifted(Bs[min(I['pair'])], SHIFT)))
        for c, Bp, Bc in pairs:
            wp = cpu.whole_peak(Bp, Bc)
            cands, hp, glob = cpu.pcorr(Bp, Bc)
            # CPU self-checks: the global fields carry the whole-picture vector (or the vote when too large)
            ok = pcorr_sane(cands, 2 if gc == 3 else 1)
            if wp is not None:
                if wp[0][2] > 0:
                    found.add(label)
                g = cands[-1][:, 0, 0]
                ok = ok and bool(np.all(g == wp[0][:2]))
                if gc == 2:
                    ok = ok and bool(np.all(glob[:2] == wp[0][:2]))
                if 'moved' in c:
                    # the motion of the synthetic pair (content moves right: the vector from cur back to prev or
                    # forward, either sign convention), and with 3 a vote that differs from it
                    ok = ok and abs(float(wp[0][0])) == SHIFT and float(wp[0][1]) == 0.0
                    if gc == 3:
                        ok = ok and bool(np.any(cands[-2][:, 0, 0] != cands[-1][:, 0, 0]))
            elif T <= 1024:
                ok = False
            R.n_exact += 1
            if not ok:
                R.exact_bad.append(f'cpu {label} pair {c} whole-picture candidates')
            line = f'  pair {c}: CPU peak {"-" if wp is None else tuple(float(v) for v in wp[0])}, global ' \
                   f'{tuple(float(v) for v in glob[:2])}, candidates {"ok" if ok else "NOT AS DESCRIBED"}'
            for g in gpus:
                t = f'{g.name} {label} pair {c}'
                try:
                    gw = g.whole_peak(Bp, Bc)
                except SI.Missing as e:
                    R.missing.add(str(e))
                    R.exact_bad.append(f'{t}: {e} missing')
                    gw = 'missing'
                if gw != 'missing':
                    if (gw is None) != (wp is None):
                        R.exact_bad.append(f'{t} whole peak: {"none" if gw is None else "a peak"} where the CPU has '
                                           f'{"none" if wp is None else "one"}')
                        line += f' | {g.name}: whole peak presence differs'
                    elif wp is not None:
                        line += (f' | {g.name}: surface {R.exact(t + " surface", gw[1], wp[1], g.exact)}, '
                                 f'peak {R.exact(t + " peak", gw[0], wp[0], g.exact)}')
                        if not g.exact:
                            line += f' (position {R.rate(t + " peak position", float(np.all(gw[0][:2] == wp[0][:2])))})'
                try:
                    gc_, ghp, gglob = g.pcorr(Bp, Bc)
                except RuntimeError as e:
                    R.exact_bad.append(f'{t} pcorr: {e}')
                    line += f' | {g.name}: pcorr {e}'
                    continue
                line += (f' | {g.name}: cands {R.exact(t + " cands", gc_, cands, g.exact)}, '
                         f'glob {R.exact(t + " glob", gglob, glob, g.exact)}')
                if not g.exact:
                    line += f' (global vector {R.rate(t + " global vector", float(np.all(gglob[:2] == glob[:2])))})'
            print(line, flush=True)
        if own:
            for c, z in I['ctr'].items():
                yp = None if z['single'] else I['y'][c - 1]
                rF, rB = cpu.fields_luma(yp, I['y'][c], I['y'][c + 1])
                line = f'  centre {c} chain from luma:'
                for g in gpus:
                    t = f'{g.name} {label} centre {c} chain'
                    try:
                        F, B, _ = g.fields_luma(yp, I['y'][c], I['y'][c + 1], reps=1)
                    except RuntimeError as e:
                        R.exact_bad.append(f'{t}: {e}')
                        line += f' | {g.name}: {e}'
                        continue
                    if g.exact:
                        line += f' | {g.name}: fwd {R.exact(t + " fwd", F, rF)}, bwd {R.exact(t + " bwd", B, rB)}'
                    else:
                        line += (f' | {g.name}: vectors (0.01 px) fwd {R.rate(t + " fwd", vec_match(F, rF, 0.01))}, '
                                 f'bwd {R.rate(t + " bwd", vec_match(B, rB, 0.01))}')
                print(line, flush=True)
    cpu.set_args(base) if base else cpu.set_size(cpu.W, cpu.H)
    R.n_exact += 1
    if not found:
        R.exact_bad.append('cpu whole-picture correlation: no surface with a peak, not exercised')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capi')
    ap.add_argument('cuda')
    ap.add_argument('src')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--args', default='')
    ap.add_argument('--no-whole', action='store_true')
    args = ap.parse_args()
    cpu = SI.CpuLib(args.capi)
    if args.args:
        cpu.set_args(args.args)
    I = SI.make_inputs(cpu, args.src, args.start, args.count)
    I['cpu'] = cpu
    R = Report()
    nglob = 2 if int(cpu.get('global_candidate')) == 3 else 1
    for c, z in I['pair'].items():
        R.n_exact += 1
        if not pcorr_sane(z['cand'], nglob):
            R.exact_bad.append(f'cpu pair {c} pcorr sanity')
            print(f'  cpu pair {c},{c + 1} pcorr: candidates not integer / in range / constant global', flush=True)
    gpus = [SI.CudaLib(args.cuda, cpu, exact) for exact in (True, False)]
    for g in gpus:
        check(g, I, R)
    if not args.no_whole:
        check_whole(cpu, gpus, I, R)
    for m in sorted(R.missing):
        print('  MISSING', m)
    print(f'exact comparisons: {R.n_exact - len(R.exact_bad)} / {R.n_exact} bit-identical')
    print(f'match rates: {len(R.rates) - sum(r < 0.99 for _, r in R.rates)} / {len(R.rates)} at least 99 %; lowest '
          + (f'{100 * min(r for _, r in R.rates):.3f} %' if R.rates else '-'))
    low = [(t, r) for t, r in R.rates if r < 0.99]
    for t in R.exact_bad[:20]:
        print('  FAIL', t)
    for t, r in low[:20]:
        print(f'  LOW {t}: {100 * r:.3f} %')
    return 1 if R.exact_bad or low else 0


if __name__ == '__main__':
    sys.exit(main())
