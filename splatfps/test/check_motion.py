"""Parity of the CUDA motion-estimation stages with the exact CPU path: phase correlation, refinement, assignment,
refinement to local fallback from candidates (fields) and the whole chain from luma.

  python -I check_motion.py <splatcore_test.dll> <mvtools_cuda.dll> <src clip> [--start S] [--count N]

The inputs and the CPU results come from the exact CPU path on N pictures of the source clip (stage_inputs.py).
- Phase correlation on the B pictures of every pair: the candidates must be integer vectors within the largest block's range and the
  global one constant (CPU and CUDA); CUDA must give the CPU's bits: candidates, prefiltered picture, global vector.
- Refinement (both sides), assignment (both sides, and forward side only), the fields from the phase-correlation
  candidates, and the fields as for an external vector source (backward candidates pointing from the centre picture, no refinement).
- The whole chain from luma, with the GPU time per source picture.
CUDA exact must be bit-identical with the CPU everywhere. CUDA fast (FMA, single-precision exp) is reported; its chain
from luma must give final vectors within 0.01 px of the CPU's on at least 99 % of the ME pixels.
Exit code 0 when every exact comparison matches and every match rate is at least 99 %.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_inputs as SI  # noqa: E402

MAX_CAND = 64  # the largest phase-correlation block of the defaults is 128 ME pixels: peaks within +-64


class Report:
    def __init__(self):
        self.exact_bad = []
        self.n_exact = 0
        self.rates = []

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


def pcorr_sane(v):
    return bool(np.all(v == np.round(v)) and np.all(np.abs(v[:-1]) <= MAX_CAND) and np.all(v[-1] == v[-1][:, :1, :1]))


def check(be, I, R):
    fast = not be.exact
    A = I['A']
    print(f'[{be.name}]', flush=True)
    for c, z in I['pair'].items():
        cands, hp, glob = be.pcorr(I['B'][c], I['B'][c + 1])
        sane = pcorr_sane(cands)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capi')
    ap.add_argument('cuda')
    ap.add_argument('src')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=5)
    args = ap.parse_args()
    cpu = SI.CpuLib(args.capi)
    I = SI.make_inputs(cpu, args.src, args.start, args.count)
    I['cpu'] = cpu
    R = Report()
    for c, z in I['pair'].items():
        R.n_exact += 1
        if not pcorr_sane(z['cand']):
            R.exact_bad.append(f'cpu pair {c} pcorr sanity')
            print(f'  cpu pair {c},{c + 1} pcorr: candidates not integer / in range / constant global', flush=True)
    for exact in (True, False):
        check(SI.CudaLib(args.cuda, cpu, exact), I, R)
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
