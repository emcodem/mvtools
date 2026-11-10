"""SplatFps with exchangeable vector sources (candidate clips) against its built-in phase correlation.

  python -I check_vectors.py <dll dir> <src.mov> <rate num/den> <frames k,k,...> [mvtools args] [cuda] [nvof]

Per frame and source: luma samples that differ from SplatFps without vectors=, and how many by more than 2 codes.
- SplatAnalyse round trip: must be identical (the same candidates, through a clip).
- MSplatVectors (MVTools MAnalyse vectors, mvtools2.dll from the same folder): a different motion estimator, so the
  frames differ, but on a clean pan the difference stays small (reported); a direction or sign error in the conversion shows up
  as large differences on every built frame.
- nvof: NVOFA vectors (CudaNVOFAnalyse, blksize 4 = the ME grid, cudafilters.dll) through MSplatVectors too.
- cuda: CudaSplatFps(exact=true, vectors=...) for both sources too, which must equal SplatFps(vectors=...).
mvtools args: extra MAnalyse arguments (default "blksize=16, overlap=8"). Runs ffmpeg hidden (-nostdin, no window).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avs_util as AU  # noqa: E402


def main():
    ddir, src, rate, which = sys.argv[1:5]
    rest = sys.argv[5:]
    cuda, nvof = 'cuda' in rest, 'nvof' in rest
    rest = [a for a in rest if a not in ('cuda', 'nvof')]
    margs = rest[0] if rest else 'blksize=16, overlap=8'
    num, den = rate.split('/')
    frames = [int(x) for x in which.split(',')]
    head = AU.source_script(src, [os.path.join(ddir, 'splatfps.dll'), os.path.join(ddir, 'mvtools2.dll')], var='c')
    _, _, w, h, _ = AU.probe(src)
    sup = 'sup = MSuper(c, pel=2, hpad=16, vpad=16)\n'
    mv = (f'bw = MAnalyse(sup, isb=true, delta=1, {margs})\nfw = MAnalyse(sup, isb=false, delta=1, {margs})\n')
    variants = {
        'builtin': f's = SplatFps(c, {num}, {den}, exact=true)\n',
        'SplatAnalyse': f's = SplatFps(c, {num}, {den}, exact=true, vectors=SplatAnalyse(c))\n',
        'MSplatVectors': sup + mv + f's = SplatFps(c, {num}, {den}, exact=true, vectors=MSplatVectors(bw, fw))\n',
        'MSplatVectors refine=false': sup + mv + f's = SplatFps(c, {num}, {den}, exact=true, vectors=MSplatVectors(bw, fw, refine=false))\n',
    }
    if nvof:
        nv = ''.join(f'{s} = CudaNVOFAnalyse(c, isb={b}, delta=1, blksize=4, overlap=0, pel=2, hpad=64, vpad=64)' + chr(10)
                     for s, b in (('nbw', 'true'), ('nfw', 'false')))
        variants['MSplatVectors NVOFA'] = (f'LoadPlugin("{os.path.join(ddir, "cudafilters.dll")}")' + chr(10) + nv +
                                           f's = SplatFps(c, {num}, {den}, exact=true, vectors=MSplatVectors(nbw, nfw))' + chr(10))
    if cuda:
        variants['Cuda exact SplatAnalyse'] = f's = CudaSplatFps(c, {num}, {den}, exact=true, vectors=SplatAnalyse(c))\n'
        variants['Cuda exact MSplatVectors'] = (sup + mv + f's = CudaSplatFps(c, {num}, {den}, exact=true, '
                                                'vectors=MSplatVectors(bw, fw))\n')
    out = {name: AU.avs_frames(head + v, frames, w, h)[0] for name, v in variants.items()}
    ref = out['builtin']
    ok = True
    for name, Y in out.items():
        if name == 'builtin':
            continue
        diffs = [np.abs(Y[j] - ref[j]) for j in range(len(frames))]
        cells = ', '.join(f'{k}: {int((d > 0).sum())} / {int((d > 2).sum())}' for k, d in zip(frames, diffs))
        print(f'{name:28s} differ / >2 per frame: {cells}', flush=True)
        if name == 'SplatAnalyse' and any(d.any() for d in diffs):
            ok = False
    for g, c in (('SplatAnalyse', 'Cuda exact SplatAnalyse'), ('MSplatVectors', 'Cuda exact MSplatVectors')):
        if c in out:
            same = all((out[g][j] == out[c][j]).all() for j in range(len(frames)))
            print(f'{c} == {g} (CPU): {same}')
            ok = ok and same
    print('all identity checks pass' if ok else 'an identity check FAILS')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
