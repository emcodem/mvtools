"""SplatFps test suite: every check of splatfps/test in one run, pass / fail per case.

  python -I run_suite.py <dll dir> <source clip> [--quick] [--lsmash <LSMASHSource.dll>] [--rate num/den]
                         [--uhd <clip>]

<dll dir>: the plugin set (splatfps.dll, mvtools2.dll, mvtools_cuda.dll, cudafilters.dll) and splatcore_test.dll
(else <dll dir>/../cmake/splatfps/splatcore_test.dll, or the file SPLATCORE_TEST names).
<source clip>: a 10-bit 4:2:2 clip (1080p for the default settings), read through L-SMASH Works: the plugin comes from
SPLATFPS_LSMASH or --lsmash. Every case runs on this clip; the stage inputs are derived from it by the exact CPU path.
--rate: the output rate of the whole-filter cases (default 60000/1001, or twice the source rate when the source is
already that fast).
--quick: fewer sizes, parameter sets and formats, no sequential run, no whole filter in check_fast.
--uhd: a 10-bit 4:2:2 3840 x 2160 clip (its own per-size defaults: ME scale 8): check_motion and check_cuda on its
first pictures and a sequential CPU == CUDA exact case on it.

Cases:
- stages: check_fast (every stage of the fast CPU path, with and without the AVX2 kernels, against the exact path, and
  the whole filter exact=false against exact=true on 10 source pictures; within the bounds stated there),
  check_motion (phase correlation, refinement, assignment, the fields and the chain from luma: CUDA exact
  bit-identical with the exact CPU path, CUDA fast within 0.01 px), check_cuda (ME pictures, match errors, local and
  temporal fallback, two-build blend, projection fields with fallback_edge, picture build: CUDA exact
  bit-identical with the exact CPU path, fast within the bounds stated there); check_motion includes the
  whole-picture correlation (global_candidate 2 and 3, every FFT size the clip allows).
  check_fast is the one place the slow exact CPU path runs as a whole filter besides check_avs cuda: elsewhere the CPU
  side is the fast path, which check_fast shows to give the exact path's bits.
- whole filter through AviSynth+: check_avs, random access, SplatFps (fast) and CudaSplatFps(exact=true): every copy
  frame == its source picture, built frames == the other backend; with the defaults and with the options
  global_candidate=2, fallback_edge=1
- vector sources: check_vectors with cuda and nvof (SplatAnalyse round trip, MSplatVectors from MAnalyse and NVOFA,
  CudaSplatFps(exact=true, vectors=) == SplatFps(vectors=)), at twice the source rate on frames 1, 3, 5
- sequential: framemd5 of SplatFps (fast) == CudaSplatFps(exact=true), built-in vectors and MSplatVectors (the device
  pipeline's look-ahead path), and with global_candidate=2, fallback_edge=1; CudaSplatFps(exact=true) twice gives the
  same framemd5 (run to run)
- sizes: framemd5 of SplatFps (fast) == CudaSplatFps(exact=true) at other frame sizes (scaled source),
  each with its per-size default settings (ME scale 2 for 720 x 576, 1280 x 720 and 1006 x 542, 4 for 1442 x 1082,
  8 for 3840 x 2160)
- parameters: the same identity with non-default named arguments at the source size (ME scale 2, other borders, a
  forced nearest-block stage, other candidate costs / peaks / boxes, the phase-correlation settings, block size 16
  with a stage off, the generated ME filters, global_candidate 3 with fallback_edge and other blend settings, the
  blend off), which must also differ from the defaults
- formats: per pixel format (4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 / 12 / 16 bit, converted in the script) the same
  identity at twice the source rate, every copy frame (each even output frame) == its source frame, 8 bit ==
  the same picture as 16 bit (converted back, at most 1 code apart); centre siting (4:2:0, 4:2:2) converted, 4:2:0
  with chroma on the rows and a siting change within the clip refused
Cases run in parallel: one GPU job at a time (CUDA renders and the checks that run CUDA), CPU-only renders and
checks beside it (at most 2); the results are printed in the order above.
Needs numpy, ffmpeg and ffprobe on PATH; runs every child hidden (no console window).
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import avs_util as AU  # noqa: E402

NOWIN = AU.NOWIN
# the options meant to be used together (off by default): the whole-picture candidate and the edge rule of the fallback
OPTIONS = 'global_candidate=2, fallback_edge=1'


# Work runs in parallel, but only one job at a time uses the GPU (several at once risk running out of device memory
# or a driver reset, and distort each other's timings): GPU holds every CUDA render and every check that runs CUDA.
# CPU-only renders and checks run beside it, at most CPU_JOBS at a time.
CPU_JOBS = 2
GPU = threading.Lock()
CPU = threading.Semaphore(CPU_JOBS)
RENDERS = ThreadPoolExecutor(8)


def run(args, timeout=3600):
    r = subprocess.run([sys.executable, '-I'] + args, capture_output=True, text=True, creationflags=NOWIN, timeout=timeout,
                       cwd=HERE, stdin=subprocess.DEVNULL)
    return r.returncode, (r.stdout + r.stderr).strip()


def proc(args, gpu=True):
    """a check script under the GPU lock (gpu) or the CPU slots"""
    with (GPU if gpu else CPU):
        return run(args)


def md5(script):
    """framemd5 of a script; scripts with CudaSplatFps hold the GPU lock"""
    with (GPU if 'CudaSplatFps' in script else CPU):
        return AU.framemd5(script)


def md5s(*scripts):
    """framemd5 of several scripts, rendered in parallel (GPU ones one at a time); results in the given order"""
    return [f.result() for f in [RENDERS.submit(md5, s) for s in scripts]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ddir')
    ap.add_argument('src')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--lsmash')
    ap.add_argument('--rate')
    ap.add_argument('--uhd')
    args = ap.parse_args()
    if args.lsmash:
        os.environ['SPLATFPS_LSMASH'] = os.path.abspath(args.lsmash)
    lsmash = AU.lsmash()  # fails with a clear message when not set
    ddir, src, quick = os.path.abspath(args.ddir), os.path.abspath(args.src), args.quick
    capi = os.environ.get('SPLATCORE_TEST') or os.path.join(ddir, 'splatcore_test.dll')
    if not os.path.isfile(capi):
        capi = os.path.join(ddir, '..', 'cmake', 'splatfps', 'splatcore_test.dll')
    cuda = os.path.join(ddir, 'mvtools_cuda.dll')
    splat = os.path.join(ddir, 'splatfps.dll')
    n_src, rin, _, _, pix = AU.probe(src)
    rout = Fraction(args.rate) if args.rate else (Fraction(60000, 1001) if rin < Fraction(60000, 1001) else 2 * rin)
    rate = f'{rout.numerator}/{rout.denominator}'
    print(f'source {src}: {n_src} pictures, {rin} fps, {pix}; whole filter at {rate}; plugins {ddir}', flush=True)
    t0 = time.perf_counter()
    cases = []  # futures of the case functions, in order; each returns a list of (name, rc, output)
    work = ThreadPoolExecutor(16)

    def add(fn):
        cases.append(work.submit(fn))

    def check(name, args, gpu=True):
        add(lambda: [(name, *proc(args, gpu))])

    # stages, on inputs derived from the first pictures of the source (check_fast is CPU only)
    check('check_fast', ['check_fast.py', capi, splat, src, '--rate', rate, '--frames', '0' if quick else '10'], False)
    check('check_motion', ['check_motion.py', capi, cuda, src])
    check('check_cuda', ['check_cuda.py', cuda, capi, src])
    uhd = os.path.abspath(args.uhd) if args.uhd else None
    if uhd:
        check('check_motion UHD', ['check_motion.py', capi, cuda, uhd])
        check('check_cuda UHD', ['check_cuda.py', cuda, capi, uhd])
    # whole filter, random access: every copy frame and a spread of built frames
    for v in ([] if quick else ['cpu']) + ['cuda']:
        check(f'check_avs {v}', ['check_avs.py', ddir, capi, src, rate, 'copies,built12' if v == 'cpu' else
                                 'copies,built6', v])
    check('check_avs cuda, global_candidate=2, fallback_edge=1',
          ['check_avs.py', ddir, capi, src, rate, 'built6', 'cuda', OPTIONS])
    # vector sources: odd frames at twice the source rate are built at phase 0.5
    check('check_vectors', ['check_vectors.py', ddir, src, f'{(2 * rin).numerator}/{(2 * rin).denominator}', '1,3,5',
                            'cuda', 'nvof'])
    r_num, r_den = rout.numerator, rout.denominator
    base = f'LoadPlugin("{lsmash}")\nLoadPlugin("{splat}")\n'
    source = f'c = LWLibavVideoSource("{src}", cache=false)'
    # sequential: the look-ahead path of the device pipeline
    if not quick:
        head_v = (base + f'LoadPlugin("{os.path.join(ddir, "mvtools2.dll")}")\n{source}.Trim(0, 24)\n'
                  'sup = MSuper(c, pel=2, hpad=16, vpad=16)\n'
                  'v = MSplatVectors(MAnalyse(sup, isb=true, delta=1, blksize=16, overlap=8), '
                  'MAnalyse(sup, isb=false, delta=1, blksize=16, overlap=8))\n')
        for label, extra in (('built-in', ''), ('MSplatVectors', ', vectors=v'), (OPTIONS, ', ' + OPTIONS)):
            def seq(label=label, extra=extra):
                scripts = [head_v + f'SplatFps(c, {r_num}, {r_den}{extra})\n',
                           head_v + f'CudaSplatFps(c, {r_num}, {r_den}, exact=true{extra})\n']
                if not extra:
                    # the exact variant must not depend on the order of work on the device
                    scripts.append(scripts[1])
                res = md5s(*scripts)
                (a, ea), (b, eb) = res[:2]
                out = [(f'sequential CPU == CUDA exact, {label}', 0 if a and a == b else 1,
                        f'{len(a)} / {len(b)} frames {ea} {eb}'.strip())]
                if not extra:
                    b2, eb2 = res[2]
                    out.append(('sequential CUDA exact run to run', 0 if b and b == b2 else 1,
                                f'{len(b)} / {len(b2)} frames, {sum(x != y for x, y in zip(b, b2))} differ {eb} {eb2}'
                                .strip()))
                return out
            add(seq)
        if uhd:
            # 3840 x 2160 with its own defaults (ME scale 8, two-build blend), sequential
            def seq_uhd():
                h = base + f'c = LWLibavVideoSource("{uhd}", cache=false).Trim(0, 8)\n'
                u_rate = 'c.FrameRateNumerator * 2, c.FrameRateDenominator'
                (a, ea), (b, eb) = md5s(h + f'SplatFps(c, {u_rate})\n', h + f'CudaSplatFps(c, {u_rate}, exact=true)\n')
                return [('sequential UHD CPU == CUDA exact', 0 if a and a == b else 1,
                         f'{len(a)} / {len(b)} frames, {sum(x != y for x, y in zip(a, b))} differ {ea} {eb}'.strip())]
            add(seq_uhd)
    # other frame sizes: SplatFps (fast) == CudaSplatFps(exact=true) on the source scaled in the script; 1006 x 542 and
    # 1442 x 1082 are not mod 8 / 16 and give odd ME grids
    for w, h in ((720, 576), (1280, 720), (1006, 542)) + (() if quick else ((1442, 1082), (3840, 2160))):
        def size(w=w, h=h):
            hd = base + f'{source}.Trim(0, 4).Spline36Resize({w}, {h})\n'
            (a, ea), (g, eg) = md5s(hd + f'SplatFps(c, {r_num}, {r_den})\n',
                                    hd + f'CudaSplatFps(c, {r_num}, {r_den}, exact=true)\n')
            return [(f'size {w}x{h}: CPU == CUDA exact', 0 if a and a == g else 1,
                     f'{len(a)} / {len(g)} frames, {sum(x != y for x, y in zip(a, g))} differ {ea} {eg}'.strip())]
        add(size)
    # non-default settings (named arguments): ME scale 2 with other borders, a forced nearest-block stage, other
    # candidate costs and box sizes; and the generated ME filters at the default scale
    params = (('me_scale=2, pad_x=24, pad_y=20, nearest_block=[-1, -1, 1], secondary_peak_cost=1.5, '
               'global_cost=0.5, zero_cost=1.5, error_edge_dist=2, refine_box=15, speed_change_cost=0.75'),
              ('me_filter_generated=true, block_peaks=[1, 3, 1], highpass_sigma=3, block_taper=1.0, noise_weight=0.5, '
               'peak_ratio=0.4, global_candidate=0, fallback_box=41, fallback_strength=2'),
              'block_size=16, block_peaks=[2, 0, 4], highpass_sigma=0',
              ('global_candidate=3, fallback_edge=1, fallback_edge_margin=2, blend_sigma=2, blend_levels=3, '
               'blend_err_lo=2, blend_err_hi=12'),
              'blend_sigma=0')
    head = base + f'{source}.Trim(0, 4)\n'
    defaults = work.submit(md5, head + f'SplatFps(c, {r_num}, {r_den})\n')  # shared by the parameter cases
    for k, extra in enumerate(params[:1] if quick else params):
        def param(k=k, extra=extra):
            (a, ea), (g, eg) = md5s(head + f'SplatFps(c, {r_num}, {r_den}, {extra})\n',
                                    head + f'CudaSplatFps(c, {r_num}, {r_den}, exact=true, {extra})\n')
            d, ed = defaults.result()
            return [(f'parameters {k + 1}: CPU == CUDA exact (and not the defaults)', 0 if a and a == g and a != d else 1,
                     f'{len(a)} / {len(g)} frames, {sum(x != y for x, y in zip(a, g))} differ, '
                     f'{sum(x != y for x, y in zip(a, d))} differ from the defaults {ea} {eg}'.strip())]
        add(param)
    # pixel formats: the CPU and CUDA exact must agree and copy frames must be the source
    to420 = 'ConvertToYUV420(ChromaOutPlacement="MPEG2")'
    formats = (('YUV420P8', f'{to420}.ConvertBits(8, dither=-1)'), ('YUV420P10', to420),
               ('YUV420P16', f'{to420}.ConvertBits(16)'), ('YUV422P8', 'ConvertBits(8, dither=-1)'),
               ('YUV422P12', 'ConvertBits(12)'), ('YUV444P8', 'ConvertToYUV444().ConvertBits(8, dither=-1)'),
               ('YUV444P16', 'ConvertToYUV444().ConvertBits(16)'),
               ('YUV420P8 centre siting', 'ConvertToYUV420(ChromaOutPlacement="MPEG1").ConvertBits(8, dither=-1)'
                '.propSet("_ChromaLocation", 1)'),
               ('YUV422P10 centre siting', 'propSet("_ChromaLocation", 1)'))
    rate2 = 'c.FrameRateNumerator * 2, c.FrameRateDenominator'
    md5_of = lambda lines: [l.split(',')[-2:] for l in lines]  # size and md5 (the time stamps differ)
    for name, conv in formats[:2] if quick else formats:
        def fmt(name=name, conv=conv):
            h = head + f'c = c.{conv}\n'
            (s, es), (a, ea), (g, eg) = md5s(h + 'c\n', h + f'SplatFps(c, {rate2})\n',
                                             h + f'CudaSplatFps(c, {rate2}, exact=true)\n')
            copies = md5_of(a)[0::2] == md5_of(s)[:len(a[0::2])]
            return [(f'format {name}: CPU == CUDA exact, copies == source', 0 if a and a == g and copies else 1,
                     f'{len(a)} / {len(g)} frames, {sum(x != y for x, y in zip(a, g))} differ, copies '
                     f'{"==" if copies else "!="} source {es} {ea} {eg}'.strip())]
        add(fmt)
    if not quick:
        # 8 bit and the same codes at 16 bit are the same values inside: the outputs differ only by the rounding
        def bits():
            h = head + f'c = c.{formats[0][1]}\n'
            (a, ea), (b, eb) = md5s(h + f'SplatFps(c, {rate2}, exact=false)\n',
                                    h + f'SplatFps(c.ConvertBits(16), {rate2}, exact=false).ConvertBits(8, dither=-1)\n')
            same = sum(x == y for x, y in zip(md5_of(a), md5_of(b)))
            copies = md5_of(a)[0::2] == md5_of(b)[0::2]
            return [('format 8 bit vs 16 bit: copy frames identical (built frames may differ by the rounding)',
                     0 if a and len(a) == len(b) and copies else 1, f'{same} / {len(a)} frames identical {ea} {eb}'.strip())]
        add(bits)

    # chroma siting: frame 0 is checked when the script opens, later frames when they are used
    def siting():
        h = head + f'c = c.{formats[0][1]}\n'
        (_, e0), (_, e1) = md5s(h + 'SplatFps(c.propSet("_ChromaLocation", 2), 50, 1)\n',
                                h + 'c = c.Trim(0, -1) ++ c.Trim(1, 0).propSet("_ChromaLocation", 1)\nSplatFps(c, 50, 1)\n')
        ok = '_ChromaLocation 2 (top_left)' in e0 and 'frame 1 has another chroma siting' in e1
        return [('chroma siting: 4:2:0 on the rows refused, a change within the clip refused', 0 if ok else 1,
                 '' if ok else f'{e0[-300:]} | {e1[-300:]}')]
    add(siting)

    # results in the order of the cases (each printed once it and every case before it are done)
    results = []
    for f in cases:
        for name, rc, out in f.result():
            ok = rc == 0
            results.append((name, ok))
            last = out.splitlines()[-1] if out else ''
            print(f'{"PASS" if ok else "FAIL"}  {name} (at {time.perf_counter() - t0:.0f} s): {last}', flush=True)
            if not ok and out:
                print('      ' + '\n      '.join(out.splitlines()[-15:]), flush=True)
    work.shutdown()
    RENDERS.shutdown()
    n_ok = sum(ok for _, ok in results)
    print(f'total {time.perf_counter() - t0:.0f} s')
    print(f'{n_ok} / {len(results)} pass')
    sys.exit(0 if n_ok == len(results) else 1)

if __name__ == '__main__':
    main()
