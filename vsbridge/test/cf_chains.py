"""cf_chains.py: cudafilters chains whose CPU fallbacks are AviSynth plugins (RgTools, nnedi3, masktools, QTGMC.avsi)
that Linux AviSynth+ does not have, so compare.py cannot run them there. References come from AviSynth+ on Windows with
the release bundle's cudafilters.dll / mvtools_cuda.dll (built from the same sources) and the real plugins; under
VapourSynth the GPU path runs with the bridge's stand-ins and must give the same frames.

The input is synthetic so both hosts make it bit for bit alike: per plane, sample = (2x^2 + 3y^2 + xy + 13n(x + 1) + 7y)
mod 2^bits (x, y plane coordinates, n frame number; exact in AviSynth+ Expr's float32).

usage (Windows):      python cf_chains.py ref         -> cf_chains.md5 next to this file
        env FFMPEG (ffmpeg.exe with AviSynth input), BUNDLE (release bundle folder), AVS_PLUGINS (FFAStrans avs_plugins)
usage (VapourSynth):  python cf_chains.py [case ...]  -> compares with cf_chains.md5
        env MVPF_PLUGIN_DIR (libcudafilters.so, libmvtools_vsbridge.so, libmvtools_cuda.so)
"""
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'cf_chains.md5')

INPUTS = {
    'hd420p8': (1920, 1080, '420', 8),
    'hd444p10': (1280, 720, '444', 10),
}
FRAMES = 10
FPS = (30000, 1001)

# case -> (input, body in AviSynth form; Cuda* calls become cf.*, REV(x) / INV(x) Reverse / Invert; result in `out`)
CASES = {
    'removegrain12': ('hd420p8', 'out = CudaDownload(CudaRemoveGrain(CudaUpload(c), 12, -1))'),
    'removegrain20': ('hd444p10', 'out = CudaDownload(CudaRemoveGrain(CudaUpload(c), 20))'),
    'nnedi3_m2': ('hd420p8', 'out = CudaDownload(CudaNNEDI3(CudaUpload(c), field=-2))'),
    'nnedi3_1': ('hd444p10', 'out = CudaDownload(CudaNNEDI3(CudaUpload(c), field=1))'),
    'logic_max': ('hd420p8', 'out = CudaDownload(CudaLogic(CudaUpload(c), CudaUpload(REV(c)), mode="max", U=3, V=3))'),
    'logic_min': ('hd444p10', 'out = CudaDownload(CudaLogic(CudaUpload(c), CudaUpload(REV(c)), mode="min", U=2, V=2))'),
    'clamp': ('hd420p8', 'out = CudaDownload(CudaClamp(CudaUpload(c), CudaUpload(REV(c)), CudaUpload(INV(c)), '
                         'overshoot=2, undershoot=3, U=3, V=3))'),
    'expand': ('hd420p8', 'out = CudaDownload(CudaExpand(CudaUpload(c), mode="vertical", U=3, V=3))'),
    'inpand': ('hd444p10', 'out = CudaDownload(CudaInpand(CudaUpload(c), mode="vertical", U=3, V=3))'),
    'kobs4': ('hd420p8', 'out = CudaDownload(CudaKeepOnlyBobShimmerFixes(CudaUpload(c), CudaUpload(REV(c)), rep=4))'),
    'kobs1_10': ('hd444p10', 'out = CudaDownload(CudaKeepOnlyBobShimmerFixes(CudaUpload(c), CudaUpload(REV(c)), rep=1, '
                             'chroma=false))'),
    'sharplimit': ('hd420p8', 'u = CudaUpload(c)\nout = CudaDownload(CudaQtgmcSharpLimit(u, CudaUpload(REV(c)), '
                              'CudaUpload(INV(c)), u, "x x y - 0.750000 * +", 12, 0, tr1=1, smode=2, precise=true, '
                              'nref=3))'),
    'sharplimit_s1': ('hd444p10', 'u = CudaUpload(c)\nout = CudaDownload(CudaQtgmcSharpLimit(u, CudaUpload(REV(c)), '
                                  'CudaUpload(INV(c)), u, "x x y - 0.400000 * +", 12, 1, tr1=2, smode=1, nref=3, '
                                  'degrain2=CudaUpload(INV(REV(c)))))'),
    'backblend': ('hd420p8', 'out = CudaDownload(CudaQtgmcBackBlend(CudaUpload(c), CudaUpload(REV(c)), 1919.75, 1079.5))'),
    'addnoise': ('hd420p8', 'out = CudaDownload(CudaQtgmcAddNoise(CudaUpload(c), CudaUpload(REV(c)), '
                            '"x 128.5 scalef - 0.700000 * range_half +", use_expr=2, lutUV=3, addUV=3))'),
    'lutxy': ('hd444p10', 'out = CudaDownload(CudaLutxy(CudaUpload(c), CudaUpload(REV(c)), '
                          '"x 3 scalef + y < x 3 scalef + x 3 scalef - y > x 3 scalef - y ? ?", use_expr=2, U=3, V=3))'),
    'binomial0': ('hd420p8', 'out = CudaDownload(CudaQtgmcBinomial0(CudaUpload(c)))'),
    'bob': ('hd420p8', 'out = CudaDownload(CudaBob(CudaUpload(c), 0.0, 0.5))'),
    'gauss': ('hd420p8', 'out = CudaDownload(CudaGaussResize(CudaUpload(c), 1280, 720, p=5.0))'),
}


def pattern_expr(bits):
    return ('sx sx * 2 * sy sy * 3 * + sx sy * + frameno 13 * sx 1 + * + sy 7 * + %d %%' % (1 << bits))


def avs_source(inp):
    w, h, layout, bits = INPUTS[inp]
    pt = {('420', 8): 'YV12'}.get((layout, bits), 'YUV%sP%d' % (layout, bits))
    e = pattern_expr(bits)
    return ('BlankClip(width=%d, height=%d, pixel_type="%s", length=%d, fps=%d, fps_denominator=%d).KillAudio()'
            '.Expr("%s", "%s", "%s")' % (w, h, pt, FRAMES, FPS[0], FPS[1], e, e, e))


def avs_body(body):
    return body.replace('REV(', 'Reverse(').replace('INV(', 'Invert(')


def make_ref():
    ffmpeg = os.environ.get('FFMPEG', r'C:\FFAStrans\Processors\ffmpeg\x64\ffmpeg.exe')
    bundle = os.environ.get('BUNDLE', r'C:\dev\mvtools\release\mvtools_gpu_qtgmc_2026-11-01b').replace('\\', '/')
    base = os.environ.get('AVS_PLUGINS', r'C:\FFAStrans\Processors\avs_plugins').replace('\\', '/')
    head = ''.join('LoadPlugin("%s/plugins/%s")\n' % (bundle, p) for p in
                   ('cudafilters.dll', 'masktools2.dll', 'RgTools.dll', 'plugins_JPSDR.dll'))
    head += 'Import("%s/avsi/QTGMC.avsi")\n' % base
    lines = []
    env = dict(os.environ, CUDAFILTERS_LOG='1')
    with tempfile.TemporaryDirectory() as tmp:
        for case, (inp, body) in CASES.items():
            avs = os.path.join(tmp, case + '.avs')
            with open(avs, 'w') as f:
                f.write(head + 'c = %s\n%s\nreturn out\n' % (avs_source(inp), avs_body(body)))
            r = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', avs, '-c:v', 'rawvideo', '-f',
                                'framemd5', '-'], capture_output=True, text=True, env=env)
            md5s = [l.split(',')[-1].strip() for l in r.stdout.splitlines() if l and not l.startswith('#')]
            if r.returncode or not md5s:
                sys.exit('%s: rc %d %s' % (case, r.returncode, r.stderr.strip()[-400:]))
            paths = sorted(set(l for l in r.stderr.splitlines() if l.startswith('Cuda')))
            lines.append('%s %d %s' % (case, len(md5s), hashlib.md5(''.join(md5s).encode()).hexdigest()))
            print(lines[-1], '|', '; '.join(paths), flush=True)
    with open(REF, 'w', newline='\n') as f:
        f.write('# AviSynth+ 3.7.5 (Windows), release bundle 2026-11-01b (cudafilters.dll 530b06a, mvtools_cuda.dll '
                '15811aa), masktools2 2.2.30, RgTools, plugins_JPSDR, QTGMC.avsi; made by cf_chains.py ref\n'
                '# case frames md5(concatenated per-frame md5s)\n' + '\n'.join(lines) + '\n')


def vs_run(names):
    import numpy as np
    import vapoursynth as vs
    sys.path.insert(0, HERE)
    from vstest import core, load_cudafilters
    load_cudafilters()
    cf = core.cuda
    ref = {}
    with open(REF) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                k, n, m = line.split()
                ref[k] = '%s %s' % (n, m)

    def source(inp):
        w, h, layout, bits = INPUTS[inp]
        blank = core.std.BlankClip(width=w, height=h, format=getattr(vs, 'YUV%sP%d' % (layout, bits)), length=FRAMES,
                                   fpsnum=FPS[0], fpsden=FPS[1])

        def fill(n, f):
            o = f.copy()
            for p in range(3):
                a = np.asarray(o[p])
                ph, pw = a.shape
                x = np.arange(pw, dtype=np.int64)[None, :]
                y = np.arange(ph, dtype=np.int64)[:, None]
                v = (2 * x * x + 3 * y * y + x * y + 13 * n * (x + 1) + 7 * y) % (1 << bits)
                a[:, :] = v.astype(a.dtype)
            return o
        return core.std.ModifyFrame(blank, blank, fill)

    bad = 0
    for case in names:
        inp, body = CASES[case]
        g = {'cf': cf, 'core': core, 'c': source(inp), 'REV': core.std.Reverse, 'INV': core.std.Invert}
        code = body.replace('true', 'True').replace('false', 'False')
        code = __import__('re').sub(r'\b(Cuda\w*)\(', r'cf.\1(', code)
        try:
            exec(code, g)
            out = g['out']
            md5s = []
            for n in range(out.num_frames):
                fr = out.get_frame(n)
                m = hashlib.md5()
                for p in range(fr.format.num_planes):
                    m.update(np.ascontiguousarray(np.asarray(fr[p])).tobytes())
                md5s.append(m.hexdigest())
            got = '%d %s' % (len(md5s), hashlib.md5(''.join(md5s).encode()).hexdigest())
        except vs.Error as e:
            got = 'ERROR ' + str(e).strip().splitlines()[-1][:300]
        ok = got == ref.get(case)
        bad += not ok
        print('%-14s %s  %s' % (case, 'same' if ok else 'DIFF', '' if ok else got), flush=True)
    print('%d of %d cases differ' % (bad, len(names)))
    return 1 if bad else 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['ref']:
        make_ref()
    else:
        sys.exit(vs_run(args or list(CASES)))
