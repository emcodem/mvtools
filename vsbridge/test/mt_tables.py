"""mt_tables.py: the bridge's port of masktools2 (vsbridge/mt_port.cpp) against masktools2 2.2.30 + AviSynth+ 3.7.5.

Each case runs mt_lutxy(px, py, expr, ...) or mt_adddiff(px, mt_lut(py, expr, ...)) on pattern clips: 'full' patterns
hold every (x, y) pair of the bit depth (px = column, py = row, in every plane), so one frame is the whole lookup table
a cudafilter builds from masktools (cudafilters/CudaKobs.cpp, lut_from_cpu); 'mix' patterns (12 / 16 bit) a spread of
pairs. Under VapourSynth the cases go through CudaLutxy / CudaQtgmcAddNoise, whose CPU fallback and GPU table are the
bridge's port: without libmvtools_cuda.so the frame is the port's CPU output, with it (8 / 10 bit) the GPU's output
from the table the port built.

usage (Windows, AviSynth+ with masktools2):  python mt_tables.py ref         -> writes mt_tables.md5 next to this file
        env FFMPEG (ffmpeg.exe with AviSynth input), MASKTOOLS (masktools2.dll)
usage (VapourSynth, the bridge's test setup):  python mt_tables.py [case ...]  -> compares with mt_tables.md5
        env MVPF_PLUGIN_DIR (libcudafilters.so + libmvtools_vsbridge.so, with or without libmvtools_cuda.so)
"""
import hashlib
import os
import subprocess
import sys
import tempfile


def need(var, what):
    """Value of a required environment variable (no machine-specific default)."""
    v = os.environ.get(var)
    if not v:
        sys.exit('set %s (%s)' % (var, what))
    return v

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'mt_tables.md5')

# expressions cudafilters turn into tables (QTGMC_gpu.avsi), then operator coverage for both evaluators
QTGMC = [
    ('kobs1', 'x 129 scalef < x y range_half < range_half y ? ?', 2),
    ('kobs2', 'x 127 scalef > x y range_half > range_half y ? ?', 1),
    ('sharp075', 'x x y - 0.750000 * +', 1),
    ('sharp022', 'x x y - 0.220000 * +', 1),
    ('precise', 'x y < x 1 scalef + x y > x 1 scalef - x ? ?', 2),
    ('tw1', 'x 3 scalef + y < x 3 scalef + x 3 scalef - y > x 3 scalef - y ? ?', 2),
    ('tw2', 'x 7 scalef + y < x 2 scalef + x 7 scalef - y > x 2 scalef - x 51 * y 49 * + 100 / ? ?', 2),
    ('match', 'x x y - 0.500000 * +', 2),
]
OPS = [
    ('absminmax', 'x y - abs 3 * x y + 0.5 * min y range_half - 2 / max'),
    ('cmp', 'x y <= y x >= & x range_max * x 1.5 * ?'),
    ('cmp2', 'x y == x y != | x y < x y > | & x 2 * y ?'),
    ('rounding', 'x 2 / round y 2 / round + x 4 / floor - y 4 / ceil + x 3 / trunc +'),
    ('stack', 'x y dup1 dup1 - abs swap2 + + 3 /'),
    ('div', 'x y 1 + / 100 *'),
    ('frac', 'x 0.1 * y 0.7 * + 1.3 *'),
    ('consts', 'x ymin ymax clip y cmin cmax clip + 2 / range_size range_min + 4 / -'),
    ('pi', 'pi x * 7 / y +'),
    ('scales', 'x 100 scaleb - abs y 16 yscalef + + 60 yscaleb -'),
]
OWN_ONLY = [('andnot', 'x y &! x y ?')]
EXPR_ONLY = [('exprfn', 'x y * sqrt x y xor 50 * + x y > not 30 * +'),
             ('exprlogic', 'x y > x 128 scalef < and x y or 200 * x ?')]

# case: name -> (filter, expr, use_expr, plane args, layout, bits, pattern); P3: every plane processed, as in the
# tables cudafilters build (masktools' defaults U=1, V=1 make CudaLutxy use its CPU filter)
P3 = {'Y': 3, 'U': 3, 'V': 3}
CASES = {}
for bits in (8, 10):
    for name, expr, ue in QTGMC:
        CASES['%s_%d' % (name, bits)] = ('lutxy', expr, ue, P3, '444', bits, 'full')
    for name, expr in OPS:
        for ue in (0, 3):
            CASES['%s_ue%d_%d' % (name, ue, bits)] = ('lutxy', expr, ue, P3, '444', bits, 'full')
    for name, expr in OWN_ONLY:
        CASES['%s_%d' % (name, bits)] = ('lutxy', expr, 0, P3, '444', bits, 'full')
    for name, expr in EXPR_ONLY:
        CASES['%s_%d' % (name, bits)] = ('lutxy', expr, 3, P3, '444', bits, 'full')
    for name, centre, w, ue, cf in (('grain', '128.5', '0.700000', 2, False), ('noise', '128.5', '0.300000', 3, True),
                                    ('noise128', '128', '0.300000', 3, True)):
        expr = 'x %s scalef - %s * range_half +' % (centre, w)
        CASES['%s_%d' % (name, bits)] = ('addnoise', expr, ue, {'clamp_float': cf, 'lutUV': 3, 'addUV': 3}, '444', bits,
                                         'full')
    # plane modes, 4:2:0
    CASES['modes_a_%d' % bits] = ('lutxy', 'x y + 2 /', 0, {'Y': 3, 'U': 2, 'V': 4}, '420', bits, 'full')
    CASES['modes_b_%d' % bits] = ('lutxy', 'x y - abs', 3, {'Y': 2, 'U': 3, 'V': 1}, '420', bits, 'full')
    CASES['grain_qtgmc_%d' % bits] = ('addnoise', 'x 128.5 scalef - 0.700000 * range_half +', 2,
                                      {'lutUV': 1, 'addUV': 2}, '420', bits, 'full')
    # planar RGB: plane-specific modes (G processed, B from the first clip, R from the second) show the bridge's
    # G, B, R <-> R, G, B plane mapping; Expr's scalef uses the luma rule on every RGB plane
    CASES['modes_rgb_%d' % bits] = ('lutxy', 'x y - abs 2 *', 0, {'Y': 3, 'U': 2, 'V': 4}, 'rgb', bits, 'full')
    CASES['scalef_rgb_%d' % bits] = ('lutxy', 'x 100 scalef - abs y +', 3, P3, 'rgb', bits, 'full')
    # different content per R / G/ B plane (built per named plane on both sides): the input side of the mapping
    CASES['modes_rgbmix_%d' % bits] = ('lutxy', 'x y - abs 2 *', 0, {'Y': 3, 'U': 2, 'V': 4}, 'rgb', bits, 'mixrgb')
for bits in (12, 16):
    for name, expr, ue in [q for q in QTGMC if q[0] in ('kobs1', 'sharp075', 'tw2')] + [
            ('absminmax_ue0', OPS[0][1], 0), ('absminmax_ue3', OPS[0][1], 3), ('frac_ue0', OPS[6][1], 0),
            ('frac_ue2', OPS[6][1], 2), ('scales_ue3', OPS[9][1], 3)]:
        CASES['%s_%d' % (name, bits)] = ('lutxy', expr, ue, P3, '444', bits, 'mix')
    CASES['grain_%d' % bits] = ('addnoise', 'x 128.5 scalef - 0.700000 * range_half +', 2,
                                {'clamp_float': False, 'lutUV': 3, 'addUV': 3}, '444', bits, 'mix')
    CASES['noise_%d' % bits] = ('addnoise', 'x 128.5 scalef - 0.300000 * range_half +', 3,
                                {'clamp_float': True, 'lutUV': 3, 'addUV': 3}, '444', bits, 'mix')

# 'mix' patterns: sample = (column * a + row * b + c) mod 2^bits, 1024 x 1024; 'mixrgb': plus an offset per R, G, B
MIX = {'x': (37, 101, 0), 'y': (53, 29, 7)}
RGB_OFFSET = (0, 85, 170)


def balanced(expr):
    """True when the RPN expression leaves exactly one value (AviSynth+ crashes on some unbalanced ones)."""
    arity = {'?': 3, 'clip': 3, 'abs': 1, 'round': 1, 'floor': 1, 'ceil': 1, 'trunc': 1, 'sqrt': 1, 'not': 1,
             'scalef': 1, 'scaleb': 1, 'yscalef': 1, 'yscaleb': 1}
    depth = 0
    for t in expr.split():
        if t.startswith('dup'):
            k = int(t[3:] or 0)
            if depth < k + 1:
                return False
            depth += 1
        elif t.startswith('swap'):
            if depth < int(t[4:] or 1) + 1:
                return False
        elif t in arity:
            if depth < arity[t]:
                return False
            depth -= arity[t] - 1
        elif t in ('+', '-', '*', '/', 'min', 'max', '<', '<=', '>', '>=', '==', '=', '!=', '&', '|', '&!', 'and', 'or',
                   'xor'):
            if depth < 2:
                return False
            depth -= 1
        else:
            depth += 1
    return depth == 1


assert all(balanced(c[1]) for c in CASES.values()), [k for k, c in CASES.items() if not balanced(c[1])]


def size(bits, pattern):
    return 1 << bits if pattern == 'full' else 1024


# --- AviSynth+ (reference) ----------------------------------------------------------------------------------------

def avs_pattern(which, layout, bits, pattern):
    n = size(bits, pattern)
    pt = {('444', 8): 'YV24', ('420', 8): 'YV12', ('rgb', 8): 'RGBP'}.get(
        (layout, bits), 'RGBP%d' % bits if layout == 'rgb' else 'YUV%sP%d' % (layout, bits))
    if pattern == 'mixrgb':
        a, b, c = MIX[which]
        y = 'Y8' if bits == 8 else 'Y%d' % bits
        g = ['BlankClip(width=%d, height=%d, pixel_type="%s", length=1).Expr("sx %d * sy %d * + %d + %d %%")'
             % (n, n, y, a, b, c + o, 1 << bits) for o in RGB_OFFSET]
        return 'CombinePlanes(%s, %s, %s, planes="RGB", source_planes="YYY", pixel_type="%s")' % (g[0], g[1], g[2], pt)
    if pattern == 'full':
        e = 'sx' if which == 'x' else 'sy'
    else:
        a, b, c = MIX[which]
        e = 'sx %d * sy %d * + %d + %d %%' % (a, b, c, 1 << bits)
    return 'BlankClip(width=%d, height=%d, pixel_type="%s", length=1).Expr("%s", "%s", "%s")' % (n, n, pt, e, e, e)


def avs_script(case, masktools):
    filt, expr, ue, pa, layout, bits, pattern = CASES[case]
    s = 'LoadPlugin("%s")\npx = %s\npy = %s\n' % (masktools, avs_pattern('x', layout, bits, pattern),
                                                  avs_pattern('y', layout, bits, pattern))
    if filt == 'lutxy':
        planes = ''.join(', %s=%d' % (k, v) for k, v in pa.items())
        return s + 'mt_lutxy(px, py, "%s", use_expr=%d%s)\n' % (expr, ue, planes)
    cf = 'true' if pa.get('clamp_float') else 'false'
    return s + 'mt_adddiff(px, mt_lut(py, "%s", use_expr=%d, clamp_float=%s, U=%d, V=%d), U=%d, V=%d)\n' % (
        expr, ue, cf, pa['lutUV'], pa['lutUV'], pa['addUV'], pa['addUV'])


def pix_fmt(layout, bits):
    # planar RGB: ffmpeg's gbrp, planes G, B, R as AviSynth stores them
    return ('gbrp' if layout == 'rgb' else 'yuv%sp' % layout) + ('' if bits == 8 else '%dle' % bits)


def make_ref():
    ffmpeg = os.environ.get('FFMPEG', 'ffmpeg')
    masktools = need('MASKTOOLS', 'masktools2.dll')
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            avs = os.path.join(tmp, case + '.avs')
            with open(avs, 'w') as f:
                f.write(avs_script(case, masktools))
            r = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', avs, '-f', 'rawvideo', '-pix_fmt',
                                pix_fmt(CASES[case][4], CASES[case][5]), '-'], capture_output=True)
            if r.returncode:
                sys.exit('%s: %s' % (case, r.stderr.decode(errors='replace').strip()))
            lines.append('%s %s' % (case, hashlib.md5(r.stdout).hexdigest()))
            print(lines[-1], flush=True)
    with open(REF, 'w', newline='\n') as f:
        f.write('# masktools2 2.2.30 + AviSynth+ 3.7.5 (Windows), made by mt_tables.py ref\n' + '\n'.join(lines) + '\n')


# --- VapourSynth (the bridge) -------------------------------------------------------------------------------------

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
                k, v = line.split()
                ref[k] = v

    def pattern(which, layout, bits, pattern):
        side = size(bits, pattern)
        fmt = getattr(vs, 'RGB%d' % (3 * bits) if layout == 'rgb' else 'YUV%sP%d' % (layout, bits))
        blank = core.std.BlankClip(width=side, height=side, format=fmt, length=1)
        dtype = np.uint8 if bits == 8 else np.uint16

        def fill(n, f):
            o = f.copy()
            for p in range(3):
                a = np.asarray(o[p])
                h, w = a.shape
                col = np.arange(w)[None, :].repeat(h, 0)
                row = np.arange(h)[:, None].repeat(w, 1)
                if pattern == 'full':
                    v = col if which == 'x' else row
                else:
                    ka, kb, kc = MIX[which]
                    if pattern == 'mixrgb':
                        kc += RGB_OFFSET[p]  # VS planes R, G, B
                    v = (col * ka + row * kb + kc) % (1 << bits)
                a[:, :] = v.astype(dtype)
            return o
        return core.std.ModifyFrame(blank, blank, fill)

    bad = 0
    for case in names:
        filt, expr, ue, pa, layout, bits, pat = CASES[case]
        px, py = pattern('x', layout, bits, pat), pattern('y', layout, bits, pat)
        try:
            u, v = cf.CudaUpload(px), cf.CudaUpload(py)
            if filt == 'lutxy':
                c = cf.CudaLutxy(u, v, expr=expr, use_expr=ue, **pa)
            else:
                c = cf.CudaQtgmcAddNoise(u, v, expr, use_expr=ue, **pa)
            fr = cf.CudaDownload(c).get_frame(0)
            m = hashlib.md5()
            for p in ((1, 2, 0) if layout == 'rgb' else (0, 1, 2)):  # VS R, G, B -> G, B, R as AviSynth / gbrp
                m.update(np.ascontiguousarray(np.asarray(fr[p])).tobytes())
            got = m.hexdigest()
        except vs.Error as e:
            got = 'ERROR ' + str(e).strip().splitlines()[-1][:200]
        ok = got == ref.get(case)
        bad += not ok
        print('%-22s %s%s' % (case, 'same' if ok else 'DIFF', '' if ok else '  ' + got), flush=True)
    print('%d of %d cases differ' % (bad, len(names)))
    return 1 if bad else 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['ref']:
        make_ref()
    else:
        sys.exit(vs_run(args or list(CASES)))
