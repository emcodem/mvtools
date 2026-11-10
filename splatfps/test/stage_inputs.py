"""Stage inputs for the SplatFps parity tests, made by the library itself from any source clip.

N consecutive source pictures (10-bit 4:2:2, any size) are read through AviSynth+ (LWLibavVideoSource) and turned into
float planes as the filter's input stage does (value = (code - 512) / 4, chroma 4:2:2 -> 4:4:4 with the filter's
half-band interpolation). The exact CPU path (splatcore_test.dll, sc_set_fast(0)) then derives every stage's input and
output from them:
  ME pictures A / B, phase-correlation candidates of each pair, refined candidates of both sides, match errors of
  every refined candidate and the zero vector, the assignment with the default candidate costs, the full fields with
  the local fallback (from the candidates, and the whole chain from luma), and built pictures at two phases.
The other paths (the fast CPU path, CUDA exact and fast) run the same stages on these inputs and are compared with the
exact CPU results. Grid sizes and candidate counts come from the library's configuration.

Also here: ctypes wrappers of splatcore_test.dll (CpuLib) and of the CUDA host test entries of mvtools_cuda.dll
(CudaLib), shared by check_fast, check_motion, check_cuda and check_avs.
"""
import ctypes as C
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avs_util as AU  # noqa: E402

f32 = np.float32
FP = C.POINTER(C.c_float)
KINDS = ('copyP', 'copyN', 'built', 'eos')


def fp(a):
    return None if a is None else a.ctypes.data_as(FP)


def cf(a):
    return np.ascontiguousarray(a, f32)


def codes10(v):
    """float samples -> 10-bit codes as the filter's output stage: 512 + round half up (4 v), clamped"""
    return (512 + np.floor(np.asarray(v, np.float64) * 4 + 0.5)).clip(0, 1023).astype(np.int32)


def out_codes(o):
    """4:4:4 float picture (3, h, w) -> Y codes (h, w), 4:2:2 C codes (2, h, w / 2) from the even samples"""
    return codes10(o[0]), np.stack([codes10(o[1][:, 0::2]), codes10(o[2][:, 0::2])])


def nbits(a, b):
    """number of values that differ bit for bit (floats compared as bits)"""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return -1
    if a.dtype == f32 and b.dtype == f32:
        return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    return int(np.count_nonzero(a != b))


class CpuLib:
    """splatcore_test.dll: the CPU stages; mode(0) exact path, mode(1) fast without AVX2, mode(2) fast"""

    def __init__(self, path, w=1920, h=1080):
        self.d = C.CDLL(os.path.abspath(path))
        if self.d.sc_version() < 8:
            sys.exit('splatcore_test.dll too old (sc_version < 8)')
        d = self.d
        d.sc_disorder_weight.argtypes = [FP, FP, C.c_int, C.c_int, FP]
        d.sc_phase_pull.restype = C.c_double
        d.sc_phase_pull.argtypes = [FP, FP, C.c_int, C.c_int, C.c_double, C.c_double, C.POINTER(C.c_double)]
        d.sc_phase_pull_strength.restype = C.c_double
        d.sc_phase_pull_strength.argtypes = [C.c_longlong] * 4
        d.sc_plan.argtypes = [C.c_int] + [C.c_longlong] * 4 + [C.c_int, C.POINTER(C.c_longlong)]
        d.sc_build.argtypes = [FP, FP, C.c_int, C.c_int, FP, FP, C.c_int, C.c_int, C.c_float, FP]
        self.set_size(w, h)

    def set_size(self, w, h):
        """the defaults of picture size w x h; sets the ME grid (mw, mh) from the configuration"""
        if self.d.sc_set_size(w, h) != 0:
            sys.exit(f'the defaults of {w} x {h} do not pass the configuration checks')
        self.W, self.H = w, h
        self.cfg = C.create_string_buffer(self.d.sc_config(None))
        self.d.sc_config(self.cfg)
        scale, pad_x, pad_y = struct.unpack_from('3i', self.cfg.raw)  # splat::Config begins with scale, pad_x, pad_y
        self.mw, self.mh = (w + 2 * pad_x) // scale, (h + 2 * pad_y) // scale
        self.n = self.d.sc_ncand(self.mw, self.mh)
        self.costs = np.empty(self.n + 1, f32)
        assert self.d.sc_costs(self.mw, self.mh, fp(self.costs)) == self.n + 1

    def mode(self, m):
        """0 exact, 1 fast portable, 2 fast with AVX2 (when the CPU has it); returns 1 when AVX2 kernels run"""
        return self.d.sc_set_fast(m)

    def field(self):
        return np.empty((4, self.mh, self.mw), f32)

    def c422to444(self, c):
        c = cf(c)
        o = np.empty(c.shape[:-1] + (2 * c.shape[-1],), f32)
        self.d.sc_c422to444(fp(c), c.shape[-1], c.size // c.shape[-1], fp(o))
        return o

    def me(self, y, fir):
        y = cf(y)
        o = np.empty((self.mh, self.mw), f32)
        self.d.sc_me_picture(fp(y), y.shape[1], y.shape[0], fir, fp(o))
        return o

    def pcorr(self, Bp, Bc):
        """candidates (n, 2, mh, mw), prefiltered picture of Bc, global (dx, dy, height, vx, vy)"""
        Bp, Bc = cf(Bp), cf(Bc)
        h, w = Bc.shape
        c = np.empty((self.d.sc_ncand(w, h), 2, h, w), f32)
        hp, glob = np.empty((h, w), f32), np.empty(5, f32)
        self.d.sc_pcorr(fp(Bp), fp(Bc), w, h, fp(c), fp(hp), fp(glob))
        return c, hp, glob

    def refine(self, cur, other, cands, neg):
        cur, other, cands = cf(cur), cf(other), cf(cands)
        o = np.empty_like(cands)
        self.d.sc_refine(fp(cur), fp(other), cur.shape[1], cur.shape[0], fp(cands), len(cands), int(neg), fp(o))
        return o

    def error(self, cur, other, v):
        cur, other = cf(cur), cf(other)
        o = np.empty_like(cur)
        self.d.sc_error(fp(cur), fp(other), cur.shape[1], cur.shape[0], fp(cf(v[0])), fp(cf(v[1])), fp(o))
        return o

    def errors(self, cur, other, vecs):
        """errors of every vector and the zero vector (last): (n + 1, mh, mw)"""
        zero = np.zeros((2,) + np.shape(cur), f32)
        return np.stack([self.error(cur, other, v) for v in list(vecs) + [zero]])

    def assign(self, rf, ef, rb, eb, bias=None):
        bias = self.costs if bias is None else cf(bias)
        h, w = ef.shape[1:]
        F, B = np.empty((4, h, w), f32), np.empty((4, h, w), f32)
        single = rb is None
        self.d.sc_assign(fp(cf(rf)), fp(cf(ef)), None if single else fp(cf(rb)), None if single else fp(cf(eb)),
                         len(rf), w, h, fp(bias), int(single), fp(F), fp(B))
        return F, B

    def disorder_weight(self, vx, vy):
        vx, vy = cf(vx), cf(vy)
        o = np.empty_like(vx)
        self.d.sc_disorder_weight(fp(vx), fp(vy), vx.shape[1], vx.shape[0], fp(o))
        return o

    def fields(self, Ap, Ac, An, cand_f, cand_b, toward_b=0, refine=1, bias=None):
        bias = self.costs if bias is None else cf(bias)
        h, w = Ac.shape
        F, B = np.empty((4, h, w), f32), np.empty((4, h, w), f32)
        self.d.sc_fields(fp(None if Ap is None else cf(Ap)), fp(cf(Ac)), fp(cf(An)), w, h, fp(cf(cand_f)),
                         fp(None if cand_b is None else cf(cand_b)), len(cand_f), int(toward_b), int(refine), fp(bias),
                         fp(F), fp(B))
        return F, B

    def fields_luma(self, yp, yc, yn):
        H, W = yc.shape
        F, B = self.field(), self.field()
        self.d.sc_fields_luma(fp(None if yp is None else cf(yp)), fp(cf(yc)), fp(cf(yn)), W, H, fp(F), fp(B))
        return F, B

    def phase_pull(self, fP, fN, d, strength):
        """adjusted phase and (m, p, f, g2)"""
        info = (C.c_double * 5)()
        r = self.d.sc_phase_pull(fp(cf(fP)), fp(cf(fN)), fP.shape[2], fP.shape[1], d, strength, info)
        return r, np.array(info[:4])

    def pull_strength(self, rin, rout):
        return self.d.sc_phase_pull_strength(rin.numerator, rin.denominator, rout.numerator, rout.denominator)

    def build(self, P, N, MEP, MEN, d):
        P, N = cf(P), cf(N)
        h, w = P.shape[1:]
        o = np.empty((3, h, w), f32)
        self.d.sc_build(fp(P), fp(N), w, h, fp(cf(MEP)), fp(cf(MEN)), MEP.shape[2], MEP.shape[1], d, fp(o))
        return o

    def plan(self, n, rin, rout):
        """the filter's frame plan (default copy threshold): list of (k, p, d as Fraction, kind name)"""
        from fractions import Fraction
        cap = int(n * rout / rin) + 4
        buf = (C.c_longlong * (5 * cap))()
        cnt = self.d.sc_plan(n, rin.numerator, rin.denominator, rout.numerator, rout.denominator, cap, buf)
        assert cnt <= cap
        return [(buf[5 * i], buf[5 * i + 1], Fraction(buf[5 * i + 2], buf[5 * i + 3]), KINDS[buf[5 * i + 4]])
                for i in range(cnt)]


class Params(C.Structure):
    # MvCudaSplatParams; config: a splat::Config (bytes) or null = the defaults of the picture size
    _fields_ = [('width', C.c_int), ('height', C.c_int), ('exact', C.c_int), ('config', C.c_void_p),
                ('bits', C.c_int), ('sub_w', C.c_int), ('sub_h', C.c_int), ('chroma_centre', C.c_int)]


class CudaLib:
    """the CUDA host test entries of mvtools_cuda.dll, exact or fast, with the CPU library's configuration"""

    def __init__(self, path, cpu, exact):
        self.d = C.CDLL(os.path.abspath(path))
        self.d.mvcuda_splat_build_host.argtypes = [C.POINTER(Params), FP, FP, FP, FP, C.c_float, FP]
        self.d.mvcuda_splat_phase_pull_host.argtypes = [C.POINTER(Params), FP, FP, C.c_double, C.c_double,
                                                        C.POINTER(C.c_double), C.POINTER(C.c_double)]
        self.cpu = cpu
        self.exact = exact
        self.name = 'cuda ' + ('exact' if exact else 'fast')
        self.prm = Params(cpu.W, cpu.H, int(exact), C.cast(cpu.cfg, C.c_void_p), 10, 1, 0, 0)

    def _ok(self, rc, fn):
        if rc:
            raise RuntimeError(f'mvcuda_splat_{fn}_host returned {rc}')

    def me(self, y, fir):
        o = np.empty((self.cpu.mh, self.cpu.mw), f32)
        self._ok(self.d.mvcuda_splat_me_picture_host(C.byref(self.prm), fp(cf(y)), fir, fp(o)), 'me_picture')
        return o

    def errors(self, cur, other, vecs):
        vecs = cf(vecs)
        o = np.empty((len(vecs) + 1,) + np.shape(cur), f32)
        self._ok(self.d.mvcuda_splat_errors_host(C.byref(self.prm), fp(cf(cur)), fp(cf(other)), fp(vecs), len(vecs),
                                                 fp(o)), 'errors')
        return o

    def disorder_weight(self, fields):
        """channel 3 of each field (vx, vy)"""
        fv = cf(np.stack([f[:2] for f in fields]))
        ch3 = np.empty((len(fields),) + np.shape(fields[0][0]), f32)
        self._ok(self.d.mvcuda_splat_disorder_weight_host(C.byref(self.prm), fp(fv), len(fields), fp(ch3)),
                 'disorder_weight')
        return ch3

    def phase_pull(self, fP, fN, d, strength):
        dd, info = C.c_double(), (C.c_double * 4)()
        self._ok(self.d.mvcuda_splat_phase_pull_host(C.byref(self.prm), fp(cf(fP)), fp(cf(fN)), d, strength,
                                                     C.byref(dd), info), 'phase_pull')
        return dd.value, np.array(info[:4])

    def pcorr(self, Bp, Bc):
        Bp, Bc = cf(Bp), cf(Bc)
        h, w = Bc.shape
        c = np.empty((self.cpu.n, 2, h, w), f32)
        hp, glob = np.empty((h, w), f32), np.empty(5, f32)
        self._ok(self.d.mvcuda_splat_pcorr_host(C.byref(self.prm), fp(Bp), fp(Bc), fp(c), fp(hp), fp(glob)), 'pcorr')
        return c, hp, glob

    def refine(self, cur, other, cands, neg):
        cands = cf(cands)
        o = np.empty_like(cands)
        self._ok(self.d.mvcuda_splat_refine_host(C.byref(self.prm), fp(cf(cur)), fp(cf(other)), fp(cands), len(cands),
                                                 int(neg), fp(o)), 'refine')
        return o

    def assign(self, rf, ef, rb, eb, bias=None):
        bias = self.cpu.costs if bias is None else cf(bias)
        h, w = ef.shape[1:]
        F, B = np.empty((4, h, w), f32), np.empty((4, h, w), f32)
        single = rb is None
        self._ok(self.d.mvcuda_splat_assign_host(C.byref(self.prm), fp(cf(rf)), fp(cf(ef)), None if single else fp(cf(rb)),
                                                 None if single else fp(cf(eb)), len(rf), fp(bias), fp(F), fp(B)), 'assign')
        return F, B

    def fields(self, Ap, Ac, An, cand_f, cand_b, toward_b=0, refine=1, bias=None):
        bias = self.cpu.costs if bias is None else cf(bias)
        h, w = Ac.shape
        F, B = np.empty((4, h, w), f32), np.empty((4, h, w), f32)
        self._ok(self.d.mvcuda_splat_fields_host(C.byref(self.prm), fp(None if Ap is None else cf(Ap)), fp(cf(Ac)),
                                                 fp(cf(An)), fp(cf(cand_f)), fp(None if cand_b is None else cf(cand_b)),
                                                 len(cand_f), int(toward_b), int(refine), fp(bias), fp(F), fp(B)),
                 'fields')
        return F, B

    def fields_luma(self, yp, yc, yn, reps=3):
        """fields of the centre picture and the best GPU times of reps runs (ms: ingest, pair, fields, per picture)"""
        F, B = self.cpu.field(), self.cpu.field()
        best = None
        for _ in range(reps):  # the GPU may be shared: keep the fastest run's times
            ms = np.zeros(4, f32)
            self._ok(self.d.mvcuda_splat_fields_luma_host(C.byref(self.prm), fp(None if yp is None else cf(yp)),
                                                          fp(cf(yc)), fp(cf(yn)), fp(F), fp(B), fp(ms)), 'fields_luma')
            if best is None or ms[3] < best[3]:
                best = ms
        return F, B, best

    def build(self, P, N, MEP, MEN, d):
        P, N = cf(P), cf(N)
        h, w = P.shape[1:]
        o = np.empty((3, h, w), f32)
        self._ok(self.d.mvcuda_splat_build_host(C.byref(self.prm), fp(P), fp(N), fp(cf(MEP)), fp(cf(MEN)), d, fp(o)),
                 'build')
        return o


def read_pictures(src, start, count):
    """count consecutive source pictures from start (10-bit 4:2:2): list of dict(Y, C codes), the size (w, h)"""
    n, _, w, h, pix = AU.probe(src)
    if pix != 'yuv422p10le':
        sys.exit(f'{src}: the stage inputs need a 10-bit 4:2:2 source, not {pix}')
    if start + count > n:
        sys.exit(f'{src}: {n} pictures, {start + count} needed')
    Y, Cc = AU.raw_frames(AU.source_script(src) + f'Trim({start}, -{count})\n', w, h)
    if len(Y) != count:
        sys.exit(f'got {len(Y)} pictures for {count}')
    return [dict(Y=Y[i], C=Cc[i]) for i in range(count)], (w, h)


def make_inputs(lib, src, start=0, count=5, phases=(0.5, 0.3), log=print):
    """every stage's input and exact CPU output on pictures start .. start + count - 1 of src (see the module text).
    Keys: y, P (3 planes 4:4:4) per picture; A, B per picture; pairs (c, c + 1): cand, hp, glob; centres c (both sides
    for 1 .. count - 2, forward side only for 0): rf, rb, ef, eb, asg (F, B), ch3 (F, B), fields (F, B),
    chain (F, B); builds (p, d): out."""
    import time
    t0 = time.perf_counter()
    raw, (w, h) = read_pictures(src, start, count)
    lib.set_size(w, h)
    lib.mode(0)  # the exact path
    I = dict(w=w, h=h, start=start, pics=raw, y={}, P={}, A={}, B={}, pair={}, ctr={}, build={})
    for c, r in enumerate(raw):
        y = ((r['Y'].astype(f32) - f32(512)) / f32(4)).astype(f32)
        cc = ((r['C'].astype(f32) - f32(512)) / f32(4)).astype(f32)
        I['y'][c] = y
        I['P'][c] = np.stack([y, lib.c422to444(cc[0]), lib.c422to444(cc[1])])
        I['A'][c], I['B'][c] = lib.me(y, 0), lib.me(y, 1)
    for c in range(count - 1):
        I['pair'][c] = dict(zip(('cand', 'hp', 'glob'), lib.pcorr(I['B'][c], I['B'][c + 1])))
    A = I['A']
    for c in range(count - 1):
        single = c == 0
        z = dict(single=single, cand_f=I['pair'][c]['cand'], cand_b=None if single else I['pair'][c - 1]['cand'])
        z['rf'] = lib.refine(A[c], A[c + 1], z['cand_f'], False)
        z['ef'] = lib.errors(A[c], A[c + 1], z['rf'])
        if not single:
            z['rb'] = lib.refine(A[c], A[c - 1], z['cand_b'], True)
            z['eb'] = lib.errors(A[c], A[c - 1], z['rb'])
        z['asg'] = lib.assign(z['rf'], z['ef'], z.get('rb'), z.get('eb'))
        z['ch3'] = [lib.disorder_weight(f[0], f[1]) for f in z['asg']]
        z['fields'] = lib.fields(None if single else A[c - 1], A[c], A[c + 1], z['cand_f'], z['cand_b'])
        z['chain'] = lib.fields_luma(None if single else I['y'][c - 1], I['y'][c], I['y'][c + 1])
        I['ctr'][c] = z
    for p in range(1, count - 2):
        for d in phases:
            I['build'][(p, d)] = lib.build(I['P'][p], I['P'][p + 1], I['ctr'][p]['fields'][0],
                                           I['ctr'][p + 1]['fields'][1], d)
    if log:
        log(f'stage inputs: pictures {start} .. {start + count - 1} of {os.path.basename(src)} ({w} x {h}), ME grid '
            f'{lib.mw} x {lib.mh}, {lib.n} candidates + zero; centres {sorted(I["ctr"])} (0: forward side only), '
            f'builds {sorted(I["build"])}; exact CPU path {time.perf_counter() - t0:.0f} s', flush=True)
    return I
