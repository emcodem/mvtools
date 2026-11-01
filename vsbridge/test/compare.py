"""compare.py: the same mvtools2 chains in AviSynth+ (avsmd5) and VapourSynth (vsmd5.py, mvtools_vsbridge) on the same
raw input with the same libmvtools2.so; the framemd5 of both must be identical (VAPOURSYNTH_PLAN.md, verification).

usage: python compare.py [chain ...]      (inside the Linux test setup; inputs from gen_inputs.py)
       python compare.py cpu | gpu | cf   (the CPU chains / the gpu_* chains / the cudafilters chains; default: cpu gpu)
env:   MVPF_PLUGIN_DIR  folder with libmvtools2.so and libmvtools_vsbridge.so (+ libmvtools_cuda.so for the GPU chains;
                        without it the gpu_* chains run the CPU fallback in both hosts and must still match)
       AVSMD5           avsmd5 binary (default ~/opt/bin/avsmd5)
       MVPF_THREADS     VapourSynth threads (default: VS's own)
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.expanduser('~/clips')
PLUGINS = os.environ.get('MVPF_PLUGIN_DIR', os.path.expanduser('~/build/test/vsplugins'))
AVSMD5 = os.environ.get('AVSMD5', os.path.expanduser('~/opt/bin/avsmd5'))
TMP = os.environ.get('TMPDIR', '/tmp')

# source name -> (AviSynth source, VapourSynth source)
SOURCES = {
    'uhd422p10': ('RawSource("%s/seg1_40.yuv", 3840, 2160, "YUV422P10", 30000, 1001)' % CLIPS,
                  "raw_source('%s/seg1_40.yuv', 3840, 2160, vs.YUV422P10, 30000, 1001)" % CLIPS),
    'hd420p8': ('RawSource("%s/seg1_40_hd420p8.yuv", 1920, 1080, "YV12", 30000, 1001)' % CLIPS,
                "raw_source('%s/seg1_40_hd420p8.yuv', 1920, 1080, vs.YUV420P8, 30000, 1001)" % CLIPS),
    'hd444p10': ('RawSource("%s/seg1_40_hd444p10.yuv", 1920, 1080, "YUV444P10", 30000, 1001)' % CLIPS,
                 "raw_source('%s/seg1_40_hd444p10.yuv', 1920, 1080, vs.YUV444P10, 30000, 1001)" % CLIPS),
    # planar RGB: the 4:4:4 planes relabelled (no chain uses it yet: mvtools2's MAnalyse accepts YUV only) without arithmetic (G=Y, B=U, R=V) on both sides
    'hdrgbp10': ('CombinePlanes(RawSource("%s/seg1_40_hd444p10.yuv", 1920, 1080, "YUV444P10", 30000, 1001), '
                 'planes="GBR", source_planes="YUV", pixel_type="RGBP10")' % CLIPS,
                 "core.std.ShufflePlanes(raw_source('%s/seg1_40_hd444p10.yuv', 1920, 1080, vs.YUV444P10, 30000, "
                 "1001), planes=[2, 0, 1], colorfamily=vs.RGB)" % CLIPS),
}

# common analysis prefixes (AviSynth, VapourSynth)
ANA = ('sup = MSuper(c, pel=2)\n'
       'bv = MAnalyse(sup, isb=true, blksize=16, overlap=8)\n'
       'fv = MAnalyse(sup, isb=false, blksize=16, overlap=8)\n',
       'sup = mv.MSuper(c, pel=2)\n'
       'bv = mv.MAnalyse(sup, isb=True, blksize=16, overlap=8)\n'
       'fv = mv.MAnalyse(sup, isb=False, blksize=16, overlap=8)\n')

# name -> (source, AviSynth body, VapourSynth body); the body leaves the result in `out`
CHAINS = {
    'vectors': ('hd420p8', ANA[0] + 'out = bv\n', ANA[1] + 'out = bv\n'),
    'storevect': ('hd420p8', ANA[0] + 'out = MStoreVect(bv, fv)\n', ANA[1] + 'out = mv.MStoreVect([bv, fv])\n'),
    'restorevect': ('hd420p8',
                    ANA[0] + 'st = MStoreVect(bv, fv)\nout = MCompensate(c, sup, MRestoreVect(st, index=1))\n',
                    ANA[1] + 'st = mv.MStoreVect([bv, fv])\nout = mv.MCompensate(c, sup, mv.MRestoreVect(st, index=1))\n'),
    'compensate': ('hd444p10', ANA[0] + 'out = MCompensate(c, sup, fv)\n', ANA[1] + 'out = mv.MCompensate(c, sup, fv)\n'),
    'degrain2': ('hd420p8',
                 'sup = MSuper(c, pel=2)\n'
                 'bv1 = MAnalyse(sup, isb=true, delta=1, blksize=16, overlap=8)\n'
                 'fv1 = MAnalyse(sup, isb=false, delta=1, blksize=16, overlap=8)\n'
                 'bv2 = MAnalyse(sup, isb=true, delta=2, blksize=16, overlap=8)\n'
                 'fv2 = MAnalyse(sup, isb=false, delta=2, blksize=16, overlap=8)\n'
                 'out = MDegrain2(c, sup, bv1, fv1, bv2, fv2, thSAD=400)\n',
                 'sup = mv.MSuper(c, pel=2)\n'
                 'bv1 = mv.MAnalyse(sup, isb=True, delta=1, blksize=16, overlap=8)\n'
                 'fv1 = mv.MAnalyse(sup, isb=False, delta=1, blksize=16, overlap=8)\n'
                 'bv2 = mv.MAnalyse(sup, isb=True, delta=2, blksize=16, overlap=8)\n'
                 'fv2 = mv.MAnalyse(sup, isb=False, delta=2, blksize=16, overlap=8)\n'
                 'out = mv.MDegrain2(c, sup, bv1, fv1, bv2, fv2, thSAD=400)\n'),
    'degrainN': ('hd444p10',
                 'sup = MSuper(c, pel=2)\nmvm = MAnalyse(sup, multi=true, delta=3, blksize=16, overlap=8)\n'
                 'out = MDegrainN(c, sup, mvm, 3, thSAD=300)\n',
                 'sup = mv.MSuper(c, pel=2)\nmvm = mv.MAnalyse(sup, multi=True, delta=3, blksize=16, overlap=8)\n'
                 'out = mv.MDegrainN(c, sup, mvm, 3, thSAD=300)\n'),
    'flow': ('hd420p8', ANA[0] + 'out = MFlow(c, sup, fv)\n', ANA[1] + 'out = mv.MFlow(c, sup, fv)\n'),
    'flowinter': ('hd420p8', ANA[0] + 'out = MFlowInter(c, sup, bv, fv, time=50, ml=70)\n',
                  ANA[1] + 'out = mv.MFlowInter(c, sup, bv, fv, time=50, ml=70)\n'),
    'flowblur': ('hd420p8', ANA[0] + 'out = MFlowBlur(c, sup, bv, fv, blur=15)\n',
                 ANA[1] + 'out = mv.MFlowBlur(c, sup, bv, fv, blur=15)\n'),
    'flowfps': ('hd444p10', ANA[0] + 'out = MFlowFps(c, sup, bv, fv, num=60000, den=1001)\n',
                ANA[1] + 'out = mv.MFlowFps(c, sup, bv, fv, num=60000, den=1001)\n'),
    'blockfps': ('hd420p8', ANA[0] + 'out = MBlockFps(c, sup, bv, fv, num=50, den=1, mode=0)\n',
                 ANA[1] + 'out = mv.MBlockFps(c, sup, bv, fv, num=50, den=1, mode=0)\n'),
    'mask': ('hd420p8', ANA[0] + 'out = MMask(c, fv, kind=0, ml=100)\n',
             ANA[1] + 'out = mv.MMask(c, fv, kind=0, ml=100)\n'),
    'scdetection': ('hd420p8', ANA[0] + 'out = MSCDetection(c, fv)\n', ANA[1] + 'out = mv.MSCDetection(c, fv)\n'),
    'show': ('hd420p8', ANA[0] + 'out = MShow(sup, fv)\n', ANA[1] + 'out = mv.MShow(sup, fv)\n'),
    'scalevect': ('hd420p8', ANA[0] + 'out = MCompensate(c, sup, MScaleVect(fv, scale=1.0))\n',
                  ANA[1] + 'out = mv.MCompensate(c, sup, mv.MScaleVect(fv, scale=1.0))\n'),
    'recalculate': ('hd420p8',
                    ANA[0] + 'sup1 = MSuper(c, pel=2, levels=1)\n'
                    'bv = MRecalculate(sup1, bv, blksize=8, overlap=4, thSAD=100)\n'
                    'out = MCompensate(c, sup, bv)\n',
                    ANA[1] + 'sup1 = mv.MSuper(c, pel=2, levels=1)\n'
                    'bv = mv.MRecalculate(sup1, bv, blksize=8, overlap=4, thSAD=100)\n'
                    'out = mv.MCompensate(c, sup, bv)\n'),
    'lin_cpu': ('uhd422p10', None, None),  # the stored Linux reference chain (lin_cpu.avs / lin_cpu.vpy)
}


def vs_body(avs):
    """The VapourSynth form of an AviSynth chain body in the subset used below: M* calls -> mv.M*, true / false."""
    vs = re.sub(r'\b(M[A-Z]\w*)\(', r'mv.\1(', avs)
    return re.sub(r'\bfalse\b', 'False', re.sub(r'\btrue\b', 'True', vs))


# GPU chains (Documentation/gpu.md): lite super (MSuper gpu=true) and the GPU paths of MAnalyse, MRecalculate,
# MCompensate, MDegrain2 and MFlowFps; normal frames in and out (token frames need cudafilters' CudaUpload)
GPU_ANA = ('sup = MSuper(c, pel=2, gpu=true)\n'
           'bv = MAnalyse(sup, isb=true, blksize=16, overlap=8, gpu=true)\n'
           'fv = MAnalyse(sup, isb=false, blksize=16, overlap=8, gpu=true)\n')
GPU_CHAINS = {
    'gpu_vectors': ('hd420p8', GPU_ANA + 'out = bv\n'),
    'gpu_vectors_normalsuper': ('hd444p10', 'sup = MSuper(c, pel=2)\n'
                                'out = MAnalyse(sup, isb=false, blksize=16, overlap=8, gpu=true)\n'),
    'gpu_compensate': ('hd444p10', GPU_ANA + 'out = MCompensate(c, sup, fv, gpu=true)\n'),
    'gpu_degrain2': ('hd420p8',
                     'sup = MSuper(c, pel=2, gpu=true)\n'
                     'bv1 = MAnalyse(sup, isb=true, delta=1, blksize=16, overlap=8, gpu=true)\n'
                     'fv1 = MAnalyse(sup, isb=false, delta=1, blksize=16, overlap=8, gpu=true)\n'
                     'bv2 = MAnalyse(sup, isb=true, delta=2, blksize=16, overlap=8, gpu=true)\n'
                     'fv2 = MAnalyse(sup, isb=false, delta=2, blksize=16, overlap=8, gpu=true)\n'
                     'out = MDegrain2(c, sup, bv1, fv1, bv2, fv2, thSAD=400, gpu=true)\n'),
    'gpu_recalculate': ('hd420p8', GPU_ANA + 'sup1 = MSuper(c, pel=2, levels=1, gpu=true)\n'
                        'bv = MRecalculate(sup1, bv, blksize=8, overlap=4, thSAD=100, gpu=true)\n'
                        'out = MCompensate(c, sup, bv, gpu=true)\n'),
    'gpu_flowfps': ('uhd422p10', 'sup = MSuper(c, pel=2, hpad=16, vpad=16, rfilter=4, gpu=true)\n'
                    'bv = MAnalyse(sup, isb=true, blksize=32, overlap=16, search=3, truemotion=true, chroma=true, gpu=true)\n'
                    'fv = MAnalyse(sup, isb=false, blksize=32, overlap=16, search=3, truemotion=true, chroma=true, gpu=true)\n'
                    'out = MFlowFps(c, sup, bv, fv, num=60000, den=1001, mask=2, ml=200, gpu=true)\n'),
    'lin_gpu': ('uhd422p10', None),  # the stored Linux GPU reference chain (lin_gpu.avs / lin_gpu.vpy)
}
CPU_NAMES = list(CHAINS)
for _name, (_src, _body) in GPU_CHAINS.items():
    CHAINS[_name] = (_src, _body, None if _body is None else vs_body(_body))

# cudafilters chains (libcudafilters.so next to libmvtools2.so; namespace cuda as `cf`): token frames from CudaUpload
# through GPU filters to CudaDownload. Under VapourSynth the CPU fallbacks that are AviSynth core filters are strict
# stand-ins (the GPU path must run; a fallback frame is an error), so without libmvtools_cuda.so these chains fail under VS.
# Bodies in AviSynth form; Cuda* -> cf.Cuda*, M* -> mv.M*, REV(x) -> Reverse(x) / core.std.Reverse(x).
CF_CHAINS = {
    'cf_roundtrip': ('uhd422p10', 'out = CudaDownload(CudaUpload(c))\n'),
    'cf_merge': ('hd420p8', 'out = CudaDownload(CudaMerge(CudaUpload(c), CudaUpload(REV(c)), weight=0.3))\n'),
    'cf_merge_avg': ('hd444p10', 'out = CudaDownload(CudaMerge(CudaUpload(c), CudaUpload(REV(c))))\n'),
    'cf_mergeluma': ('uhd422p10', 'out = CudaDownload(CudaMergeLuma(CudaUpload(c), CudaUpload(REV(c)), weight=0.7))\n'),
    'cf_mergechroma': ('hd420p8', 'out = CudaDownload(CudaMergeChroma(CudaUpload(c), CudaUpload(REV(c)), weight=0.6))\n'),
    'cf_degrain2': ('hd420p8',
                    'u = CudaUpload(c)\n'
                    'sup = MSuper(u, pel=2, gpu=true)\n'
                    'bv1 = MAnalyse(sup, isb=true, delta=1, blksize=16, overlap=8, gpu=true)\n'
                    'fv1 = MAnalyse(sup, isb=false, delta=1, blksize=16, overlap=8, gpu=true)\n'
                    'bv2 = MAnalyse(sup, isb=true, delta=2, blksize=16, overlap=8, gpu=true)\n'
                    'fv2 = MAnalyse(sup, isb=false, delta=2, blksize=16, overlap=8, gpu=true)\n'
                    'out = CudaDownload(MDegrain2(u, sup, bv1, fv1, bv2, fv2, thSAD=400, gpu=true))\n'),
    'cf_flowfps_merge': ('uhd422p10',
                         'u = CudaUpload(c)\n'
                         'sup = MSuper(u, pel=2, hpad=16, vpad=16, gpu=true)\n'
                         'bv = MAnalyse(sup, isb=true, blksize=32, overlap=16, gpu=true)\n'
                         'fv = MAnalyse(sup, isb=false, blksize=32, overlap=16, gpu=true)\n'
                         'f = MFlowFps(u, sup, bv, fv, num=60000, den=1001, gpu=true)\n'
                         'out = CudaDownload(CudaMerge(f, CudaUpload(REV(MFlowFps(c, MSuper(c, pel=2, hpad=16, vpad=16), '
                         'MAnalyse(MSuper(c, pel=2, hpad=16, vpad=16), isb=true, blksize=32, overlap=16), '
                         'MAnalyse(MSuper(c, pel=2, hpad=16, vpad=16), isb=false, blksize=32, overlap=16), '
                         'num=60000, den=1001))), weight=0.4))\n'),
    # cudafilters whose CPU fallback exists in Linux AviSynth+ (VS: the bridge's stand-ins; the others: cf_chains.py)
    # BFF(x): AssumeBFF in AviSynth+ (avsmd5's RawSource has no parity: Bob crashes in C_VideoFilter::GetParity),
    # nothing in VS (the bridge reports BFF for VS clips)
    'cf_bob': ('hd420p8', 'out = CudaDownload(CudaBob(CudaUpload(BFF(c)), 0.0, 0.5))\n'),
    'cf_bob_h': ('uhd422p10', 'out = CudaDownload(CudaBob(CudaUpload(BFF(c)), height=1080))\n'),
    'cf_tsoften': ('hd444p10', 'out = CudaDownload(CudaTemporalSoften(CudaUpload(c), 2, 4, 8, scenechange=15, mode=2))\n'),
    'cf_gauss': ('hd420p8', 'out = CudaDownload(CudaGaussResize(CudaUpload(c), 1280, 720, p=5.0))\n'),
    'cf_bilinear': ('hd444p10', 'out = CudaDownload(CudaBilinearResize(CudaUpload(c), 960, 540, src_left=2.5))\n'),
    'cf_binomial0': ('hd420p8', 'out = CudaDownload(CudaQtgmcBinomial0(CudaUpload(c)))\n'),
}


def cf_vs_body(avs):
    vs = re.sub(r'\bBFF\(', '(', re.sub(r'\bREV\(', 'core.std.Reverse(', vs_body(avs)))
    return re.sub(r'\b(Cuda\w*)\(', r'cf.\1(', vs)


for _name, (_src, _body) in CF_CHAINS.items():
    CHAINS[_name] = (_src, _body.replace('REV(', 'Reverse(').replace('BFF(', 'AssumeBFF('), cf_vs_body(_body))


def write(path, text):
    with open(path, 'w') as fh:
        fh.write(text)


def run_chain(name):
    src, avs_body, vs_body = CHAINS[name]
    avs_src, vs_src = SOURCES[src]
    avs = os.path.join(TMP, 'vsb_%s.avs' % name)
    vpy = os.path.join(TMP, 'vsb_%s.vpy' % name)
    if avs_body is None:
        avs_text = open(os.path.expanduser('~/build/test/%s.avs' % name)).read()
        avs_text = avs_text.replace('/home/emcodem/build/test/plugins', PLUGINS)
        write(avs, avs_text)
        vpy = os.path.join(HERE, '%s.vpy' % name)
    else:
        cf = name in CF_CHAINS
        write(avs, 'LoadPlugin("%s/libmvtools2.so")\n%sc = %s\n%sreturn out\n'
              % (PLUGINS, 'LoadPlugin("%s/libcudafilters.so")\n' % PLUGINS if cf else '', avs_src, avs_body))
        write(vpy, 'import vapoursynth as vs\nfrom vstest import core, load_mvtools, load_cudafilters, raw_source\n'
                   'load_mvtools()\n%smv = core.mvpf\nc = %s\n%sout.set_output()\n'
              % ('load_cudafilters()\ncf = core.cuda\n' if cf else '', vs_src, vs_body))
    a_md5, v_md5 = avs[:-4] + '.avs.md5', vpy.replace('.vpy', '') + '.vs.md5'
    if avs_body is None:
        v_md5 = os.path.join(TMP, 'vsb_%s.vs.md5' % name)
    env = dict(os.environ, MVPF_PLUGIN_DIR=PLUGINS)
    t0 = time.time()
    a = subprocess.run([AVSMD5, avs, a_md5], capture_output=True, text=True)
    t1 = time.time()
    cmd = [sys.executable, os.path.join(HERE, 'vsmd5.py'), vpy, v_md5]
    if os.environ.get('MVPF_THREADS'):
        cmd.append(os.environ['MVPF_THREADS'])
    v = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    t2 = time.time()
    if a.returncode or v.returncode:
        return 'ERROR', 'avs: %s | vs: %s' % (a.stderr.strip()[-300:], v.stderr.strip()[-300:]), t1 - t0, t2 - t1
    al = open(a_md5).read().splitlines()[3:]
    vl = open(v_md5).read().splitlines()[3:]
    if al == vl:
        return 'same', '%d frames' % len(al), t1 - t0, t2 - t1
    diff = sum(1 for x, y in zip(al, vl) if x != y) + abs(len(al) - len(vl))
    first = next((i for i, (x, y) in enumerate(zip(al, vl)) if x != y), min(len(al), len(vl)))
    return 'DIFF', '%d of %d lines differ, first at line %d' % (diff, max(len(al), len(vl)), first), t1 - t0, t2 - t1


def main():
    names = []
    for a in sys.argv[1:] or ['cpu', 'gpu']:
        names += CPU_NAMES if a == 'cpu' else list(GPU_CHAINS) if a == 'gpu' else list(CF_CHAINS) if a == 'cf' else [a]
    bad = 0
    for name in names:
        status, info, ta, tv = run_chain(name)
        bad += status != 'same'
        print('%-13s %-5s avs %6.2f s  vs %6.2f s  %s' % (name, status, ta, tv, info), flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
