"""stand_ins.py: the VideoInfo of the bridge's stand-ins (vsbridge/bridge.cpp, stand_in_specs) against the AviSynth+
filters they stand in for. A cudafilter takes its VideoInfo from its CPU fallback, so the stand-in must report exactly
what the AviSynth+ filter reports (size, frames, rate, pixel type, image_type), or the same error.

usage (Windows, AviSynth+ 3.7.5 with masktools2, RgTools, plugins_JPSDR, QTGMC.avsi):
        python stand_ins.py ref          -> stand_ins.ref next to this file (avs_info.py: AviSynth+'s C API)
        env AVS_PLUGINS (folder with MaskTools/, RgTools/, JPSDR/, avsi/ as in FFAStrans' avs_plugins)
usage (VapourSynth, the bridge's test setup):
        python stand_ins.py [case ...]   -> each case's cudafilter under VS with MVBRIDGE_DEBUG=1, compared with the ref
        env MVPF_PLUGIN_DIR (libcudafilters.so + libmvtools_vsbridge.so)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'stand_ins.ref')

# inputs: AviSynth BlankClip arguments / VapourSynth BlankClip arguments
INPUTS = {
    'hd420p8': ('width=1920, height=1080, pixel_type="YV12", length=10, fps=30000, fps_denominator=1001',
                'width=1920, height=1080, format=vs.YUV420P8, length=10, fpsnum=30000, fpsden=1001'),
    'hd444p10': ('width=1280, height=720, pixel_type="YUV444P10", length=10, fps=25',
                 'width=1280, height=720, format=vs.YUV444P10, length=10, fpsnum=25, fpsden=1'),
    'odd420p8': ('width=1920, height=1082, pixel_type="YV12", length=10, fps=25',
                 'width=1920, height=1082, format=vs.YUV420P8, length=10, fpsnum=25, fpsden=1'),
}

# case -> (input, stand-in name as logged, AviSynth call on c, VapourSynth call on c (cf = core.cuda))
CASES = {
    'bob': ('hd420p8', 'Bob', 'Bob(c)', 'cf.CudaBob(c)'),
    'bob_args': ('hd420p8', 'Bob', 'Bob(c, 0.0, 0.5, height=720)', 'cf.CudaBob(c, b=0.0, c=0.5, height=720)'),
    'bob_444p10': ('hd444p10', 'Bob', 'Bob(c, height=1440)', 'cf.CudaBob(c, height=1440)'),
    'bob_err420': ('odd420p8', 'Bob', 'Bob(c)', 'cf.CudaBob(c)'),
    'bob_errheight': ('hd420p8', 'Bob', 'Bob(c, height=721)', 'cf.CudaBob(c, height=721)'),
    'temporalsoften': ('hd420p8', 'TemporalSoften', 'TemporalSoften(c, 2, 4, 8, scenechange=15, mode=2)',
                       'cf.CudaTemporalSoften(c, 2, 4, 8, scenechange=15, mode=2)'),
    'gauss': ('hd420p8', 'GaussResize', 'GaussResize(c, 1280, 720, 0.0, 0.0, 1920.0, 1080.0, p=5.0)',
              'cf.CudaGaussResize(c, 1280, 720, src_left=0.0, src_top=0.0, src_width=1920.0, src_height=1080.0, p=5.0)'),
    'gauss_10': ('hd444p10', 'GaussResize', 'GaussResize(c, 1921, 1081)', 'cf.CudaGaussResize(c, 1921, 1081)'),
    'gauss_err': ('hd420p8', 'GaussResize', 'GaussResize(c, 1281, 720)', 'cf.CudaGaussResize(c, 1281, 720)'),
    'bilinear': ('hd420p8', 'BilinearResize', 'BilinearResize(c, 960, 540, src_left=2.5)',
                 'cf.CudaBilinearResize(c, 960, 540, src_left=2.5)'),
    'removegrain': ('hd420p8', 'RemoveGrain', 'RemoveGrain(c, mode=12, modeU=-1)', 'cf.CudaRemoveGrain(c, 12, -1)'),
    'removegrain_10': ('hd444p10', 'RemoveGrain', 'RemoveGrain(c, mode=20)', 'cf.CudaRemoveGrain(c, 20)'),
    'nnedi3_m2': ('hd420p8', 'nnedi3', 'nnedi3(c, field=-2)', 'cf.CudaNNEDI3(c, field=-2)'),
    'nnedi3_dh': ('hd420p8', 'nnedi3', 'nnedi3(c, field=-1, dh=true)', 'cf.CudaNNEDI3(c, field=-1, dh=True)'),
    'nnedi3_1': ('hd444p10', 'nnedi3', 'nnedi3(c, field=1)', 'cf.CudaNNEDI3(c, field=1)'),
    'nnedi3_3': ('hd444p10', 'nnedi3', 'nnedi3(c, field=3)', 'cf.CudaNNEDI3(c, field=3)'),
    'nnedi3_err': ('hd420p8', 'nnedi3', 'nnedi3(c, field=2, dh=true)', 'cf.CudaNNEDI3(c, field=2, dh=True)'),
    'mt_logic': ('hd420p8', 'mt_logic', 'mt_logic(c, c, "max", U=3, V=3)', 'cf.CudaLogic(c, c, mode="max", U=3, V=3)'),
    'mt_clamp': ('hd444p10', 'mt_clamp', 'mt_clamp(c, c, c, 2, 3, U=3, V=3)',
                 'cf.CudaClamp(c, c, c, overshoot=2, undershoot=3, U=3, V=3)'),
    'mt_expand': ('hd420p8', 'mt_expand', 'mt_expand(c, mode="vertical", U=3, V=3)',
                  'cf.CudaExpand(c, mode="vertical", U=3, V=3)'),
    'mt_inpand': ('hd444p10', 'mt_inpand', 'mt_inpand(c, mode="vertical")', 'cf.CudaInpand(c, mode="vertical")'),
    'kobs': ('hd420p8', 'QTGMC_KeepOnlyBobShimmerFixes', 'QTGMC_KeepOnlyBobShimmerFixes(c, c, 1, true)',
             'cf.CudaKeepOnlyBobShimmerFixes(c, c, rep=1, chroma=True)'),
}

FIELDS = ('width', 'height', 'pixel_type', 'fps_numerator', 'fps_denominator', 'num_frames', 'image_type')


def make_ref():
    sys.path.insert(0, HERE)
    from avs_info import video_info
    base = os.environ.get('AVS_PLUGINS', r'C:\FFAStrans\Processors\avs_plugins').replace('\\', '/')
    head = ('LoadPlugin("%s/MaskTools/x64/masktools2.dll")\nLoadPlugin("%s/RgTools/x64/RgTools.dll")\n'
            'LoadPlugin("%s/JPSDR/x64/plugins_JPSDR.dll")\nImport("%s/avsi/QTGMC.avsi")\n' % (base, base, base, base))
    lines = []
    for case, (inp, _, avs, _) in CASES.items():
        script = head + 'c = BlankClip(%s).KillAudio()\n%s\n' % (INPUTS[inp][0], avs)
        try:
            vi = video_info(script)
            lines.append('%s %s' % (case, ' '.join(str(vi[k]) for k in FIELDS)))
        except RuntimeError as e:
            lines.append('%s ERROR %s' % (case, str(e).strip().splitlines()[0]))
        print(lines[-1], flush=True)
    with open(REF, 'w', newline='\n') as f:
        f.write('# AviSynth+ 3.7.5 (Windows) with masktools2 2.2.30, RgTools, plugins_JPSDR, QTGMC.avsi; made by '
                'stand_ins.py ref\n# case width height pixel_type fps_numerator fps_denominator num_frames image_type '
                '| ERROR message\n' + '\n'.join(lines) + '\n')


def vs_case(case):
    inp, name, _, vs_call = CASES[case]
    script = ('import sys\nsys.path.insert(0, %r)\nimport vapoursynth as vs\nfrom vstest import core, load_cudafilters\n'
              'load_cudafilters()\ncf = core.cuda\nc = core.std.BlankClip(%s)\n'
              'try:\n    %s\nexcept vs.Error as e:\n    print("VSERROR", str(e).strip().splitlines()[-1])\n'
              % (HERE, INPUTS[inp][1], vs_call))
    env = dict(os.environ, MVBRIDGE_DEBUG='1')
    r = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, cwd=HERE)
    m = None
    for line in r.stderr.splitlines():
        mm = re.match(r'mvtools_vsbridge: stand-in (\S+): (\d+)x(\d+) pixel_type (-?\d+) fps (\d+)/(\d+) frames (\d+) '
                      r'image_type (-?\d+)$', line)
        if mm and mm.group(1) == name:
            m = mm  # the last one of that name: the one the cudafilter takes its VideoInfo from
    if m:
        return ' '.join(m.group(i) for i in range(2, 9))
    err = next((l[8:] for l in r.stdout.splitlines() if l.startswith('VSERROR ')), None)
    return 'ERROR ' + err if err else 'NONE rc=%d %s' % (r.returncode, r.stderr.strip()[-300:])


def vs_run(names):
    ref = {}
    with open(REF) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                k, v = line.rstrip('\n').split(' ', 1)
                ref[k] = v
    bad = 0
    for case in names:
        got = vs_case(case)
        want = ref.get(case, '?')
        # errors: the AviSynth+ message must appear in the VapourSynth one (VS adds the function name in front)
        ok = got == want or (want.startswith('ERROR ') and got.startswith('ERROR ') and want[6:] in got)
        bad += not ok
        print('%-15s %s  %s%s' % (case, 'same' if ok else 'DIFF', got, '' if ok else '   (AviSynth+: %s)' % want),
              flush=True)
    print('%d of %d cases differ' % (bad, len(names)))
    return 1 if bad else 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['ref']:
        make_ref()
    else:
        sys.exit(vs_run(args or list(CASES)))
