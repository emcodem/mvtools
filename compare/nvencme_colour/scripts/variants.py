"""Colour-mode variants (DTL's proposal: one grey ME run per channel + per-cell selection by Y+U+V SAD).

Same uc_FRC chain as compare/nvencme_frc (analysis 32/16 on RemoveGrain(22), MRecalculate 16/8, MFlowFps 24000/1001
mask=2), CPU MSuper / MRecalculate / MFlowFps (bit-identical to the GPU ones), plugins of the 2026-11-13 bundle.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# plugins folder of a GPU bundle (mvtools2.dll, cudafilters.dll, RgTools.dll), L-SMASH, the prototype DLL, the clip
BUNDLE = os.environ.get('NVME_BUNDLE', r'C:\path\to\mvtools_gpu_qtgmc_2026-11-13\plugins')
LSMASH = os.environ.get('NVME_LSMASH', r'C:\path\to\plugins\LSMASHSource.dll')
PLUGIN = os.environ.get('NVME_PLUGIN', os.path.normpath(os.path.join(HERE, '..', '..', 'nvencme_frc', 'proto', 'nvencme_proto.dll')))
SEGFILE = os.environ.get('NVME_SEGFILE', r'C:\path\to\clips\seg1_wires_pole.mp4')
RGTOOLS = os.environ.get('NVME_RGTOOLS', os.path.join(BUNDLE, 'RgTools.dll'))

# key, label, analysis ('exact' | 'nvof' | (dx12, colour)), extra environment
ALL = [
    ('exact', 'exact search', 'exact', {}),
    ('medium', 'NVOFA medium', 'nvof', {}),
    ('p4', 'NVENC SDK P4 luma', (0, ''), {}),
    ('p4ynb', 'NVENC SDK P4 luma + nb select', (0, 'ynb'), {}),
    ('p4yuv', 'NVENC SDK P4 Y/U/V select', (0, 'yuv'), {}),
    ('p4rgb', 'NVENC SDK P4 R/G/B select', (0, 'rgb'), {}),
    ('p4yuvnb', 'NVENC SDK P4 Y/U/V + nb select', (0, 'yuvnb'), {}),
    # second round (NVENC SDK only): encoder QP, smoothness term, swapped run + forward-backward check
    ('p4qp10', 'P4 luma QP 10', (0, ''), {'NVME_QP': '10'}),
    ('p4qp30', 'P4 luma QP 30', (0, ''), {'NVME_QP': '30'}),
    ('p4qp51', 'P4 luma QP 51', (0, ''), {'NVME_QP': '51'}),
    ('p4ynbl1', 'P4 ynb lambda 1000', (0, 'ynb'), {'NVME_LAMBDA': '1000'}),
    ('p4ynbl4', 'P4 ynb lambda 4000', (0, 'ynb'), {'NVME_LAMBDA': '4000'}),
    ('p4ynbl16', 'P4 ynb lambda 16000', (0, 'ynb'), {'NVME_LAMBDA': '16000'}),
    ('p4yuvnbl4', 'P4 yuvnb lambda 4000', (0, 'yuvnb'), {'NVME_LAMBDA': '4000'}),
    ('p4ysw', 'P4 y + swapped', (0, 'y'), {'NVME_SWAP': '1'}),
    ('p4yswfb', 'P4 y + swapped, fb 1000', (0, 'y'), {'NVME_SWAP': '1', 'NVME_FBPEN': '1000'}),
    ('p4ynbswfbl4', 'P4 ynb + swapped, fb 1000, lambda 4000', (0, 'ynb'), {'NVME_SWAP': '1', 'NVME_FBPEN': '1000', 'NVME_LAMBDA': '4000'}),
    # the plugin (nvencmv.dll in NVME_BUNDLE): CudaNVENCAnalyse with these arguments
    ('cn_plain', 'CudaNVENCAnalyse qp=-1, swap=false', ('cuda', 'qp=-1, swap=false'), {}),
    ('cn_def', 'CudaNVENCAnalyse defaults (QP 10 + swapped run)', ('cuda', ''), {}),
    ('cn_yuv', 'CudaNVENCAnalyse channels yuv', ('cuda', 'channels="yuv"'), {}),
    # third round
    ('p4qp38', 'P4 luma QP 38', (0, ''), {'NVME_QP': '38'}),
    ('p4qp44', 'P4 luma QP 44', (0, ''), {'NVME_QP': '44'}),
    ('p4ynbl100', 'P4 ynb lambda 100', (0, 'ynb'), {'NVME_LAMBDA': '100'}),
    ('p4ynbl250', 'P4 ynb lambda 250', (0, 'ynb'), {'NVME_LAMBDA': '250'}),
    ('p4yswqp38', 'P4 y + swapped, QP 38', (0, 'y'), {'NVME_SWAP': '1', 'NVME_QP': '38'}),
    ('p4yswqp44', 'P4 y + swapped, QP 44', (0, 'y'), {'NVME_SWAP': '1', 'NVME_QP': '44'}),
    ('p4t', 'P4 luma temporal hints', (0, ''), {'NVME_TEMPORAL': '1'}),
    ('p4tqp38', 'P4 luma temporal hints, QP 38', (0, ''), {'NVME_TEMPORAL': '1', 'NVME_QP': '38'}),
    ('p4qp0', 'P4 luma QP 0', (0, ''), {'NVME_QP': '0'}),
    ('p4yswqp10', 'P4 y + swapped, QP 10', (0, 'y'), {'NVME_SWAP': '1', 'NVME_QP': '10'}),
    ('dx8', 'D3D12 8x8 luma', (8, ''), {}),
    ('dx8ynb', 'D3D12 8x8 luma + nb select', (8, 'ynb'), {}),
    ('dx8yuv', 'D3D12 8x8 Y/U/V select', (8, 'yuv'), {}),
    ('dx8rgb', 'D3D12 8x8 R/G/B select', (8, 'rgb'), {}),
    ('dx8yuvnb', 'D3D12 8x8 Y/U/V + nb select', (8, 'yuvnb'), {}),
]
BY_KEY = {v[0]: v for v in ALL}


def head():
    s = ('LoadPlugin("%s")\nLoadPlugin("%s")\nLoadPlugin("%s\\mvtools2.dll")\nLoadPlugin("%s\\cudafilters.dll")\n'
         'LoadPlugin("%s")\n' % (LSMASH, RGTOOLS, BUNDLE, BUNDLE, PLUGIN))
    if os.path.exists(os.path.join(BUNDLE, 'nvencmv.dll')):
        s += 'LoadPlugin("%s\\nvencmv.dll")\n' % BUNDLE
    return s


def analysis(key, clip):
    """bv / fv lines (32/16 blocks) for the analysis of `clip` (a script variable)"""
    an = BY_KEY[key][2]
    lines = []
    for d, isb in (('bv', 'true'), ('fv', 'false')):
        if an == 'exact':
            lines.append('%s = MAnalyse(supf, isb=%s, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true)' % (d, isb))
        elif an == 'nvof':
            lines.append('%s = CudaNVOFAnalyse(%s, isb=%s, blksize=32, overlap=16, pel=2, hpad=16, vpad=16, chroma=true, preset=2)' % (d, clip, isb))
        elif an[0] == 'cuda':
            lines.append('%s = CudaNVENCAnalyse(%s, isb=%s, blksize=32, overlap=16, pel=2, hpad=16, vpad=16%s)' % (d, clip, isb, (', ' + an[1]) if an[1] else ''))
        else:
            lines.append('%s = NVENCMEProto(%s, isb=%s, blksize=32, overlap=16, pel=2, hpad=16, vpad=16, preset=4, dx12=%d)' % (d, clip, isb, an[0]))
    if an == 'exact':
        lines.insert(0, 'supf = MSuper(%s, pel=2, hpad=16, vpad=16, rfilter=4)' % clip)
    return '\n'.join(lines) + '\n'


RECALC = ('bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true)\n'
          'fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true)\n')


def script_text(key, tail='out'):
    """tail: 'out' (MFlowFps, frames 8..247), 'vec' (Interleave(bv, fv) after MRecalculate), 'skip' (interpolated odd
    frames from the even ones, stacked over the true odd frames, 8 bit)"""
    s = head() + 'm = LWLibavVideoSource("%s", threads=4, cachefile="%s.lwi")\n' % (SEGFILE, SEGFILE)
    if tail == 'skip':
        s += 'full = m\nm = m.SelectEven()\n'
    s += 'pre = m.RemoveGrain(22)\nsup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1)\n'
    s += analysis(key, 'pre') + RECALC
    if tail == 'out':
        s += 'out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false)\nreturn out.Trim(8, 247).Prefetch(4)\n'
    elif tail == 'vec':
        s += 'return Interleave(bv, fv).Prefetch(4)\n'
    else:
        s += ('it = MFlowInter(m, sup, bv, fv, time=50, ml=200, blend=false)\n'
              'truth = full.SelectOdd()\n'
              'n = min(it.FrameCount, truth.FrameCount)\n'
              'return Interleave(it.Trim(0, -n), truth.Trim(0, -n)).ConvertBits(8).Prefetch(4)\n')
    return s


def write_script(work, key, text=None):
    path = os.path.join(work, 'flow_%s.avs' % key)
    with open(path, 'w') as fh:
        fh.write(script_text(key) if text is None else text)
    return path


def env_for(key):
    env = dict(os.environ)
    for k in ('NVME_COLOUR', 'NVME_CHROMA', 'NVME_NOINTRA', 'NVME_P16', 'NVME_PSEUDO', 'NVME_STATS', 'NVME_QP', 'NVME_LAMBDA', 'NVME_FBPEN', 'NVME_SWAP', 'NVME_TEMPORAL'):
        env.pop(k, None)
    an = BY_KEY[key][2]
    if isinstance(an, tuple) and an[0] != 'cuda' and an[1]:
        env['NVME_COLOUR'] = an[1]
    env.update(BY_KEY[key][3])
    return env
