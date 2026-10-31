"""Variants of the DTL DX12_ME comparison and the AviSynth scripts that produce them.

Every variant is the uc_FRC chain (compare/nvofa_frc/scripts/avs/frc_flow.avs: MRecalculate 16/8 thSAD=100 truemotion,
MFlowFps 24000/1001 mask=2 ml=200, output Trim(8, 247), Prefetch(3)) on seg1_wires_pole.mp4 converted to 8-bit 4:2:0
(ConvertBits(8), ConvertToYV12: the only format of DTL's DX12_ME path). Only the source of the first vectors differs:
- ours: profile\\build_nvofa\\mvtools2.dll, exact GPU search or NVOFA medium (MVCUDA_NVOFA=medium), MAnalyse 32/16;
- ours + prototype: NVENCMEProto(dx12=8|16) (profile\\build_nvme_proto, compare/nvencme_frc/proto) in place of MAnalyse;
- DTL: release r.2.7.46-a.29, DX12/AVX2/mvtools2.dll unchanged (profile\\dtl_dx12), the WHOLE chain in his DLL (his
  MSuper / MAnalyse / MRecalculate / MFlowFps; never loaded together with ours). His call for the DX12 search:
    MSuper(pre, mt=false, pel=4, pelrefine=false, chroma=true, levels=1)
    MAnalyse(super, blksize=8|16, isb=.., delta=1, chroma=true, optSearchOption=5|6, levels=1)
  optSearchOption=6 (SAD on the CPU) refuses pel 4 without refined planes unless UseSubShift=1 (MVAnalyse.cpp:210), so
  the =6 variants add UseSubShift=1. optSearchOption=5 (shader SAD) crashes in multithreaded use ("Previous GetFrame
  did not finished properly. There was a crash", MFlowFps, within the first 9..65 frames under Prefetch(3)); single-
  threaded it runs and is deterministic, so the =5 variants run with Prefetch(1). `dtl_cpu`: his DLL with the uc_FRC MAnalyse (CPU search, 32/16, pel 2).
  His MAnalyse loads Compute.cso from the process's current directory (DX::ReadData(L"Compute.cso")), so every run
  goes through run_dir(), which puts a copy there.
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = r'C:\dev\mvtools\profile\build_nvofa\mvtools2.dll'
DTL = os.environ.get('DTL_PLUGIN', r'C:\dev\mvtools\profile\dtl_dx12\mvtools2.dll')
PROTO = os.environ.get('NVME_PLUGIN', r'C:\dev\mvtools\profile\build_nvme_proto\nvencme_proto.dll')
TIMEOUT = 900  # seconds for one ffmpeg run; a stalled GPU path must not hang the comparison

SRC = r'''LoadPlugin("C:\dev\uconvert\dlls\LSMASHSource.dll")
LoadPlugin("C:\dev\uconvert\dlls\RgTools.dll")
LoadPlugin("%s")
%sSEGFILE = "C:\dev\mvtools\profile\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, threads=4, cachefile=SEGFILE + ".lwi").ConvertBits(8).ConvertToYV12()
pre = m.RemoveGrain(22)
'''

# our chain, as frc_flow.avs
OURS_CHAIN = '''supf = MSuper(pre, pel=2, hpad=16, vpad=16, rfilter=4, gpu=true)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, gpu=true)
bv = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fv = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
'''
PROTO_CHAIN = '''sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, gpu=true)
bv = NVENCMEProto(pre, isb=true, blksize=%(b)d, overlap=%(o)d, pel=2, hpad=16, vpad=16, preset=4, half=false, levels=1, dx12=%(d)d)
fv = NVENCMEProto(pre, isb=false, blksize=%(b)d, overlap=%(o)d, pel=2, hpad=16, vpad=16, preset=4, half=false, levels=1, dx12=%(d)d)
'''
OURS_TAIL = '''bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false, mt=true, gpu=true)
return out.Trim(8, 247).Prefetch(3)
'''

# DTL's DLL: no gpu / mt parameters on MRecalculate and MFlowFps (his signatures, Interface.cpp:766-777)
DTL_CPU_CHAIN = '''supf = MSuper(pre, pel=2, hpad=16, vpad=16, rfilter=4)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1)
bv = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true)
fv = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true)
'''
DTL_DX12_CHAIN = '''supa = MSuper(pre, mt=false, pel=4, pelrefine=false, chroma=true, levels=1)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1)
bv = MAnalyse(supa, blksize=%(b)d, isb=true, delta=1, chroma=true, optSearchOption=%(o)d, levels=1%(x)s)
fv = MAnalyse(supa, blksize=%(b)d, isb=false, delta=1, chroma=true, optSearchOption=%(o)d, levels=1%(x)s)
'''
DTL_TAIL = '''bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true)
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true)
out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false)
return out.Trim(8, 247).Prefetch(3)
'''

# key, label, kind, parameter, MVCUDA_NVOFA
ALL = [
    ('exact', 'exact search', 'ours', None, ''),
    ('medium', 'NVOFA medium', 'ours', None, 'medium'),
    ('dx8', 'our D3D12 ME 8x8', 'proto', 8, ''),
    ('dx16', 'our D3D12 ME 16x16', 'proto', 16, ''),
    # one engine cell per block, no median over 32/16 blocks: the block layout DTL's MAnalyse gives MRecalculate
    ('dx8raw', 'our D3D12 ME 8x8, blocks 8/0', 'proto', (8, 8, 0), ''),
    ('dx16raw', 'our D3D12 ME 16x16, blocks 16/0', 'proto', (16, 16, 0), ''),
    ('dtl_cpu', 'DTL CPU search', 'dtl_cpu', None, ''),
    ('dtl5_8', 'DTL DX12_ME opt5 8x8', 'dtl', (5, 8), ''),
    ('dtl5_16', 'DTL DX12_ME opt5 16x16', 'dtl', (5, 16), ''),
    ('dtl6_8', 'DTL DX12_ME opt6 8x8', 'dtl', (6, 8), ''),
    ('dtl6_16', 'DTL DX12_ME opt6 16x16', 'dtl', (6, 16), ''),
]
BY_KEY = {v[0]: v for v in ALL}


def script_text(key):
    """self-contained AviSynth script of a variant (ending in `out = ...` and `return out...`)"""
    _, _, kind, par, _ = BY_KEY[key]
    if kind == 'ours':
        return SRC % (OURS, '') + OURS_CHAIN + OURS_TAIL
    if kind == 'proto':
        d, b, o = par if isinstance(par, tuple) else (par, 32, 16)
        return SRC % (OURS, 'LoadPlugin("%s")\n' % PROTO) + PROTO_CHAIN % {'b': b, 'o': o, 'd': d} + OURS_TAIL
    if kind == 'dtl_cpu':
        return SRC % (DTL, '') + DTL_CPU_CHAIN + DTL_TAIL
    o, b = par
    tail = DTL_TAIL.replace('Prefetch(3)', 'Prefetch(1)') if o == 5 else DTL_TAIL
    return SRC % (DTL, '') + DTL_DX12_CHAIN % {'o': o, 'b': b, 'x': ', UseSubShift=1' if o == 6 else ''} + tail


def write_script(work, key, text=None):
    path = os.path.join(work, 'flow_%s.avs' % key)
    with open(path, 'w') as fh:
        fh.write(script_text(key) if text is None else text)
    return path


def env_for(key):
    env = dict(os.environ)
    for k in ('MVCUDA_NVOFA', 'NVME_CHROMA', 'NVME_NOINTRA', 'NVME_P16', 'NVME_PSEUDO', 'NVME_FRESH'):
        env.pop(k, None)
    if BY_KEY[key][4]:
        env['MVCUDA_NVOFA'] = BY_KEY[key][4]
    return env


def run_dir(folder):
    """folder to run ffmpeg in: DTL's MAnalyse reads Compute.cso from the current directory"""
    os.makedirs(folder, exist_ok=True)
    cso = os.path.join(folder, 'Compute.cso')
    if not os.path.exists(cso):
        shutil.copy(os.path.join(os.path.dirname(DTL), 'Compute.cso'), cso)
    return folder
