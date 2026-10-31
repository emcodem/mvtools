"""Variants of the NVENC ME comparison and the AviSynth scripts that produce them.

Every variant is the uc_FRC chain on seg1_wires_pole.mp4, output frames 8..247 of MFlowFps (as compare/nvofa_frc):
- exact / medium: compare/nvofa_frc/scripts/avs/frc_flow.avs (profile\\build_nvofa), MVCUDA_NVOFA unset / medium;
- NVENC: avs/frc_nvencme.avs with the prototype profile\\build_nvme_proto\\nvencme_proto.dll (source: the session
  scratchpad, see ../README.md) in place of MAnalyse.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NVOFA_AVS = os.path.normpath(os.path.join(HERE, '..', '..', 'nvofa_frc', 'scripts', 'avs', 'frc_flow.avs'))
NVME_AVS = os.path.join(HERE, 'avs', 'frc_nvencme.avs')
PLUGIN = os.environ.get('NVME_PLUGIN', r'C:\dev\mvtools\profile\build_nvme_proto\nvencme_proto.dll')

# key, label, NVENC (preset, half) or None, MVCUDA_NVOFA, extra environment (prototype switches)
ALL = [
    ('exact', 'exact search', None, ''),
    ('medium', 'NVOFA medium', None, 'medium'),
    ('p1', 'NVENC H.264 P1', (1, False), ''),
    ('p4', 'NVENC H.264 P4', (4, False), ''),
    ('p7', 'NVENC H.264 P7', (7, False), ''),
    ('p7h', 'NVENC H.264 P7 half-res', (7, True), ''),
    ('p4c', 'NVENC P4 + chroma', (4, False), '', {'NVME_CHROMA': '1'}),
    ('p4n', 'NVENC P4 no intra', (4, False), '', {'NVME_NOINTRA': '1'}),
    ('p4cn', 'NVENC P4 + chroma, no intra', (4, False), '', {'NVME_CHROMA': '1', 'NVME_NOINTRA': '1'}),
    ('p4cn16', 'NVENC P4 + chroma, no intra, 16x16', (4, False), '', {'NVME_CHROMA': '1', 'NVME_NOINTRA': '1', 'NVME_P16': '1'}),
]
ALL = [v if len(v) == 5 else v + ({},) for v in ALL]
BY_KEY = {v[0]: v for v in ALL}


def script_text(key):
    """self-contained AviSynth script of a variant (the flow chain, ending in `return out...`)"""
    nv = BY_KEY[key][2]
    if nv is None:
        return open(NVOFA_AVS).read()
    preset, half = nv
    head = ('global NVME_PLUGIN = "%s"\nglobal NVME_PRESET = %d\nglobal NVME_HALF = %s\n'
            % (PLUGIN, preset, 'true' if half else 'false'))
    return head + open(NVME_AVS).read()


def write_script(work, key, text=None):
    path = os.path.join(work, 'flow_%s.avs' % key)
    with open(path, 'w') as fh:
        fh.write(script_text(key) if text is None else text)
    return path


def env_for(key):
    env = dict(os.environ)
    for k in ('MVCUDA_NVOFA', 'NVME_CHROMA', 'NVME_NOINTRA', 'NVME_P16'):
        env.pop(k, None)
    env.update(BY_KEY[key][4])
    if BY_KEY[key][3]:
        env['MVCUDA_NVOFA'] = BY_KEY[key][3]
    return env
