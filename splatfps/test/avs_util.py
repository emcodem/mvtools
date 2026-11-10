"""Shared helpers of the SplatFps tests: AviSynth+ scripts read through ffmpeg (avs input), ffprobe.

The L-SMASH Works source plugin (LSMASHSource.dll) is taken from the environment variable SPLATFPS_LSMASH. Every child
process runs hidden (no console window) and ffmpeg never reads stdin.
"""
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

NOWIN = 0x08000000 if os.name == 'nt' else 0


def lsmash():
    p = os.environ.get('SPLATFPS_LSMASH')
    if not p:
        sys.exit('SPLATFPS_LSMASH is not set: point it at LSMASHSource.dll (the L-SMASH Works AviSynth+ plugin), '
                 'or pass --lsmash <path> to run_suite.py')
    if not os.path.isfile(p):
        sys.exit(f'SPLATFPS_LSMASH: {p} not found')
    return p


def source_script(src, plugins=(), var=None):
    """AviSynth+ lines loading L-SMASH Works and the given plugins, then the source clip (assigned to var, if given)"""
    s = f'LoadPlugin("{lsmash()}")\n' + ''.join(f'LoadPlugin("{os.path.abspath(p)}")\n' for p in plugins)
    return s + (f'{var} = ' if var else '') + f'LWLibavVideoSource("{os.path.abspath(src)}", cache=false)\n'


def ffmpeg_script(script, out_args, binary=True):
    """runs ffmpeg on an AviSynth+ script (hidden); returns (returncode, stdout, stderr text)"""
    with tempfile.NamedTemporaryFile('w', suffix='.avs', delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path] + out_args + ['-'], capture_output=True,
                           creationflags=NOWIN)
    finally:
        os.unlink(path)
    err = r.stderr.decode(errors='replace').strip()
    return r.returncode, (r.stdout if binary else r.stdout.decode(errors='replace')), err


def framemd5(script):
    """framemd5 lines of the script's clip (no comments) and ffmpeg's error text"""
    _, out, err = ffmpeg_script(script, ['-f', 'framemd5'], binary=False)
    return [l for l in out.splitlines() if l and not l.startswith('#')], err


# planar formats: (bytes per sample, log2 chroma subsampling x, y)
PIX = {'yuv420p': (1, 1, 1), 'yuv422p': (1, 1, 0), 'yuv444p': (1, 0, 0)}
for _b in (10, 12, 16):
    for _n, (_, _sx, _sy) in list(PIX.items())[:3]:
        PIX[f'{_n}{_b}le'] = (2, _sx, _sy)


def raw_frames(script, w, h, pix_fmt='yuv422p10le'):
    """every frame of the script's clip as integer codes: Y (n, h, w), C (n, 2, h >> sy, w >> sx)"""
    bps, sx, sy = PIX[pix_fmt]
    rc, out, err = ffmpeg_script(script, ['-f', 'rawvideo', '-pix_fmt', pix_fmt])
    if rc:
        sys.exit('ffmpeg failed: ' + err[-2000:])
    cw, ch = (w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy
    x = np.frombuffer(out, np.uint16 if bps == 2 else np.uint8).astype(np.int32).reshape(-1, w * h + 2 * cw * ch)
    return x[:, :w * h].reshape(-1, h, w), x[:, w * h:].reshape(-1, 2, ch, cw)


def avs_frames(script, frames, w, h, pix_fmt='yuv422p10le'):
    """the frames of the script's clip s, selected inside the script (no other frame of s is ever requested), in the
    order given"""
    return raw_frames(script + ' ++ '.join(f's.Trim({k}, -1)' for k in frames) + '\n', w, h, pix_fmt)


def probe(src):
    """frame count, frame rate (Fraction), width, height, pix_fmt of the first video stream"""
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                        'stream=nb_read_packets,r_frame_rate,width,height,pix_fmt', '-of', 'default=nw=1', src],
                       capture_output=True, text=True, creationflags=NOWIN, stdin=subprocess.DEVNULL)
    if r.returncode:
        sys.exit(f'ffprobe failed on {src}: {r.stderr.strip()[-500:]}')
    kv = dict(l.split('=', 1) for l in r.stdout.splitlines() if '=' in l)
    return (int(kv['nb_read_packets']), Fraction(kv['r_frame_rate']), int(kv['width']), int(kv['height']),
            kv['pix_fmt'])
