"""Helpers for the VapourSynth test scripts of mvtools_vsbridge (see vsmd5.py)."""
import os

import numpy as np
import vapoursynth as vs

core = vs.core


def raw_source(path, width, height, fmt, fpsnum, fpsden):
    """Planar raw video (Y, U, V planes per frame, unpadded rows), as AviSynth's RawSource reads it."""
    blank = core.std.BlankClip(format=fmt, width=width, height=height, length=1, fpsnum=fpsnum, fpsden=fpsden)
    ff = blank.format
    dtype = np.uint8 if ff.bytes_per_sample == 1 else (np.uint16 if ff.sample_type == vs.INTEGER else np.float32)
    shapes = []
    for p in range(ff.num_planes):
        w = width >> (ff.subsampling_w if p else 0)
        h = height >> (ff.subsampling_h if p else 0)
        shapes.append((h, w))
    frame_bytes = sum(h * w for h, w in shapes) * ff.bytes_per_sample
    count = os.path.getsize(path) // frame_bytes
    clip = core.std.BlankClip(blank, length=count)

    def fill(n, f):
        out = f.copy()
        with open(path, 'rb') as fh:
            fh.seek(n * frame_bytes)
            for p, (h, w) in enumerate(shapes):
                data = np.frombuffer(fh.read(h * w * ff.bytes_per_sample), dtype=dtype).reshape(h, w)
                np.asarray(out[p])[:, :] = data
        return out

    return core.std.ModifyFrame(clip, clip, fill)


def load_mvtools(plugin_dir=None):
    """Loads libmvtools2.so (namespace mvpf) from plugin_dir or $MVPF_PLUGIN_DIR."""
    d = plugin_dir or os.environ['MVPF_PLUGIN_DIR']
    if not hasattr(core, 'mvpf'):
        core.std.LoadPlugin(os.path.join(d, 'libmvtools2.so'))


def load_cudafilters(plugin_dir=None):
    """Loads libcudafilters.so (namespace cuda) from plugin_dir or $MVPF_PLUGIN_DIR."""
    d = plugin_dir or os.environ['MVPF_PLUGIN_DIR']
    if not hasattr(core, 'cuda'):
        core.std.LoadPlugin(os.path.join(d, 'libcudafilters.so'))
