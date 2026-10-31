"""Writes the HD test inputs of compare.py from the UHD source, once (both hosts then read the same raw files).

usage: python gen_inputs.py  (writes next to $MVPF_RAW, default ~/clips/seg1_40.yuv)
"""
import os

import numpy as np
import vapoursynth as vs

from vstest import core, raw_source

src_path = os.environ.get('MVPF_RAW', os.path.expanduser('~/clips/seg1_40.yuv'))
src = raw_source(src_path, 3840, 2160, vs.YUV422P10, 30000, 1001)
for name, fmt in (('hd420p8', vs.YUV420P8), ('hd444p10', vs.YUV444P10)):
    out_path = src_path.replace('.yuv', '_%s.yuv' % name)
    if os.path.exists(out_path):
        continue
    c = core.resize.Bicubic(src, 1920, 1080, format=fmt)
    with open(out_path + '.tmp', 'wb') as fh:
        for f in c.frames():
            for p in range(f.format.num_planes):
                fh.write(np.ascontiguousarray(np.asarray(f[p])).tobytes())
    os.replace(out_path + '.tmp', out_path)
    print(out_path)
