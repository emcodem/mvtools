"""vsmd5: framemd5 of a VapourSynth script, line for line comparable with avsmd5 (AviSynth+) and ffmpeg -f framemd5:
MD5 of the unpadded planes of each frame in plane order (Y, U, V).

usage: python vsmd5.py script.vpy out.md5 [threads]
"""
import hashlib
import os
import runpy
import sys
import time

import vapoursynth as vs


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    script, out_path = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        vs.core.num_threads = int(sys.argv[3])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    runpy.run_path(script, run_name='__vapoursynth__')
    clip = vs.get_output(0)
    if isinstance(clip, vs.VideoOutputTuple):
        clip = clip.clip
    t0 = time.time()
    with open(out_path, 'w') as out:
        out.write('#format: frame checksums\n#version: 2\n#hash: MD5\n')
        for n, f in enumerate(clip.frames()):
            m = hashlib.md5()
            size = 0
            for p in range(f.format.num_planes):
                row = f.width * f.format.bytes_per_sample if p == 0 else \
                    (f.width >> f.format.subsampling_w) * f.format.bytes_per_sample
                h = f.height if p == 0 else (f.height >> f.format.subsampling_h)
                mv = memoryview(f[p]).cast('B')
                stride = f.get_stride(p)
                for y in range(h):
                    m.update(mv[y * stride:y * stride + row])
                size += row * h
            out.write('0, %10d, %10d,        1, %8d, %s\n' % (n, n, size, m.hexdigest()))
    print('vsmd5: %d frames, %.2f s' % (clip.num_frames, time.time() - t0), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
