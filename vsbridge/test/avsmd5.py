"""avsmd5.py: framemd5 of an AviSynth+ script through AviSynth+'s C API (ctypes), in the format of vsmd5.py and of the
Linux avsmd5 tool: MD5 of the unpadded planes of each frame in plane order Y, U, V (planar RGB: G, B, R; packed
formats such as the vector clips: the one interleaved plane). Windows (AviSynth.dll) or Linux ($AVS_LIB).

usage: python avsmd5.py script.avs out.md5
"""
import ctypes
import hashlib
import os
import struct
import sys
import time

from avs_info import AVS_Value, AVS_VideoInfo

PLANAR_Y, PLANAR_U, PLANAR_V = 1 << 0, 1 << 1, 1 << 2
PLANAR_A = 1 << 4
PLANAR_R, PLANAR_G, PLANAR_B = 1 << 5, 1 << 6, 1 << 7
# AVSMD5_ZERO_VECTOR_HEADER=1 (compare.py sets it on Windows): hash the unused bytes 96..255 of packed one-row frames
# (vector clips) as zeros
ZERO_VECTOR_HEADER = os.environ.get('AVSMD5_ZERO_VECTOR_HEADER', '0') not in ('', '0')
CS_PLANAR, CS_INTERLEAVED, CS_BGR, CS_YUVA, CS_RGBA_TYPE = 1 << 31, 1 << 30, 1 << 28, 1 << 27, 1 << 1


def zero_unwritten(data):
    """Packed frames of the vector clips with the bytes mvtools2 never writes set to zero, as the bridge allocates them:
    MAnalyse / MRecalculate write 96 bytes of the 256-byte header of a vector frame, MStoreVect copies such frames behind
    its own header and leaves the rest of its (mod-2 height) frame unwritten. AviSynth+ on Windows leaves whatever the
    new frame's memory held there, varying run to run."""
    def vec(buf, pos):
        if pos + 256 <= len(buf):
            buf[pos + 96:pos + 256] = bytes(160)

    buf = bytearray(data)
    words = struct.unpack_from('<3i', buf) if len(buf) >= 12 else (0, 0, 0)
    if words[0] == 0xBEAD:  # MStoreVect: key, version, n, n offsets + end offset (int32 units), then the vector frames
        n = words[2]
        offs = struct.unpack_from('<%di' % (n + 1), buf, 12)
        for o in offs[:-1]:
            vec(buf, o * 4)
        end = offs[-1] * 4
        buf[end:] = bytes(len(buf) - end)
    elif words[0] == 256 and len(buf) >= 256:  # vector frame: header size first
        vec(buf, 0)
    return bytes(buf)


def load():
    name = 'AviSynth.dll' if sys.platform == 'win32' else os.environ.get(
        'AVS_LIB', os.path.expanduser('~/.local/lib/libavisynth.so'))
    lib = ctypes.CDLL(name)
    proto = {
        'avs_create_script_environment': (ctypes.c_void_p, [ctypes.c_int]),
        'avs_invoke': (AVS_Value, [ctypes.c_void_p, ctypes.c_char_p, AVS_Value, ctypes.c_void_p]),
        'avs_take_clip': (ctypes.c_void_p, [AVS_Value, ctypes.c_void_p]),
        'avs_get_video_info': (ctypes.POINTER(AVS_VideoInfo), [ctypes.c_void_p]),
        'avs_get_frame': (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int]),
        'avs_clip_get_error': (ctypes.c_char_p, [ctypes.c_void_p]),
        'avs_release_video_frame': (None, [ctypes.c_void_p]),
        'avs_release_clip': (None, [ctypes.c_void_p]),
        'avs_release_value': (None, [AVS_Value]),
        'avs_delete_script_environment': (None, [ctypes.c_void_p]),
        'avs_get_read_ptr_p': (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int]),
        'avs_get_pitch_p': (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        'avs_get_row_size_p': (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        'avs_get_height_p': (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    }
    for fn, (res, args) in proto.items():
        f = getattr(lib, fn)
        f.restype = res
        f.argtypes = args
    return lib


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    script, out_path = os.path.abspath(sys.argv[1]), sys.argv[2]
    lib = load()
    env = lib.avs_create_script_environment(8)
    arg = AVS_Value()
    arg.type = ord('s')
    arg.d.string = script.encode()
    v = lib.avs_invoke(env, b'Import', arg, None)
    if v.type == ord('e'):
        print('avsmd5: %s' % v.d.string.decode(errors='replace'), file=sys.stderr)
        return 1
    if v.type != ord('c'):
        print('avsmd5: the script did not return a clip', file=sys.stderr)
        return 1
    clip = lib.avs_take_clip(v, env)
    vip = lib.avs_get_video_info(clip)
    vi = vip.contents
    pt = vi.pixel_type & 0xffffffff
    if not pt & CS_PLANAR:
        planes = [0]  # packed (BGR24 / BGR32 / YUY2: the vector clips)
    elif pt & CS_INTERLEAVED:
        planes = [PLANAR_Y]  # CS_GENERIC_Y
    elif pt & CS_BGR:
        planes = [PLANAR_G, PLANAR_B, PLANAR_R] + ([PLANAR_A] if pt & CS_RGBA_TYPE else [])
    else:
        planes = [PLANAR_Y, PLANAR_U, PLANAR_V] + ([PLANAR_A] if pt & CS_YUVA else [])
    t0 = time.time()
    rc = 0
    with open(out_path, 'w') as out:
        out.write('#format: frame checksums\n#version: 2\n#hash: MD5\n')
        for n in range(vi.num_frames):
            f = lib.avs_get_frame(clip, n)
            err = lib.avs_clip_get_error(clip)
            if not f or err:
                print('avsmd5: frame %d: %s' % (n, err.decode(errors='replace') if err else 'no frame'), file=sys.stderr)
                rc = 1
                break
            m = hashlib.md5()
            size = 0
            for p in planes:
                ptr = lib.avs_get_read_ptr_p(f, p)
                pitch = lib.avs_get_pitch_p(f, p)
                row = lib.avs_get_row_size_p(f, p)
                h = lib.avs_get_height_p(f, p)
                data = b''.join(ctypes.string_at(ptr + y * pitch, row) for y in range(h))
                if p == 0 and ZERO_VECTOR_HEADER:
                    data = zero_unwritten(data)
                m.update(data)
                size += row * h
            lib.avs_release_video_frame(f)
            out.write('0, %10d, %10d,        1, %8d, %s\n' % (n, n, size, m.hexdigest()))
    print('avsmd5: %d frames, %.2f s' % (vi.num_frames, time.time() - t0), file=sys.stderr)
    sys.stderr.flush()
    # No teardown: on Windows, avs_delete_script_environment after a Prefetch chain (lin_cpu) sometimes crashes with an
    # access violation (seen with the solution-built mvtools2.dll too; ffmpeg's teardown does not hit it). The checksums
    # are written; leave the process directly.
    os._exit(rc)


if __name__ == '__main__':
    sys.exit(main())
