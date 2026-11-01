"""avs_info.py: VideoInfo and GetParity of an AviSynth+ script result through AviSynth+'s C API (ctypes), for checking the
bridge's stand-ins against the filters they stand in for (stand_ins.py ref). Windows or Linux, any host with AviSynth+.

    from avs_info import video_info
    vi = video_info('BlankClip(pixel_type="YV12").Bob()')   # dict, or raises RuntimeError with AviSynth's message
"""
import ctypes
import os
import sys


class AVS_Value(ctypes.Structure):
    class D(ctypes.Union):
        _fields_ = [('clip', ctypes.c_void_p), ('boolean', ctypes.c_char), ('integer', ctypes.c_int),
                    ('floating_pt', ctypes.c_float), ('string', ctypes.c_char_p), ('array', ctypes.c_void_p),
                    ('longlong', ctypes.c_int64), ('double_pt', ctypes.c_double)]
    _fields_ = [('type', ctypes.c_short), ('array_size', ctypes.c_short), ('d', D)]


class AVS_VideoInfo(ctypes.Structure):
    _fields_ = [('width', ctypes.c_int), ('height', ctypes.c_int), ('fps_numerator', ctypes.c_uint),
                ('fps_denominator', ctypes.c_uint), ('num_frames', ctypes.c_int), ('pixel_type', ctypes.c_int),
                ('audio_samples_per_second', ctypes.c_int), ('sample_type', ctypes.c_int),
                ('num_audio_samples', ctypes.c_int64), ('nchannels', ctypes.c_int), ('image_type', ctypes.c_int)]


_lib = None
_env = None


def _load():
    global _lib, _env
    if _lib is None:
        name = 'AviSynth.dll' if sys.platform == 'win32' else os.environ.get(
            'AVS_LIB', os.path.expanduser('~/.local/lib/libavisynth.so'))
        _lib = ctypes.CDLL(name)
        _lib.avs_create_script_environment.restype = ctypes.c_void_p
        _lib.avs_create_script_environment.argtypes = [ctypes.c_int]
        _lib.avs_invoke.restype = AVS_Value
        _lib.avs_invoke.argtypes = [ctypes.c_void_p, ctypes.c_char_p, AVS_Value, ctypes.c_void_p]
        _lib.avs_take_clip.restype = ctypes.c_void_p
        _lib.avs_take_clip.argtypes = [AVS_Value, ctypes.c_void_p]
        _lib.avs_get_video_info.restype = ctypes.POINTER(AVS_VideoInfo)
        _lib.avs_get_video_info.argtypes = [ctypes.c_void_p]
        _lib.avs_get_parity.restype = ctypes.c_int
        _lib.avs_get_parity.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _lib.avs_release_clip.argtypes = [ctypes.c_void_p]
        _lib.avs_release_value.argtypes = [AVS_Value]
        _env = _lib.avs_create_script_environment(8)
    return _lib, _env


def video_info(script, parities=4):
    """VideoInfo fields of the script's result, plus 'parity': GetParity(0 .. parities-1) as a string of 0 / 1."""
    lib, env = _load()
    arg = AVS_Value()
    arg.type = ord('s')
    arg.d.string = script.encode()
    v = lib.avs_invoke(env, b'Eval', arg, None)
    if v.type == ord('e'):
        msg = v.d.string.decode(errors='replace')
        lib.avs_release_value(v)
        raise RuntimeError(msg)
    if v.type != ord('c'):
        raise RuntimeError('script did not return a clip')
    clip = lib.avs_take_clip(v, env)
    vi = lib.avs_get_video_info(clip).contents
    out = {k: getattr(vi, k) for k, _ in AVS_VideoInfo._fields_}
    out['parity'] = ''.join(str(int(bool(lib.avs_get_parity(clip, n)))) for n in range(parities))
    lib.avs_release_clip(clip)
    lib.avs_release_value(v)
    return out


if __name__ == '__main__':
    print(video_info(sys.argv[1]))
