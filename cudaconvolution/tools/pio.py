"""Raw yuv422p10le access to the decoded probe and the reference output.
Set CUDACONVOLUTION_DATA to a folder holding pin.yuv (probe, 200 UHD frames) and pref.yuv (reference output, 100 HD frames):
  ffmpeg -i <probe .mov>             -map 0:v -f rawvideo -pix_fmt yuv422p10le pin.yuv
  ffmpeg -i <reference output .mov>  -map 0:v -f rawvideo -pix_fmt yuv422p10le pref.yuv"""
import os
import numpy as np
S = os.environ["CUDACONVOLUTION_DATA"]
_in = np.memmap(os.path.join(S, "pin.yuv"), np.uint16, "r").reshape(200, -1)
_ref = np.memmap(os.path.join(S, "pref.yuv"), np.uint16, "r").reshape(100, -1)
def planes(buf, w, h):
    y = buf[:w*h].reshape(h, w); c = w//2*h
    return y, buf[w*h:w*h+c].reshape(h, w//2), buf[w*h+c:].reshape(h, w//2)
def inp(n, pl=0): return planes(_in[n], 3840, 2160)[pl]
def ref(k, pl=0): return planes(_ref[k], 1920, 1080)[pl]
def src_of(k, y): return 2*k + (y % 2)   # progressive frame feeding output row y
