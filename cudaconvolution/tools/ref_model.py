"""Python model of the measured reference UHD p -> 1080i (mode "uhd" of preset balanced), derived from the bit-exact
probe (2026-10-14); taps are read from presets/balanced.json (single source of truth).
Per output frame k: field p (p = output row parity) comes from progressive frame 2k+p; output row y <- UHD row 2y, col x <- UHD col 2x.
Arithmetic: float32, d = x - 512, H pass then V pass, each tap accumulated with FMA in ascending tap order,
intermediate clip [-512, 508], output 512 + trunc(o + 0.5), output clip 1..1020 (the reference's; CudaConvolution applies it
only with clip_min=1, clip_max=1020; out_clip=(0, 1023) models the default)."""
import json, pathlib
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from preset_io import measured
f32 = np.float32
_P = measured()
def _taps(k):
    """taps as float64 (integer / divisor, or float32 values with divisor 1), offsets"""
    t = _P["taps"][k]; d = t.get("divisor", _P["taps"]["divisor"])
    return np.array(t["taps"]) / float(d), np.arange(t["first"], t["first"] + len(t["taps"]))
def _edges(k):
    """edge kernels of an H kernel: [(side, i, src_first, taps)], see EdgeKernel in src/core/preset.h"""
    e = _P["taps"][k].get("edges", {})
    return [(side, i, ek["src_first"], np.array(ek["taps"], dtype=np.float64)) for side in ("left", "right") for i, ek in enumerate(e.get(side, []))]
V, VR = _taps("v")
HY, HYR = _taps("luma_h")
HC, HCR = _taps("chroma_h")
HCE = _edges("chroma_h")
def fma_acc(acc, c, x):
    # float32 FMA emulated exactly: the float64 product of two float32 values is exact, one rounding to float32
    # (double rounding float64 -> float32 is harmless here, see p_f32b.py: 0 mismatches)
    return (acc.astype(np.float64) + np.float64(c) * x.astype(np.float64)).astype(f32)
def pad_idx(n, lo, hi, mode):
    i = np.arange(lo, hi)
    if mode == "replicate": return np.clip(i, 0, n - 1)
    if mode == "mirror": i = np.abs(i); return np.where(i > n - 1, 2 * (n - 1) - i, i)          # reflect without edge repeat
    if mode == "mirror_rep": i = np.where(i < 0, -i - 1, i); return np.where(i > n - 1, 2 * n - 1 - i, i)   # symmetric with edge repeat
    raise ValueError(mode)
def field(U, p, H=HY, HR=HYR, edge="replicate", edges=(), out_clip=(1, 1020)):
    """U: UHD plane (2160 x w) of the progressive frame for field p; returns output rows of parity p (540 x w/2).
    edges: edge kernels replacing H for the outermost output columns (in-plane sources, no border rule)"""
    h, w = U.shape; ow = w // 2
    X = (U.astype(f32) - f32(512))
    cols = pad_idx(w, HR.min(), w + HR.max() + 1, edge)
    Xp = X[:, cols]; c0 = -HR.min()
    acc = np.zeros((h, ow), f32)
    for c, j in zip(H, HR): acc = fma_acc(acc, f32(c), Xp[:, c0 + 2 * np.arange(ow) + j])
    for side, i, first, taps in edges:
        x = i if side == "left" else ow - 1 - i; s0 = first if side == "left" else w + first
        e = np.zeros(h, f32)
        for t, c in enumerate(taps): e = fma_acc(e, f32(c), X[:, s0 + t])
        acc[:, x] = e
    acc = np.clip(acc, f32(-512), f32(508))
    ys = np.arange(p, h // 2, 2)
    rows = pad_idx(h, VR.min(), h + VR.max() + 1, edge); r0 = -VR.min(); Ap = acc[rows]
    o = np.zeros((len(ys), ow), f32)
    for c, i in zip(V, VR): o = fma_acc(o, f32(c), Ap[r0 + 2 * ys + i])
    return np.clip(512 + np.trunc(o.astype(np.float64) + 0.5).astype(np.int64), *out_clip)
def frame(src0, src1, out_clip=(1, 1020)):
    """src0/src1: (Y, U, V) planes of progressive frames 2k / 2k+1 (TFF). Returns the interlaced output frame (Y, U, V)."""
    out = []
    for pl, (H, HR, E) in enumerate([(HY, HYR, ()), (HC, HCR, HCE), (HC, HCR, HCE)]):
        h, w = src0[pl].shape; o = np.empty((h // 2, w // 2), np.int64)
        o[0::2] = field(src0[pl], 0, H, HR, edges=E, out_clip=out_clip); o[1::2] = field(src1[pl], 1, H, HR, edges=E, out_clip=out_clip); out.append(o)
    return out
