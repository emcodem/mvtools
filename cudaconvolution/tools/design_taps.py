"""Design the per-factor kernel tables of preset balanced and write presets/balanced.json.

Factor 2 (H and V) is the measured reference filter, factor 1 is the hd filter (H identity, V 9 taps). Factors 3..8 are
designed here so that every factor has the same look: the target response, relative to the OUTPUT grid, is the measured
2:1 response. A factor-n kernel acts on input samples (frequency f, cycles/input sample); output-relative frequency is
F = n f, the measured 2:1 kernel sees F / 2, so the target is T(f) = R2(n f / 2) for n f / 2 <= 0.5, else 0.
Symmetric FIR (the measured chroma kernel is symmetrised for the target), +-K taps with K = ceil(K2 * n / 2), DC gain
exactly 1 (least squares on a dense grid), integer taps / 65536. Frozen in the JSON: no floating-point design at run time.
usage: python -I tools/design_taps.py            (rewrites presets/balanced.json, prints the fit quality)"""
import json, math, pathlib, re
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
PRESET = ROOT / "presets" / "balanced.json"
DIV = 65536
F = np.linspace(0, 0.5, 8001)

def response(taps, first, f):
    j = np.arange(first, first + len(taps))
    return (np.asarray(taps)[None, :] * np.cos(2 * np.pi * f[:, None] * j[None, :])).sum(1)

def design(ref, ref_first, n):
    """ref: measured 2:1 kernel (float values), offsets ref_first.. ; returns (first, integer taps / DIV, rms error)"""
    ref = np.asarray(ref, dtype=np.float64)
    sym = 0.5 * (ref + ref[::-1]) if ref_first == -(len(ref) // 2) else ref
    k2 = len(ref) // 2
    K = math.ceil(k2 * n / 2)
    arg = n * F / 2
    target = np.where(arg <= 0.5, response(sym, ref_first, np.minimum(arg, 0.5)), 0.0)
    k = np.arange(1, K + 1)
    A = 2 * (np.cos(2 * np.pi * F[:, None] * k[None, :]) - 1)          # DC fixed: h0 = 1 - 2 sum h_k
    hk = np.linalg.lstsq(A, target - 1, rcond=None)[0]
    h = np.r_[hk[::-1], 1 - 2 * hk.sum(), hk]
    q = np.round(h * DIV).astype(int); q[K] += DIV - q.sum()
    err = response(q / DIV, -K, F) - target
    return -K, q.tolist(), float(np.sqrt((err ** 2).mean()))

def kernel_values(k, divisor):
    return [v / k.get("divisor", divisor) for v in k["taps"]]

def main():
    j = json.loads(PRESET.read_text())
    if "modes" in j:  # first conversion from the per-size modes
        u, h = j["modes"]["uhd"], j["modes"]["hd"]
        prof = lambda m: {k: m[k] for k in ("arith", "offset_code", "intermediate_clip", "border")}
        hv = {"1": {"profile": "designed", "provenance": h["provenance"],
                    "luma": h["taps"]["luma_h"], "chroma": h["taps"]["chroma_h"]},
              "2": {"profile": "measured", "provenance": u["provenance"],
                    "luma": dict(u["taps"]["luma_h"], divisor=u["taps"]["divisor"]), "chroma": u["taps"]["chroma_h"]}}
        hv["1"]["luma"] = dict(hv["1"]["luma"], divisor=16384); hv["1"]["chroma"] = dict(hv["1"]["chroma"], divisor=16384)
        vv = {"1": {"profile": "designed", "provenance": h["provenance"], "kernel": dict(h["taps"]["v"], divisor=16384)},
              "2": {"profile": "measured", "provenance": u["provenance"], "kernel": dict(u["taps"]["v"], divisor=16384)}}
        j = {"name": j["name"],
             "description": "Progressive -> interlaced (R p -> R/2 i) with optional integer downscale. Factor 2 is the "
                            "measured reference filter, factor 1 the hd interlace filter, factors 3..8 designed with the "
                            "same look (tools/design_taps.py).",
             "target": {"scan": j["target"]["scan"], "field_order": j["target"]["field_order"]},
             "profiles": {"measured": prof(u), "designed": prof(h)},
             "h": hv, "v": vv}
    H2, V2 = j["h"]["2"], j["v"]["2"]
    prov = {"type": "designed", "method": "tools/design_taps.py: LS fit to the measured 2:1 response on the output grid",
            "date": "2026-10-15"}
    print("factor  axis    taps  rms error")
    for n in range(3, 9):
        hl = design(kernel_values(H2["luma"], 1), H2["luma"]["first"], n)
        hc = design(kernel_values(H2["chroma"], 1), H2["chroma"]["first"], n)
        vk = design(kernel_values(V2["kernel"], 1), V2["kernel"]["first"], n)
        j["h"][str(n)] = {"profile": "designed", "provenance": prov,
                          "luma": {"divisor": DIV, "first": hl[0], "taps": hl[1]},
                          "chroma": {"divisor": DIV, "first": hc[0], "taps": hc[1]}}
        j["v"][str(n)] = {"profile": "designed", "provenance": prov, "kernel": {"divisor": DIV, "first": vk[0], "taps": vk[1]}}
        for name, d in (("H luma", hl), ("H chroma", hc), ("V", vk)):
            print(f"  {n}     {name:8s} {len(d[1]):4d}  {d[2]:.5f}")
    # sanity: the method applied at factor 2 / factor 1 reproduces the measured / hd filters closely
    for n, ref, label in ((2, V2["kernel"], "V 2 vs measured"), (1, j["v"]["1"]["kernel"], "V 1 vs hd")):
        d = design(kernel_values(V2["kernel"], 1), V2["kernel"]["first"], n)
        diff = np.abs(response(np.array(d[1]) / DIV, d[0], F) - response(kernel_values(ref, 1), ref["first"], F)).max()
        print(f"  check {label}: max response difference {diff:.4f}")
    txt = json.dumps(j, indent=2)
    txt = re.sub(r'\[\s+([-0-9.e,\s]+?)\s+\]', lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']', txt)
    PRESET.write_text(txt + "\n")

if __name__ == "__main__":
    main()
