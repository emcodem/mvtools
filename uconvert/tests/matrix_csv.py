"""Write the UConvert test matrix as CSV: one row per generated input file (manifest.json of gen_matrix.py) with its
target and, where present, the results of run_matrix.py on the GPU and CPU chains (report.json).

usage: python -I tests/matrix_csv.py [--dir work/uc_matrix] [--out <dir>/test_matrix.csv]
"""
import os, sys, csv, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ucdata import data_root
ROOT = data_root()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=ROOT + "/work/uc_matrix")
    ap.add_argument("--out")
    a = ap.parse_args()
    man = json.load(open(a.dir + "/manifest.json"))
    rep = {}
    for chain in ("gpu", "cpu"):
        p = f"{a.dir}/run_{chain}/report.json"
        rep[chain] = {r["name"]: r for r in json.load(open(p))} if os.path.isfile(p) else {}
    out = a.out or a.dir + "/test_matrix.csv"
    cols = ["case", "file", "container", "video_codec", "pix_fmt", "width", "height", "fps", "scan", "sar",
            "duration_s", "audio_streams", "audio_channels", "audio_codecs", "audio_rates", "audio_length",
            "channel_tones_hz", "target_width", "target_height", "target_scan", "target_fps",
            "gpu_result", "gpu_plan", "gpu_errors", "cpu_result", "cpu_plan", "cpu_errors"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:   # BOM: Excel opens it as UTF-8
        w = csv.writer(f, delimiter=";")                          # ';' : Excel with German locale
        w.writerow(cols)
        for c in man:
            v, t, s = c["video"], c["target"], c["audio"]["streams"]
            vargs = v[2]
            row = [c["name"], os.path.basename(c["file"]), v[1], vargs[vargs.index("-c:v") + 1], v[3], v[4], v[5],
                   f"{v[6][0]}/{v[6][1]}", v[7], f"{v[8][0]}:{v[8][1]}", c["dur"],
                   len(s), sum(x[0] for x in s), " + ".join(f"{x[0]}ch {x[1]}" for x in s) or "none",
                   " + ".join(str(x[2]) for x in s) or "-",
                   {0: "= video"}.get(c["audio"].get("dur", 0), f"{c['audio'].get('dur', 0):+.1f} s"),
                   " ".join(str(x) for x in c["chan_freq"]) or "-",
                   t[1], t[0], t[2] or "p", f"{t[3]}/{t[4]}"]
            for chain in ("gpu", "cpu"):
                r = rep[chain].get(c["name"])
                row += ["not run", "", ""] if r is None else \
                       ["FAIL" if r["errors"] else "PASS", r["plan"], " | ".join(r["errors"])]
            w.writerow(row)
    print(f"{out}: {len(man)} cases")


if __name__ == "__main__":
    main()
