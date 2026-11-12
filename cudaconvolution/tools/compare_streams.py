"""Frame-by-frame comparison of two videos (any ffmpeg/AviSynth input) as yuv422p10le 1920x1080: PSNR per plane
(peak 1023), share of identical samples, histogram of |difference|.  usage: python -I tools/compare_streams.py <a> <b> [frames]"""
import subprocess, sys
import numpy as np
W, H = 1920, 1080; Y = W * H; C = W // 2 * H; N = Y + 2 * C
def reader(path):
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv422p10le", "-"],
                         stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(N * 2)
        if len(b) < N * 2: return
        yield np.frombuffer(b, np.uint16)
a, b = reader(sys.argv[1]), reader(sys.argv[2]); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
se = np.zeros(3); eq = np.zeros(3); cnt = np.zeros(3); hist = np.zeros(6, np.int64); frames = 0
for fa, fb in zip(a, b):
    d = fa.astype(np.int32) - fb.astype(np.int32)
    for i, (s, e) in enumerate([(0, Y), (Y, Y + C), (Y + C, N)]):
        di = d[s:e]; se[i] += np.dot(di.astype(np.float64), di); eq[i] += np.count_nonzero(di == 0); cnt[i] += di.size
    hist += np.bincount(np.minimum(np.abs(d), 5), minlength=6)[:6]
    frames += 1
    if frames >= limit: break
psnr = 10 * np.log10(1023 ** 2 / np.maximum(se / cnt, 1e-12))
print(f"{frames} frames | PSNR Y {psnr[0]:.2f} U {psnr[1]:.2f} V {psnr[2]:.2f} dB | identical Y {eq[0]/cnt[0]*100:.2f}% "
      f"U {eq[1]/cnt[1]*100:.2f}% V {eq[2]/cnt[2]*100:.2f}% | |diff| 0/1/2/3/4/5+: " + " ".join(f"{h/hist.sum()*100:.2f}%" for h in hist))
