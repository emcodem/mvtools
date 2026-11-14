"""QTGMC against ground truth: a clean 1080p50 progressive clip, made 8-bit 4:2:0 and interlaced (TFF, field n from
source frame n), QTGMC bobs it back to 50p; output frame n is compared with source frame n. Motion analysis variants:
exact GPU search (= stock), NVOFA, NVENC (CudaNVENCAnalyse with arguments).

python qtgt.py <work> [--presets Fast,Slower] [--clips crowd,park,tree] [--frames 80] [--keys ...] [--time]
"""
import argparse, json, os, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get('QT_BUNDLE', os.path.join(HERE, 'bundle')).replace('\\', '/')
T = os.environ.get('QT_SVT_DIR', 'C:/path/to/svt/')  # the SVT 1080p50 sequences (MXF)
CLIPS = {
    'crowd': T + '1_CrowdRun_1080p50_CgrLevels_SINC_FILTER_SVTdec05_.mxf',
    'park': T + '2_ParkJoy_1080p50_CgrLevels_SINC_FILTER_SVTdec05_.mxf',
    'ducks': T + '3_DucksTakeOff_1080p50_CgrLevels_SINC_FILTER_SVTdec05_.mxf',
    'tree': T + '4_InToTree_1080p50_CgrLevels_SINC_FILTER_SVTdec05_.mxf',
    'town': T + '5_OldTownCross_1080p50_CgrLevels_SINC_FILTER_SVTdec05_.mxf',
}
VARIANTS = {
    'exact': ('exact GPU search (= stock QTGMC)', ''),
    'nvof': ('NVOFA medium', 'global QTGMC_gpu_nvof = true\n'),
    'nvenc_plain': ('NVENC qp=-1, swap=false', 'global QTGMC_gpu_nvenc = true\nglobal QTGMC_gpu_nvenc_args = """, qp=-1, swap=false"""\n'),
    'nvenc': ('NVENC defaults (QP 10 + swapped run)', 'global QTGMC_gpu_nvenc = true\n'),
    'nvenc_yuv': ('NVENC channels yuv', 'global QTGMC_gpu_nvenc = true\nglobal QTGMC_gpu_nvenc_args = """, channels="yuv" """\n'),
    'nvenc_qp30': ('NVENC QP 30 + swapped run', 'global QTGMC_gpu_nvenc = true\nglobal QTGMC_gpu_nvenc_args = """, qp=30"""\n'),
}
W, H = 1920, 1080
YS = W * H
FS = YS * 3 // 2
B = 16

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--presets', default='Fast,Slower')
ap.add_argument('--clips', default='crowd,park,tree')
ap.add_argument('--frames', type=int, default=80)
ap.add_argument('--keys', default=','.join(VARIANTS))
ap.add_argument('--time', action='store_true', help='also time QTGMC alone (best of 2)')
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)


def script(clip, preset, key, compare):
    s = ('Import("%s/QTGMC_gpu_load.avsi")\nQTGMC_gpu_on()\n%s'
         'src = FFVideoSource("%s", cachefile="%s/%s.ffindex").Trim(0, %d).ConvertBits(8).ConvertToYV12()\n'
         'i = src.AssumeFrameBased().AssumeTFF().SeparateFields().SelectEvery(4, 0, 3).Weave()\n'
         'q = i.QTGMC(Preset="%s")\n'
         % (BUNDLE, VARIANTS[key][1], CLIPS[clip], a.work.replace('\\', '/'), clip, a.frames - 1, preset))
    if compare:
        s += 'return Interleave(q, src).Prefetch(4)\n'
    else:
        s += 'return q.Prefetch(4)\n'
    return s


def run(avs_text, name, raw):
    p = os.path.join(a.work, name + '.avs')
    open(p, 'w').write(avs_text)
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', p]
    cmd += ['-f', 'rawvideo', '-'] if raw else ['-f', 'null', '-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE if raw else subprocess.DEVNULL, stderr=subprocess.PIPE)


res = {}
for preset in a.presets.split(','):
    for clip in a.clips.split(','):
        for key in a.keys.split(','):
            tag = '%s_%s_%s' % (preset, clip, key)
            cache = os.path.join(a.work, tag + '.json')
            if os.path.exists(cache):
                res[tag] = json.load(open(cache))
                continue
            pr = run(script(clip, preset, key, True), tag, True)
            psnr, blocks, worst = [], [], []
            n = 0
            while True:
                q = pr.stdout.read(FS)
                s = pr.stdout.read(FS)
                if len(s) < FS:
                    break
                n += 1
                if n <= 3 or n > a.frames - 3:
                    continue  # temporal edges
                q = np.frombuffer(q, np.uint8)
                s = np.frombuffer(s, np.uint8)
                d = q[:YS].astype(np.float64) - s[:YS]
                psnr.append(10 * np.log10(255 * 255 / max(np.mean(d * d), 1e-9)))
                nb = H // B  # 67 block rows (the last 8 luma rows are left out)
                yd = np.abs(q[:YS].astype(np.int16) - s[:YS]).reshape(H, W)[:nb * B]
                yb = yd.reshape(nb, B, W // B, B).mean(axis=(1, 3))
                cs = (W // 2) * (H // 2)
                cd = np.abs(q[YS:].astype(np.int16) - s[YS:]).reshape(2, H // 2, W // 2)[:, :nb * 8]
                cu = cd.reshape(2, nb, 8, W // 16, 8).mean(axis=(2, 4)).sum(axis=0)
                tot = yb + cu
                blocks.append(tot.ravel().astype(np.float32))
                worst.append((float(tot.max()), n - 1))
            err = pr.stderr.read().decode()
            pr.wait()
            if not psnr:
                print(tag, 'FAILED', err[-400:])
                continue
            allb = np.concatenate(blocks)
            r = {'psnr': float(np.mean(psnr)), 'psnr_min': float(np.min(psnr)), 'p99': float(np.percentile(allb, 99)),
                 'p999': float(np.percentile(allb, 99.9)), 'gt8': int((allb > 8).sum()), 'gt16': int((allb > 16).sum()),
                 'worst': sorted(worst, reverse=True)[:3]}
            if a.time:
                best = 1e9
                for _ in range(2):
                    t = time.perf_counter()
                    pt = run(script(clip, preset, key, False), tag + '_t', False)
                    pt.wait()
                    best = min(best, time.perf_counter() - t)
                r['time'] = best
            res[tag] = r
            json.dump(r, open(cache, 'w'))
            print('done', tag, '%.2f dB' % r['psnr'], flush=True)

for preset in a.presets.split(','):
    for clip in a.clips.split(','):
        print('\n### %s, %s (%d frames, frame n vs source n)\n' % (preset, clip, a.frames))
        print('| variant | Y PSNR | min | block p99 | p99.9 | blocks > 8 | > 16 | wall | worst frames |\n|---|---:|---:|---:|---:|---:|---:|---:|---|')
        for key in a.keys.split(','):
            tag = '%s_%s_%s' % (preset, clip, key)
            if tag not in res:
                continue
            r = res[tag]
            print('| %s | %.2f | %.2f | %.2f | %.2f | %d | %d | %s | %s |' % (
                VARIANTS[key][0], r['psnr'], r['psnr_min'], r['p99'], r['p999'], r['gt8'], r['gt16'],
                ('%.2f s' % r['time']) if 'time' in r else '', ', '.join('%d (%.0f)' % (f, e) for e, f in r['worst'])))
