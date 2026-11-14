"""QTGMC on real interlaced material: wall time per motion source (whole GPU chain) and the spots where NVENC's output
differs most from the stock (exact search) output, as crops for the eye: exact | NVENC | NVOFA, 2x nearest.

python realcheck.py <work> [--clips xdcam,pal] [--presets Fast,Slower] [--frames 200] [--spots 6]
"""
import argparse, json, os, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get('QT_BUNDLE', os.path.join(HERE, 'bundle')).replace('\\', '/')
CLIPS = {
    'xdcam': (os.environ.get('QT_XDCAM', 'C:/path/to/xdcamhd_1920_25i.mxf'), 1920, 1080),
    'pal': (os.environ.get('QT_PAL', 'C:/path/to/pal_576i.mxf'), 720, 576),
}
VARIANTS = {
    'exact': ('exact GPU search (stock)', ''),
    'nvof': ('NVOFA medium', 'global QTGMC_gpu_nvof = true\n'),
    'nvenc': ('NVENC defaults', 'global QTGMC_gpu_nvenc = true\n'),
    'nvenc_yuv': ('NVENC channels yuv', 'global QTGMC_gpu_nvenc = true\nglobal QTGMC_gpu_nvenc_args = """, channels="yuv" """\n'),
}
ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('--clips', default='xdcam,pal')
ap.add_argument('--presets', default='Fast,Slower')
ap.add_argument('--frames', type=int, default=200)
ap.add_argument('--spots', type=int, default=6)
ap.add_argument('--keys', default=','.join(VARIANTS))
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
wk = a.work.replace('\\', '/')


def body(clip, preset, key):
    f, w, h = CLIPS[clip]
    return ('Import("%s/QTGMC_gpu_load.avsi")\nQTGMC_gpu_on()\n%s'
            'src = FFVideoSource("%s", cachefile="%s/%s.ffindex").Trim(0, %d).AssumeTFF()\n'
            'q = src.QTGMC(Preset="%s").ConvertToYV12()\n' % (BUNDLE, VARIANTS[key][1], f, wk, clip, a.frames - 1, preset))


def ff(p, args, **kw):
    return subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', p] + args, **kw)


out = {}
for clip in a.clips.split(','):
    f, W, H = CLIPS[clip]
    FS = W * H * 3 // 2
    for preset in a.presets.split(','):
        tag = '%s_%s' % (clip, preset)
        cache = os.path.join(a.work, tag + '.json')
        if os.path.exists(cache):
            out[tag] = json.load(open(cache))
            continue
        r = {'time': {}}
        for key in a.keys.split(','):
            p = os.path.join(a.work, '%s_%s.avs' % (tag, key))
            open(p, 'w').write(body(clip, preset, key) + 'return q.Prefetch(4)\n')
            best = 1e9
            for _ in range(2):
                t = time.perf_counter()
                rr = ff(p, ['-f', 'null', '-'], capture_output=True)
                if rr.returncode:
                    print(tag, key, rr.stderr.decode()[-300:])
                best = min(best, time.perf_counter() - t)
            r['time'][key] = best
            print(tag, key, '%.2f s' % best, flush=True)
        # differences NVENC vs exact per 16x16 block (luma mean abs), whole run
        p = os.path.join(a.work, '%s_diff.avs' % tag)
        txt = body(clip, preset, 'exact').replace('q = src.QTGMC', 'qe = src.QTGMC')
        txt = ('Import("%s/QTGMC_gpu_load.avsi")\nQTGMC_gpu_on()\n' % BUNDLE +
               'src = FFVideoSource("%s", cachefile="%s/%s.ffindex").Trim(0, %d).AssumeTFF()\n' % (f, wk, clip, a.frames - 1) +
               'qe = src.QTGMC(Preset="%s").ConvertToYV12()\n' % preset +
               'global QTGMC_gpu_nvenc = true\n' +
               'qn = src.QTGMC(Preset="%s").ConvertToYV12()\n' % preset +
               'return Interleave(qe, qn).Prefetch(4)\n')
        open(p, 'w').write(txt)
        pr = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', p, '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
        cand = []
        k = 0
        nb = H // 16
        while True:
            e = pr.stdout.read(FS)
            n = pr.stdout.read(FS)
            if len(n) < FS:
                break
            d = np.abs(np.frombuffer(e, np.uint8)[:W * H].astype(np.int16) - np.frombuffer(n, np.uint8)[:W * H]).reshape(H, W)[:nb * 16]
            bd = d.reshape(nb, 16, W // 16, 16).mean(axis=(1, 3))
            for idx in np.argsort(bd, axis=None)[::-1][:3]:
                by, bx = np.unravel_index(idx, bd.shape)
                cand.append((float(bd[by, bx]), k, int(bx), int(by)))
            k += 1
        pr.wait()
        cand.sort(reverse=True)
        spots = []
        for v, fr, bx, by in cand:
            if any(abs(fr - f2) < 10 for _, f2, _, _ in spots):
                continue
            spots.append((v, fr, bx, by))
            if len(spots) >= a.spots:
                break
        r['spots'] = spots
        # crops: 192x144 around the block (PAL: 160x120), 2x nearest, exact | NVENC | NVOFA
        cw, ch = (192, 144) if W > 1000 else (160, 120)
        for i, (v, fr, bx, by) in enumerate(spots):
            x = min(max(bx * 16 + 8 - cw // 2, 0), W - cw) // 2 * 2
            y = min(max(by * 16 + 8 - ch // 2, 0), H - ch) // 2 * 2
            p = os.path.join(a.work, '%s_crop.avs' % tag)
            txt = ('Import("%s/QTGMC_gpu_load.avsi")\nQTGMC_gpu_on()\n' % BUNDLE +
                   'src = FFVideoSource("%s", cachefile="%s/%s.ffindex").Trim(0, %d).AssumeTFF()\n' % (f, wk, clip, a.frames - 1))
            for key, gl in (('exact', ''), ('nvenc', 'global QTGMC_gpu_nvenc = true\n'), ('nvof', 'global QTGMC_gpu_nvenc = false\nglobal QTGMC_gpu_nvof = true\n')):
                txt += gl + 'c_%s = src.QTGMC(Preset="%s").ConvertToYV12().Trim(%d, -1).Crop(%d, %d, %d, %d).PointResize(%d, %d)\n' % (
                    key, preset, fr, x, y, cw, ch, cw * 2, ch * 2)
            txt += 'return StackHorizontal(c_exact, c_nvenc, c_nvof)\n'
            open(p, 'w').write(txt)
            png = os.path.join(a.work, '%s_spot%d_f%d.png' % (tag, i + 1, fr))
            ff(p, ['-frames:v', '1', '-y', png], capture_output=True)
        out[tag] = r
        json.dump(r, open(cache, 'w'))

print('\n| clip, preset | ' + ' | '.join(VARIANTS[k][0] for k in a.keys.split(',')) + ' |\n|---|' + '---:|' * len(a.keys.split(',')))
for tag, r in out.items():
    print('| %s | %s |' % (tag, ' | '.join('%.2f s' % r['time'][k] for k in a.keys.split(','))))
for tag, r in out.items():
    print(tag, 'spots (diff, frame, block x, y):', ', '.join('%.0f f%d (%d,%d)' % s for s in r['spots']))
