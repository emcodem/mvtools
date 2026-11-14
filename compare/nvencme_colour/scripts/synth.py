"""Synthetic check: flat luma, a texture in U only, shifted by (8, 4) luma px between two frames. Prints per engine and
colour mode the share of 32/16 blocks that get that vector.   python synth.py <work folder>"""
import sys, os, subprocess, numpy as np
import variants as V
work = sys.argv[1]
# flat luma, chroma-only texture shifted by (8, 4) luma px between frames 0 and 1 (4:2:2: U shift 4, 4)
body = V.head() + '''
m = LWLibavVideoSource("%s", cachefile="%s.lwi").Trim(40, 40).ConvertBits(8).BilinearResize(1920, 1080)
t = m.ExtractY().Blur(0.5)
u0 = t.Crop(16, 16, 960, 1048)
u1 = t.Crop(12, 12, 960, 1048)
yf = BlankClip(u0, width=1920, color_yuv=$808080, pixel_type="Y8")
v = BlankClip(u0, pixel_type="Y8", color_yuv=$808080)
f0 = CombinePlanes(yf, u0, v, planes="YUV", source_planes="YYY", pixel_type="YUV422P8")
f1 = CombinePlanes(yf, u1, v, planes="YUV", source_planes="YYY", pixel_type="YUV422P8")
c = f0 + f1
bv = NVENCMEProto(c, isb=false, blksize=32, overlap=16, pel=2, hpad=16, vpad=16, preset=4, dx12=DX)
return bv.Trim(1, 1)
''' % (V.SEGFILE, V.SEGFILE)
for dx in (0, 8):
    for col in ('', 'ynb', 'yuv', 'rgb'):
        avs = os.path.join(work, 'synth.avs')
        open(avs, 'w').write(body.replace('DX', str(dx)))
        env = dict(os.environ); env.pop('NVME_COLOUR', None)
        if col: env['NVME_COLOUR'] = col
        b = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', avs, '-f', 'rawvideo', '-'], env=env, capture_output=True)
        if b.returncode: print(dx, col, b.stderr.decode()[-300:]); continue
        w = np.frombuffer(b.stdout, np.int32)[64:]
        v = w[3:3 + (w[0] - 3)].reshape(-1, 3)
        vx, vy = v[:, 0], v[:, 1]
        print('dx12=%d colour=%-4s  exact (-16,-8) or (16,8): %.1f %%   median (%d, %d)' % (dx, col or '-', 100 * np.mean((np.abs(vx) == 16) & (np.abs(vy) == 8)), np.median(vx), np.median(vy)))
