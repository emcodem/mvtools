"""Slow-motion clip of several variants with a labelled grid (default: exact | NVOFA medium | NVOFA fast).

python slowmo.py <work folder> [start s] [end s] [hold] [--variants key:label,...] [--cols n] [--name file]
  defaults: 1.7 2.6 4, exact/medium/fast in one row, ../img/slowmo_<start>-<end>.mp4

Inputs: <work>/<key>_crop.mp4 (render.ps1 / render_lambda.ps1: the 960x540 crop at x 2880, y 240 of the uc_FRC output).
Each panel: 120 px grid, cells A1..H5 (column letter, row number), header with the label, the time in the clip and
the output frame number. Every frame is held `hold` times; panels in rows of --cols.
"""
import argparse
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('work')
ap.add_argument('t0', nargs='?', type=float, default=1.7)
ap.add_argument('t1', nargs='?', type=float, default=2.6)
ap.add_argument('hold', nargs='?', type=int, default=4)
ap.add_argument('--variants', default='exact:exact search,medium:NVOFA medium,fast:NVOFA fast')
ap.add_argument('--cols', type=int, default=0)
ap.add_argument('--name', default='')
a = ap.parse_args()
variants = [tuple(v.split(':', 1)) for v in a.variants.split(',')]
cols = a.cols or len(variants)
name = a.name or 'slowmo_%.1f-%.1f' % (a.t0, a.t1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img', name + '.mp4')

FONT = 'C\\:/Windows/Fonts/consola.ttf'
CELL, W, H, HEAD = 120, 960, 540, 48

labels = []
for r in range((H + CELL - 1) // CELL):
    for c in range(W // CELL):
        labels.append("drawtext=fontfile='%s':text='%s%d':x=%d:y=%d:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=3"
                      % (FONT, chr(ord('A') + c), r + 1, c * CELL + 4, HEAD + r * CELL + 4))

chains = []
for i, (key, label) in enumerate(variants):
    f = ['trim=start=%g:end=%g' % (a.t0, a.t1),
         'pad=%d:%d:0:%d:color=black' % (W, H + HEAD, HEAD),
         'drawgrid=x=0:y=%d:w=%d:h=%d:t=1:c=yellow@0.6' % (HEAD, CELL, CELL)]
    f += labels
    f.append('drawbox=x=0:y=0:w=%d:h=%d:color=black:t=fill' % (W, HEAD))
    f.append("drawtext=fontfile='%s':text='%s':x=10:y=12:fontsize=26:fontcolor=white" % (FONT, label))
    # time in the clip and output frame number (23.976 fps), taken before the timestamps are stretched
    f.append("drawtext=fontfile='%s':text='t=%%{eif\\:floor(t)\\:d}.%%{eif\\:mod(round(t*1000),1000)\\:d\\:3} s  frame %%{eif\\:round(t*24000/1001)\\:d}':x=%d:y=14:fontsize=22:fontcolor=yellow"
             % (FONT, W - 340))
    f.append('setpts=%d*(PTS-STARTPTS)' % a.hold)
    chains.append('[%d:v]%s[v%d]' % (i, ','.join(f), i))
rows = []
for r in range(0, len(variants), cols):
    ids = list(range(r, min(r + cols, len(variants))))
    if len(ids) == 1:
        rows.append('[v%d]' % ids[0])
        continue
    chains.append('%shstack=%d[r%d]' % (''.join('[v%d]' % i for i in ids), len(ids), r))
    rows.append('[r%d]' % r)
if len(rows) == 1:
    chains.append('%sformat=yuv420p[out]' % rows[0])
else:
    chains.append('%svstack=%d,format=yuv420p[out]' % (''.join(rows), len(rows)))
graph = ';'.join(chains)

script = os.path.join(a.work, 'slowmo_graph.txt')
with open(script, 'w') as fh:
    fh.write(graph)
cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-y', '-loglevel', 'error']
for key, _ in variants:
    cmd += ['-i', os.path.join(a.work, key + '_crop.mp4')]
cmd += ['-/filter_complex', script, '-map', '[out]', '-r', '24000/1001', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', out]
subprocess.run(cmd, check=True)
print(os.path.normpath(out))
