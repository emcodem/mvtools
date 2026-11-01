"""baseline.py: framemd5 of compare.py's chains in AviSynth+ only, kept in a folder, to compare two builds (e.g. before
and after a merge, or ours against plain upstream; REBASE_PLAN.md step 0).

usage: python baseline.py run <plugin dir> <out dir> [group | chain ...]
       python baseline.py diff <dir A> <dir B>
groups: cpu     compare.py's CPU chains (`, gpu=false` removed, so they also run on builds without the gpu parameter)
        gpucpu  the GPU chains with `gpu=true` removed: the CPU result each GPU chain must equal (Documentation/gpu.md)
        gpu     the GPU chains as they are
        cf      the cudafilters chains (cudafilters.dll in the plugin dir)
        default: cpu gpucpu gpu, plus cf when cudafilters.dll is in the plugin dir
`run` also reports, in the same folder, every GPU chain against its gpucpu twin (must be `same`).
env: MVPF_CLIPS, MVPF_LSMASH, AVSMD5 as in compare.py (the plugin dir comes from the command line).
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare  # noqa: E402  (chains, sources, avsmd5 location)

CPU_SUFFIX = '@cpu'


def gpu_names():
    return [n for n in compare.GPU_CHAINS if n != 'lin_gpu']  # lin_gpu's CPU twin is lin_cpu


def avs_text(name, plugins):
    """The AviSynth script of a chain (compare.run_chain's form), or of a GPU chain's CPU twin (name + '@cpu')."""
    cpu_twin = name.endswith(CPU_SUFFIX)
    base = name[:-len(CPU_SUFFIX)] if cpu_twin else name
    src, body, _ = compare.CHAINS[base]
    avs_src = compare.SOURCES[src][0]
    pre = 'LoadPlugin("%s")\n' % compare.LSMASH if compare.WIN else ''
    if body is None:  # lin_cpu / lin_gpu
        text = pre + compare.LIN_AVS.format(plugins=plugins, mvtools2=compare.lib_name('mvtools2'), src=avs_src,
                                            gpu='true' if base == 'lin_gpu' else 'false')
    else:
        cf = base in compare.CF_CHAINS
        text = ('%sLoadPlugin("%s/%s")\n%sc = %s\n%sreturn out\n'
                % (pre, plugins, compare.lib_name('mvtools2'),
                   'LoadPlugin("%s/%s")\n' % (plugins, compare.lib_name('cudafilters')) if cf else '', avs_src, body))
    text = re.sub(r',\s*gpu=false', '', text)  # the default; builds without the parameter accept the chain
    if cpu_twin:
        text = re.sub(r',\s*gpu=true', '', text)
    return text


def expand(args, plugins):
    groups = {
        'cpu': [n for n in compare.CPU_NAMES],
        'gpucpu': [n + CPU_SUFFIX for n in gpu_names()],
        'gpu': list(compare.GPU_CHAINS),
        'cf': list(compare.CF_CHAINS),
    }
    if not args:
        args = ['cpu', 'gpucpu', 'gpu']
        if os.path.exists(os.path.join(plugins, compare.lib_name('cudafilters'))):
            args.append('cf')
    names = []
    for a in args:
        for n in groups.get(a, [a]):
            if n not in names:
                names.append(n)
    return names


def md5_lines(path):
    with open(path) as fh:
        return fh.read().splitlines()[3:]  # past the header (compare.py)


def cmp_lines(a, b):
    if a == b:
        return 'same', '%d frames' % len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return 'DIFF', '%d of %d lines differ, first at line %d' % (diff, max(len(a), len(b)), first)


def run(plugins, out, args):
    plugins = os.path.abspath(plugins).replace('\\', '/')
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, MVPF_PLUGIN_DIR=plugins)
    if compare.WIN:  # AviSynth+ on Windows leaves the unused vector-header bytes uninitialised (avsmd5.py)
        env['AVSMD5_ZERO_VECTOR_HEADER'] = '1'
    acmd = [sys.executable, compare.AVSMD5] if compare.AVSMD5.endswith('.py') else [compare.AVSMD5]
    names = expand(args, plugins)
    # the plugin folder's libraries: avstp.dll there or not changes MAnalyse's output (slices with avstp threads), so a
    # diff between two folders made with different libraries is not a code difference
    with open(os.path.join(out, 'plugins.txt'), 'w') as fh:
        for f in sorted(os.listdir(plugins)):
            if f.lower().endswith(('.dll', '.so')):
                fh.write('%-28s %12d\n' % (f, os.path.getsize(os.path.join(plugins, f))))
    failed = 0
    for name in names:
        fname = name.replace('@', '_at_')
        avs = os.path.join(out, fname + '.avs')
        md5 = os.path.join(out, fname + '.md5')
        with open(avs, 'w') as fh:
            fh.write(avs_text(name, plugins))
        t0 = time.time()
        r = subprocess.run(acmd + [avs, md5], capture_output=True, text=True, env=env, cwd=HERE)
        dt = time.time() - t0
        if r.returncode:
            failed += 1
            if os.path.exists(md5):
                os.remove(md5)
            with open(os.path.join(out, fname + '.err'), 'w') as fh:
                fh.write(r.stderr)
            print('%-28s ERROR %6.1f s  %s' % (name, dt, r.stderr.strip().splitlines()[-1:] or ''), flush=True)
        else:
            print('%-28s ok    %6.1f s  %d frames' % (name, dt, len(md5_lines(md5))), flush=True)
    # GPU chains against their CPU twins
    for name in gpu_names():
        g, c = (os.path.join(out, n.replace('@', '_at_') + '.md5') for n in (name, name + CPU_SUFFIX))
        if os.path.exists(g) and os.path.exists(c):
            status, info = cmp_lines(md5_lines(g), md5_lines(c))
            print('gpu = cpu  %-24s %-5s %s' % (name, status, info), flush=True)
            failed += status != 'same'
    return 1 if failed else 0


def avstp_present(d):
    p = os.path.join(d, 'plugins.txt')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return any(line.lower().startswith(('avstp.dll', 'libavstp.so')) for line in fh)


def diff(a_dir, b_dir):
    names = sorted({f[:-4] for d in (a_dir, b_dir) for f in os.listdir(d) if f.endswith('.md5')})
    pa, pb = avstp_present(a_dir), avstp_present(b_dir)
    if pa is not None and pb is not None and pa != pb:
        print('WARNING: avstp only in %s: MAnalyse slices differently, vectors differ for that reason alone'
              % (a_dir if pa else b_dir))
    bad = 0
    for n in names:
        a, b = os.path.join(a_dir, n + '.md5'), os.path.join(b_dir, n + '.md5')
        if not (os.path.exists(a) and os.path.exists(b)):
            print('%-28s only in %s' % (n.replace('_at_', '@'), a_dir if os.path.exists(a) else b_dir))
            continue
        status, info = cmp_lines(md5_lines(a), md5_lines(b))
        print('%-28s %-5s %s' % (n.replace('_at_', '@'), status, info))
        bad += status != 'same'
    return 1 if bad else 0


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == 'run':
        return run(sys.argv[2], sys.argv[3], sys.argv[4:])
    if len(sys.argv) == 4 and sys.argv[1] == 'diff':
        return diff(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
