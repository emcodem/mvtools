"""Build a redistributable UConvert folder: uconvert.avsi + dlls/ with every plugin it needs.

The plugin set is read from load_plugins.avs (LoadPlugin / Import lines, variables FF and T resolved). Each plugin
DLL is copied to <out>/dlls/ together with the non-system DLLs it links to (PE import table, followed
recursively, looked up next to the plugin) and licence/readme files found next to it. uconvert.avsi loads
everything from its own folder (ScriptDir() + "dlls\\") when that folder exists.

usage: python -I build_dist.py [--out dist/UConvert]
"""
import os, re, sys, glob, shutil, struct, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SYS32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
# provided by the system / the VC++ runtime that AviSynth+ itself needs; reported, not copied
RUNTIME = re.compile(r"^(api-ms-win-.*|ext-ms-.*|msvcp\d+.*|vcruntime\d+.*|concrt\d+|vcomp\d+|ucrtbase)\.dll$", re.I)
LICENCE = re.compile(r"(licen[cs]e|copying|copyright|readme|gpl|lgpl|notice|authors).*", re.I)
# this repository (mvtools fork): licence and documentation of the plugins built from it
REPO = os.path.dirname(HERE)
# licence / readme files of a plugin's own distribution that are not next to the DLL, relative to its folder
EXTRA = {
    "LSMASHSource": ["../AviSynth/LICENSE", "../AviSynth/README.md"],
}
# licence / documentation from this repository (paths relative to the repository root)
MVTOOLS_DOCS = ["Documentation/gpl.txt"]
REPO_DOCS = {
    "mvtools2": MVTOOLS_DOCS + ["Documentation/gpu.md", "Documentation/mvtools2.html"],
    "cudafilters": MVTOOLS_DOCS + ["cudafilters/README.md"],
    "splatfps": MVTOOLS_DOCS + ["Documentation/splatfps.md"],
    "CudaConvolution": MVTOOLS_DOCS + ["cudaconvolution/THIRD_PARTY.md", "cudaconvolution/docs/usage.md"],
}
# where each component comes from (written to licences/<name>/SOURCE.txt): public download locations only
MVTOOLS_URL = "https://github.com/emcodem/mvtools (branch gpu_work)"
AVS_STUFF = "https://github.com/realfinder/AVS-Stuff/blob/master/avs%202.6%20and%20up/"
SOURCES = {
    "mvtools2": MVTOOLS_URL,
    "cudafilters": MVTOOLS_URL,
    "splatfps": MVTOOLS_URL,
    "CudaConvolution": MVTOOLS_URL + ", folder cudaconvolution",
    "QTGMC_gpu": AVS_STUFF + "QTGMC.avsi (QTGMC 3.393s), patched by qtgmc/make_qtgmc_gpu.py of " + MVTOOLS_URL,
    "Zs_RF_Shared": AVS_STUFF + "Zs_RF_Shared.avsi",
    "mxfsource": "https://github.com/emcodem/avisynth_mxfsource",
}
# DLLs a plugin loads at run time from its own folder (not in its import table): copied next to it
RUNTIME_DLLS = {
    "mvtools2": ["mvtools_cuda.dll", "avstp.dll"],
}
# files uconvert.avsi reads from its own folder
SIDE_FILES = ["uc_interlace.json"]
# readme-like files next to a plugin that are not licences (the GPU bench package's own instructions)
NOT_LICENCE = {"readme.txt", "readme_bench.txt"}


def pe_imports(path):
    """Names of the DLLs in a PE file's import and delay-import tables."""
    d = open(path, "rb").read()
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    nsec = struct.unpack_from("<H", d, pe + 6)[0]
    optsz = struct.unpack_from("<H", d, pe + 20)[0]
    opt = pe + 24
    dd = opt + (112 if struct.unpack_from("<H", d, opt)[0] == 0x20B else 96)
    secs = [struct.unpack_from("<IIII", d, pe + 24 + optsz + i * 40 + 8) for i in range(nsec)]  # vsize, va, rawsize, rawptr

    def off(rva):
        for vs, va, rs, rp in secs:
            if va <= rva < va + max(vs, rs):
                return rva - va + rp
        return None

    def cstr(o):
        return d[o:d.index(b"\0", o)].decode("ascii", "replace")

    names = []
    for idx, size in ((1, 20), (13, 32)):  # import table, delay-import table
        rva = struct.unpack_from("<I", d, dd + idx * 8)[0]
        o = off(rva) if rva else None
        while o is not None:
            name_rva = struct.unpack_from("<I", d, o + (12 if idx == 1 else 4))[0]
            if not name_rva:
                break
            names.append(cstr(off(name_rva)))
            o += size
    return names


def plugin_list(avs):
    """(kind, path) for every LoadPlugin / Import in load_plugins.avs."""
    txt = open(avs).read()
    var = dict(re.findall(r'^\s*(\w+)\s*=\s*"([^"]*)"', txt, re.M))
    out = []
    for kind, arg in re.findall(r'^\s*(LoadPlugin|Import)\((.*)\)\s*$', txt, re.M):
        parts = [p.strip() for p in arg.split("+")]
        out.append((kind, "".join(var[p] if p in var else p.strip('"') for p in parts)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "dist", "UConvert"))
    a = ap.parse_args()
    dlls = os.path.join(a.out, "dlls")
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(dlls)
    plugins, scripts, runtime, missing, copied, nolic = [], [], set(), set(), {}, []

    def copy(src, dst_name=None):
        dst = os.path.join(dlls, dst_name or os.path.basename(src))
        if os.path.basename(dst).lower() in copied:
            if os.path.normcase(copied[os.path.basename(dst).lower()]) != os.path.normcase(src):
                sys.exit(f"name clash in dlls/: {src} vs {copied[os.path.basename(dst).lower()]}")
            return False
        shutil.copy2(src, dst)
        copied[os.path.basename(dst).lower()] = src
        return True

    def with_deps(path):
        todo = [path]
        while todo:
            p = todo.pop()
            if not copy(p):
                continue
            for imp in pe_imports(p):
                local = os.path.join(os.path.dirname(p), imp)
                if RUNTIME.match(imp):
                    runtime.add(imp.lower())
                elif os.path.isfile(local):
                    todo.append(local)
                elif not os.path.isfile(os.path.join(SYS32, imp)):
                    missing.add(f"{imp} (needed by {os.path.basename(p)})")

    for kind, path in plugin_list(os.path.join(HERE, "load_plugins.avs")):
        if not os.path.isfile(path):
            sys.exit(f"not found: {path}")
        if kind == "LoadPlugin":
            with_deps(path)
            plugins.append(os.path.basename(path))
            for dll in RUNTIME_DLLS.get(os.path.splitext(os.path.basename(path))[0], []):
                rt = os.path.join(os.path.dirname(path), dll)
                if not os.path.isfile(rt):
                    sys.exit(f"not found: {rt} (loaded at run time by {os.path.basename(path)})")
                with_deps(rt)
        else:
            copy(path)
            scripts.append(os.path.basename(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        lic_dir = os.path.join(a.out, "licences", stem)
        near = glob.glob(os.path.join(os.path.dirname(path), "*")) + glob.glob(os.path.join(os.path.dirname(os.path.dirname(path)), "*"))
        extra = [os.path.normpath(os.path.join(os.path.dirname(path), f)) for f in EXTRA.get(stem, [])]
        for f in REPO_DOCS.get(stem, []):
            extra.append(os.path.join(REPO, f))
            if not os.path.isfile(extra[-1]):
                sys.exit(f"not found in the repository: {f}")
        for f in [f for f in near if LICENCE.match(os.path.basename(f)) and os.path.basename(f).lower() not in NOT_LICENCE] + extra:
            if os.path.isfile(f) and not f.lower().endswith((".dll", ".avsi")):
                os.makedirs(lic_dir, exist_ok=True)
                shutil.copy2(f, lic_dir)
        if stem in SOURCES:
            os.makedirs(lic_dir, exist_ok=True)
            with open(os.path.join(lic_dir, "SOURCE.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"{stem}: {SOURCES[stem]}\n")
        if not os.path.isdir(lic_dir):
            nolic.append(stem)

    shutil.copy2(os.path.join(HERE, "uconvert.avsi"), a.out)
    for f in SIDE_FILES:
        shutil.copy2(os.path.join(HERE, f), a.out)
    open(os.path.join(dlls, "uc_load.avs"), "w").write(
        "# Generated by build_dist.py: loads UConvert's plugins from this folder (imported by uconvert.avsi).\n"
        "uc_dlls = ScriptDir()\n" + "".join(f'LoadPlugin(uc_dlls + "{p}")\n' for p in plugins) +
        "".join(f'Import(uc_dlls + "{s}")\n' for s in scripts))
    print(f"{a.out}\n  plugins: {', '.join(plugins)}\n  scripts: {', '.join(scripts)}")
    print(f"  dependencies copied: {', '.join(sorted(set(copied) - {p.lower() for p in plugins} - {s.lower() for s in scripts}))}")
    print(f"  system runtime needed (not copied): {', '.join(sorted(runtime))}")
    if nolic:
        print(f"  no licence file found for: {', '.join(nolic)}")
    if missing:
        print("  NOT FOUND (must exist on the target system): " + "; ".join(sorted(missing)))


if __name__ == "__main__":
    main()
