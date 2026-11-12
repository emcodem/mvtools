"""Per-frame MD5 of the reference model output (yuv422p10le bytes, Y then U then V) on the probe.
usage: CUDACONVOLUTION_DATA=<folder> python -I tools/model_md5.py <k0> <k1> > tests/data/model_balanced_uhd.md5"""
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pio import inp
from ref_model import frame
from multiprocessing import Pool
def md5(k):
    out = frame([inp(2*k, pl) for pl in range(3)], [inp(2*k+1, pl) for pl in range(3)])
    return hashlib.md5(b"".join(np.ascontiguousarray(p, dtype="<u2").tobytes() for p in out)).hexdigest()
if __name__ == "__main__":
    with Pool(max(1, os.cpu_count() // 2)) as pool:
        for h in pool.imap(md5, range(int(sys.argv[1]), int(sys.argv[2]))): print(h, flush=True)
