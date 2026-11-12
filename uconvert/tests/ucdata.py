"""Where the UConvert tests find their media and write their work files.

The test media (sd/, deint/, probes/, about 2 GB) and the work folders (work/uc_tests, work/uc_matrix) are not in
the repo. Their root folder comes from the environment variable UCONVERT_TEST_DATA, or else from the first line of
tests/data_root.txt (not tracked: one machine's path).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def data_root():
    root = os.environ.get("UCONVERT_TEST_DATA", "").strip()
    cfg = os.path.join(HERE, "data_root.txt")
    if not root and os.path.isfile(cfg):
        root = open(cfg, encoding="utf-8").readline().strip()
    if not root:
        raise SystemExit("UConvert test data folder not set: set UCONVERT_TEST_DATA or write the folder into " + cfg)
    if not os.path.isdir(root):
        raise SystemExit(f"UConvert test data folder not found: {root}")
    return os.path.abspath(root).replace("\\", "/")
