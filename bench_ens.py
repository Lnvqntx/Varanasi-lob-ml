"""Time solution.predict() per row for several submission dirs on the first N valid sequences.

  taskset -c 31 python bench_ens.py --n 100 DIR [DIR ...]

Same loop as bench_vs_baseline.py (DataPoint per row, need_prediction = step >= 99, real valid rows),
but over N sequences, with the dirs alternating per sequence (order flipped every other sequence) so
drift hits all of them equally. Parquet reads are outside the timed region. Reports predict()-only and
whole-loop us/row and the platform projection (predict() us/row x 39.4M / 60e6 / 1.89).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from bench_vs_baseline import STARTER, load_model
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, DataPoint

TEST_ROWS = 39.4e6
PLATFORM_SPEEDUP = 1.89  # 59 us/row here -> 20m32s on the platform for the test set


def time_one(model, feats, seq):
    tp = 0.0
    t = time.perf_counter()
    for step in range(SEQUENCE_LENGTH):
        p = DataPoint(seq, step, step >= 99, feats[step])
        t0 = time.perf_counter()
        model.predict(p)
        tp += time.perf_counter() - t0
    return tp, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    pf = pq.ParquetFile(f"{STARTER}/datasets/valid.parquet")
    print(f"CPU affinity {sorted(os.sched_getaffinity(0))}, {args.n} valid sequences x {SEQUENCE_LENGTH} rows", flush=True)
    models = {d: load_model(d) for d in args.dirs}
    tp = {d: 0.0 for d in args.dirs}
    tl = {d: 0.0 for d in args.dirs}
    for g in range(args.n):
        t = pf.read_row_group(g, use_threads=False)
        feats = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        seq = int(t["seq_ix"][0].as_py())
        for d in (args.dirs if g % 2 == 0 else args.dirs[::-1]):
            a, b = time_one(models[d], feats, seq)
            tp[d] += a
            tl[d] += b
        if (g + 1) % 10 == 0:
            print(f"  {g + 1}/{args.n}: " + " | ".join(f"{tp[d] / ((g + 1) * SEQUENCE_LENGTH) * 1e6:.1f} us/row "
                                                      f"{os.path.basename(d.rstrip('/'))}" for d in args.dirs), flush=True)
    rows = args.n * SEQUENCE_LENGTH
    for d in args.dirs:
        p, l = tp[d] / rows * 1e6, tl[d] / rows * 1e6
        print(f"{p:6.1f} us/row predict(), {l:6.1f} us/row loop -> here {p * TEST_ROWS / 60e6:5.1f} min, "
              f"platform {p * TEST_ROWS / 60e6 / PLATFORM_SPEEDUP:5.1f} min  {d}")


if __name__ == "__main__":
    main()
