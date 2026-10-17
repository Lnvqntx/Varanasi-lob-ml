"""Time solution.predict() per row vs the starter baseline, same loop, same core, alternating rounds.

  taskset -c 31 python bench_vs_baseline.py DIR [DIR ...]
Uses real valid rows (first sequence) so the inputs look like the scorer's.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import sys
import time

import numpy as np
import pyarrow.parquet as pq

STARTER = "/home/jovyan/work/wnn/wnn_connectome_starterpack"
sys.path.insert(0, STARTER)
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, DataPoint  # noqa: E402

ROUNDS = 3


def load_model(d):
    spec = importlib.util.spec_from_file_location(f"sol_{abs(hash(d))}", os.path.join(d, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PredictionModel()


def time_one(model, feats, seq):
    t = time.perf_counter()
    for step in range(SEQUENCE_LENGTH):
        model.predict(DataPoint(seq, step, step >= 99, feats[step]))
    return (time.perf_counter() - t) / SEQUENCE_LENGTH * 1e6


def main():
    dirs = [f"{STARTER}/baseline"] + sys.argv[1:]
    t = pq.ParquetFile(f"{STARTER}/datasets/valid.parquet").read_row_group(0)
    feats = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    print(f"CPU affinity {sorted(os.sched_getaffinity(0))}, {ROUNDS} rounds x {SEQUENCE_LENGTH} rows each")
    models = {d: load_model(d) for d in dirs}
    res = {d: [] for d in dirs}
    for r in range(ROUNDS):
        for d in dirs:
            res[d].append(time_one(models[d], feats, 1000 * r + 1 + dirs.index(d)))
    base = np.median(res[dirs[0]])
    for d in dirs:
        med = np.median(res[d])
        print(f"{med:7.1f} us/row (rounds {', '.join(f'{x:.1f}' for x in res[d])})  "
              f"{med / base:5.2f}x baseline  -> {med * 39.4e6 / 60e6:5.1f} min/39.4M rows  {d}")


if __name__ == "__main__":
    main()
