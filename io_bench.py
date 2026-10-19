"""Read-throughput benchmark for the training loader's access pattern (CPU only, read-only on the cache).

train_v2m.py reads one batch = 16 random train sequences, each WHOLE (all 20,000 steps), from four .npy
memmaps (prices f32 3.52 MB, features f16 2.72 MB, targets f32 0.16 MB, maskp f16 0.04 MB per sequence,
6.44 MB in all; 103 MB per batch, which feeds 20 optimizer steps of 1000-step chunks from RAM).
read_batch sorts the indices and copies v[idx] for each array in one Prefetcher thread.

Methods, each on fresh sequences (never re-read) whose byte ranges are first dropped from the page cache
with posix_fadvise(DONTNEED) on exactly those ranges, so every read comes from the disk:
  read_batch    the original function, one thread, batch by batch
  mmap xN       per-(sequence, array) copies out of the memmaps on a pool of N threads
  pread xN      per-(sequence, array) os.preadv of the whole contiguous row on a pool of N threads

  python io_bench.py [--batches 6]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CACHE = "/home/jovyan/work/cache"
KEYS = ["prices", "features", "targets", "maskp"]
BATCH = 16
STEPS_PER_BATCH = 20  # 20,000 steps / 1000-step chunks


def load():
    return {k: np.load(f"{CACHE}/train_{k}.npy", mmap_mode="r") for k in KEYS}


def row_bytes(a):
    return int(np.prod(a.shape[1:])) * a.itemsize


def drop_cache(data, fds, seqs):
    for k, a in data.items():
        rb = row_bytes(a)
        for i in seqs:
            os.posix_fadvise(fds[k], a.offset + int(i) * rb, rb, os.POSIX_FADV_DONTNEED)


def read_batch(data, idx):  # as train_v2m.read_batch
    idx = np.sort(np.asarray(idx))
    return {k: np.ascontiguousarray(v[idx]) for k, v in data.items()}


def pread_row(fd, a, i):
    buf = np.empty(a.shape[1:], dtype=a.dtype)
    mv, off, done = memoryview(buf).cast("B"), a.offset + int(i) * row_bytes(a), 0
    while done < len(mv):
        done += os.preadv(fd, [mv[done:]], off + done)
    return buf


def run(method, n_threads, data, fds, batches):
    t = time.perf_counter()
    if method == "read_batch":
        for b in batches:
            read_batch(data, b)
    else:
        tasks = [(k, i) for b in batches for i in np.sort(b) for k in KEYS]
        if method == "mmap":
            f = lambda ki: np.array(data[ki[0]][ki[1]])  # noqa: E731
        else:
            f = lambda ki: pread_row(fds[ki[0]], data[ki[0]], ki[1])  # noqa: E731
        with ThreadPoolExecutor(n_threads) as ex:
            for _ in ex.map(f, tasks):
                pass
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=6, help="batches per configuration")
    args = ap.parse_args()
    data = load()
    fds = {k: os.open(f"{CACHE}/train_{k}.npy", os.O_RDONLY) for k in KEYS}
    mb_per_batch = BATCH * sum(row_bytes(a) for a in data.values()) / 1e6
    configs = [("read_batch", 1)] + [(m, n) for m in ["mmap", "pread"] for n in [1, 2, 4, 8, 16]]
    perm = np.random.default_rng(12345).permutation(data["prices"].shape[0])
    per = args.batches * BATCH
    assert len(configs) * per <= len(perm)
    print(f"{mb_per_batch:.1f} MB per batch of {BATCH} whole sequences; {args.batches} batches "
          f"({args.batches * mb_per_batch:.0f} MB) per configuration, fresh sequences each, cache dropped first")
    print(f"{'method':<11} {'threads':>7} {'seconds':>8} {'MB/s':>7} {'batches/s':>9} {'steps/s ceiling':>15}")
    for c, (method, n) in enumerate(configs):
        seqs = perm[c * per:(c + 1) * per]
        drop_cache(data, fds, seqs)
        sec = run(method, n, data, fds, [seqs[j:j + BATCH] for j in range(0, per, BATCH)])
        mbs = args.batches * mb_per_batch / sec
        print(f"{method:<11} {n:>7} {sec:>8.1f} {mbs:>7.0f} {args.batches / sec:>9.2f} "
              f"{args.batches / sec * STEPS_PER_BATCH:>15.0f}", flush=True)


if __name__ == "__main__":
    main()
