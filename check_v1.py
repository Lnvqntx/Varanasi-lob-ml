"""Check submission_v1 on the first N valid sequences, CPU only, GPUs hidden.

1. Two independent runs of solution.PredictionModel through a replica of ScorerStepByStep.score
   (limited to N row groups), pinned to one CPU core: timing + bitwise determinism.
2. PyTorch reference (train_v1 model + build_inputs on the .npy cache, 1000-step chunks):
   max |ONNX - PyTorch| on prediction rows.
3. Score both with utils.GlobalAccumulator on is_scored & need_prediction rows.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"

import argparse
import sys
import time

import numpy as np
import pyarrow.parquet as pq

STARTER = "/home/jovyan/work/wnn/wnn_connectome_starterpack"
SUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SUB", "submission_v1"))
sys.path.insert(0, STARTER)
sys.path.insert(0, SUB_DIR)
from utils import (FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, DataPoint,  # noqa: E402
                   GlobalAccumulator, validate_sequence)
from solution import PredictionModel  # noqa: E402

TEST_ROWS = 39_400_000
CKPT = os.environ.get("CKPT", "/home/jovyan/work/checkpoints/train_v1_best.pt")


def load_valid(n):
    pf = pq.ParquetFile(f"{STARTER}/datasets/valid.parquet")
    seqs = []
    for g in range(n):
        t = pf.read_row_group(g, use_threads=False)
        seq, need = validate_sequence(t)
        seqs.append(dict(
            seq=seq, need=need,
            features=np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32),
            targets=np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32),
            mask=t["is_scored"].to_numpy().astype(bool) & need))
    return seqs


def run_solution(seqs):
    """Same per-row loop as ScorerStepByStep.score; returns predictions and timings."""
    model = PredictionModel()
    preds, t_predict, t_loop = [], 0.0, 0.0
    for s in seqs:
        p = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        tl = time.perf_counter()
        for step in range(SEQUENCE_LENGTH):
            point = DataPoint(s["seq"], step, bool(s["need"][step]), s["features"][step])
            tp = time.perf_counter()
            value = model.predict(point)
            t_predict += time.perf_counter() - tp
            if not s["need"][step]:
                if value is not None:
                    raise ValueError("returned a value during warm-up")
            else:
                value = np.asarray(value, dtype=np.float32)
                if value.shape != (2,) or not np.isfinite(value).all():
                    raise ValueError("bad prediction")
                p[step] = value
        t_loop += time.perf_counter() - tl
        preds.append(p)
    return np.stack(preds), t_predict, t_loop


def score(seqs, preds):
    acc = GlobalAccumulator()
    for s, p in zip(seqs, preds):
        acc.add(s["targets"], p, s["mask"])
    return acc.result()


def torch_reference(seqs):
    import torch
    import train_v1  # sets CUDA_VISIBLE_DEVICES=0 on import; hide the GPUs again before CUDA init
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert torch.cuda.device_count() == 0, "GPU visible"
    torch.set_num_threads(4)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = train_v1.GRUModel(hidden=ck["args"].get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.cpu() for k, v in ck["stats"].items()}
    valid = train_v1.load_split("valid")
    seq_ix = np.load("/home/jovyan/work/cache/valid_seq_ix.npy")
    n = len(seqs)
    assert [int(x) for x in seq_ix[:n]] == [s["seq"] for s in seqs], "cache order != parquet order"
    b = train_v1.read_batch(valid, np.arange(n))
    out = []
    with torch.no_grad():
        x = train_v1.build_inputs(torch.from_numpy(b["prices"]), torch.from_numpy(b["features"]), stats)
        h = None
        for s in range(0, SEQUENCE_LENGTH, 1000):
            p, h = model(x[:, s:s + 1000], h)
            out.append(p)
    return torch.cat(out, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    seqs = load_valid(args.n)
    rows = args.n * SEQUENCE_LENGTH
    print(f"submission dir {SUB_DIR}; checkpoint {CKPT}; {args.n} valid sequences, {rows:,} rows")

    allowed = sorted(os.sched_getaffinity(0))
    core = allowed[-1]
    os.sched_setaffinity(0, {core})
    print(f"pinned to CPU core {core}")
    runs = []
    for r in range(2):
        preds, tp, tl = run_solution(seqs)
        runs.append(preds)
        print(f"run {r + 1}: predict() {tp / rows * 1e6:.1f} us/row, full scorer loop {tl / rows * 1e6:.1f} us/row "
              f"-> projected {TEST_ROWS:,} rows: {tp / rows * TEST_ROWS / 60:.1f} min (predict only), "
              f"{tl / rows * TEST_ROWS / 60:.1f} min (loop)")
    os.sched_setaffinity(0, set(allowed))
    need = np.stack([s["need"] for s in seqs])
    print(f"runs bitwise identical on prediction rows: {np.array_equal(runs[0][need], runs[1][need])}")

    ref = torch_reference(seqs)
    diff = np.abs(runs[0][need] - ref[need])
    print(f"ONNX vs PyTorch on {need.sum():,} prediction rows: max abs diff {diff.max():.3e}, mean {diff.mean():.3e}, "
          f"pred std {ref[need].std(0)}")

    r_onnx, r_torch = score(seqs, runs[0]), score(seqs, ref)
    for name, r in [("ONNX solution", r_onnx), ("PyTorch", r_torch)]:
        print(f"{name:>14}: wp={r['weighted_pearson']:.6f} (t0 {r['t0']:.6f}, t1 {r['t1']:.6f}), "
              f"scored rows {r['selected_rows']:,}")


if __name__ == "__main__":
    main()
