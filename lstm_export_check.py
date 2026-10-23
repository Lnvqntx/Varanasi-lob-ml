"""Export check for the 2-layer LSTM (export_v4.py + solution_v4.py), CPU only.

1. An untrained LSTM (hidden 128, torch seed 0, input stats and column lists from the v7 checkpoint) is
   saved to ~/work/checkpoints/lstm_untrained_h128_seed0.pt and exported int8 (--cell matmul) to
   ~/work/submission_lstm_untrained_int8 for the speed check (bench_ens.py).
2. Correctness: the 50-step smoke LSTM checkpoint is exported fp32 and int8 (--cell matmul); each runs
   through solution.py row by row on the first STEPS rows of valid sequence 0 and is compared with the
   PyTorch LSTM on the same rows from the .npy cache.

  python lstm_export_check.py
"""
import export_v4  # noqa: I001  (sets CUDA_VISIBLE_DEVICES="" before torch is imported)

import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

import train_v4m
from bench_vs_baseline import load_model

STARTER = "/home/jovyan/work/wnn/wnn_connectome_starterpack"
sys.path.insert(0, STARTER)
from utils import FEATURE_COLUMNS, DataPoint  # noqa: E402

CKPT_DIR = "/home/jovyan/work/checkpoints"
V7 = f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt"
SMOKE = f"{CKPT_DIR}/verify_v4m_lstm_smoke_best.pt"
UNTRAINED = f"{CKPT_DIR}/lstm_untrained_h128_seed0.pt"
OUT = "/home/jovyan/work"
STEPS = 3000


def untrained():
    v7 = torch.load(V7, map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    model = train_v4m.LSTMModel(hidden=128)
    ck = {"model": model.state_dict(), "stats": v7["stats"], "args": dict(v7["args"], arch="lstm"), "step": 0,
          "price_cols": v7["price_cols"], "other_cols": v7["other_cols"], "flag_cols": v7["flag_cols"]}
    torch.save(ck, UNTRAINED)
    return ck


def onnx_preds(d, feats, seq):
    m = load_model(d)
    return np.stack([m.predict(DataPoint(seq, s, True, feats[s])) for s in range(STEPS)])


def main():
    for p in [UNTRAINED, f"{OUT}/submission_lstm_untrained_int8", f"{OUT}/submission_lstm_smoke_fp32",
              f"{OUT}/submission_lstm_smoke_int8"]:
        if os.path.exists(p):
            sys.exit(f"{p} exists; not overwriting")
    export_v4.export(untrained(), f"{OUT}/submission_lstm_untrained_int8", None, "matmul", True)

    ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
    assert ck["args"]["arch"] == "lstm"
    export_v4.export(ck, f"{OUT}/submission_lstm_smoke_fp32", None, "matmul", False)
    export_v4.export(ck, f"{OUT}/submission_lstm_smoke_int8", None, "matmul", True)

    t = pq.ParquetFile(f"{STARTER}/datasets/valid.parquet").read_row_group(0, use_threads=False)
    feats = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    seq = int(t["seq_ix"][0].as_py())
    valid = train_v4m.load_split("valid")
    model = train_v4m.LSTMModel(hidden=ck["args"]["hidden"])
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        x = train_v4m.build_inputs(torch.from_numpy(np.array(valid["prices"][:1, :STEPS])),
                                   torch.from_numpy(np.array(valid["features"][:1, :STEPS])), ck["stats"])
        ref = model(x)[0][0].numpy()
    print(f"smoke LSTM (step {ck['step']}), valid sequence {seq}, first {STEPS} rows; PyTorch pred std {ref.std(0)}")
    for kind in ["fp32", "int8"]:
        p = onnx_preds(f"{OUT}/submission_lstm_smoke_{kind}", feats, seq)
        d = np.abs(p - ref)
        print(f"  ONNX {kind} vs PyTorch: max abs diff {d.max():.3e}, mean {d.mean():.3e}, "
              f"corr {np.corrcoef(p.ravel(), ref.ravel())[0, 1]:.6f}")


if __name__ == "__main__":
    main()
