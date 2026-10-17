"""Export train_v1_best.pt to a single-step ONNX model + preprocessing stats, and zip the submission.

Needs the `onnx` package only for export (not at inference), e.g.:
  PYTHONPATH=<dir with onnx> CUDA_VISIBLE_DEVICES= python export_v1.py

Writes into submission_v1/ (next to solution.py):
  model.onnx  inputs  x (1,1,168) f32, h (2,1,256) f32;  outputs pred (1,1,2) f32, h_out (2,1,256) f32
  stats.npz   mean/std (156,), index arrays into the 112-feature state, fill value
and zips solution.py + model.onnx + stats.npz to ~/work/submissions/submission_v1.zip.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import zipfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/jovyan/work/wnn/wnn_connectome_starterpack")
from utils import FEATURE_COLUMNS  # noqa: E402

CKPT = "/home/jovyan/work/checkpoints/train_v1_best.pt"
SUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v1")
ZIP_PATH = "/home/jovyan/work/submissions/submission_v1.zip"
FILL = 5.19921875  # must match train_v1.FILL


class GRUModel(nn.Module):
    """Same module layout as train_v1.GRUModel so the state_dict loads unchanged."""

    def __init__(self, n_in=168, hidden=256, layers=2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return self.head(out), h


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"checkpoint step {ck['step']}, valid wp {ck['valid']['weighted_pearson']:.4f}")
    model = GRUModel()
    model.load_state_dict(ck["model"])
    model.eval()

    onnx_path = os.path.join(SUB_DIR, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, 1, 168), torch.zeros(2, 1, 256)), onnx_path,
                          opset_version=17, dynamo=False, input_names=["x", "h"],
                          output_names=["pred", "h_out"])

    price_cols, other_cols, flag_cols = ck["price_cols"], ck["other_cols"], ck["flag_cols"]
    np.savez(os.path.join(SUB_DIR, "stats.npz"),
             mean=ck["stats"]["mean"].cpu().numpy().astype(np.float32),
             std=ck["stats"]["std"].cpu().numpy().astype(np.float32),
             price_idx=np.array([FEATURE_COLUMNS.index(c) for c in price_cols], dtype=np.int64),
             other_idx=np.array([FEATURE_COLUMNS.index(c) for c in other_cols], dtype=np.int64),
             flag_idx=np.array([other_cols.index(c) for c in flag_cols], dtype=np.int64),
             fill=np.float32(FILL))

    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["solution.py", "model.onnx", "stats.npz"]:
            z.write(os.path.join(SUB_DIR, f), arcname=f)
    for f in ["solution.py", "model.onnx", "stats.npz"]:
        print(f"  {f}: {os.path.getsize(os.path.join(SUB_DIR, f)):,} bytes")
    print(f"zip {ZIP_PATH}: {os.path.getsize(ZIP_PATH):,} bytes")


if __name__ == "__main__":
    main()
