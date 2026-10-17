"""Faster export of train_v1_best.pt: preprocessing + explicit GRU cell math in one ONNX graph.

ORT's GRU operator costs ~170 us per single step on one core; the same math as MatMul/Sigmoid/Tanh
uses MLAS kernels. Preprocessing (train_v1.build_inputs, one row) runs inside the graph, with the
standardization stats as constants, so solution.py makes one session.run per row.

  inputs   state (1,112) f32, prev_prices (1,44) f32 (= this row's prices at step 0 -> diff 0), h (2,1,256) f32
  outputs  pred (1,2) f32, h_out (2,1,256) f32, prices (1,44) f32 (pass back as prev_prices)

Needs the `onnx` package only for export:
  PYTHONPATH=<dir with onnx> CUDA_VISIBLE_DEVICES= python export_v1_fast.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import zipfile

import numpy as np
import torch
import torch.nn as nn

from export_v1 import CKPT, FEATURE_COLUMNS, FILL

SUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v1_fast")
ZIP_PATH = "/home/jovyan/work/submissions/submission_v1_fast.zip"
H = 256


class FastStep(nn.Module):
    def __init__(self, ck):
        super().__init__()
        sd = ck["model"]
        self.register_buffer("price_idx", torch.tensor([FEATURE_COLUMNS.index(c) for c in ck["price_cols"]]))
        self.register_buffer("other_idx", torch.tensor([FEATURE_COLUMNS.index(c) for c in ck["other_cols"]]))
        self.register_buffer("flag_idx", torch.tensor([ck["other_cols"].index(c) for c in ck["flag_cols"]]))
        self.register_buffer("mean", ck["stats"]["mean"].cpu().float()[None])
        self.register_buffer("std", ck["stats"]["std"].cpu().float()[None])
        for l in range(2):
            self.register_buffer(f"wih{l}", sd[f"gru.weight_ih_l{l}"].t().contiguous())
            self.register_buffer(f"whh{l}", sd[f"gru.weight_hh_l{l}"].t().contiguous())
            self.register_buffer(f"bih{l}", sd[f"gru.bias_ih_l{l}"][None])
            self.register_buffer(f"bhh{l}", sd[f"gru.bias_hh_l{l}"][None])
        self.register_buffer("wout", sd["head.weight"].t().contiguous())
        self.register_buffer("bout", sd["head.bias"][None])

    def cell(self, x, h, l):
        gi = x @ getattr(self, f"wih{l}") + getattr(self, f"bih{l}")
        gh = h @ getattr(self, f"whh{l}") + getattr(self, f"bhh{l}")
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1 - z) * n + z * h

    def forward(self, state, prev_prices, h):
        prices = state.index_select(1, self.price_idx)
        other = state.index_select(1, self.other_idx).half().float()
        cont = torch.cat([prices, prices - prev_prices, other], dim=1)
        flags = (other.index_select(1, self.flag_idx).abs() == FILL).float()
        x = torch.cat([(cont - self.mean) / self.std, flags], dim=1)
        h0 = self.cell(x, h[0], 0)
        h1 = self.cell(h0, h[1], 1)
        pred = h1 @ self.wout + self.bout
        return pred, torch.stack([h0, h1], 0), prices


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"checkpoint step {ck['step']}, valid wp {ck['valid']['weighted_pearson']:.4f}")
    m = FastStep(ck).eval()
    os.makedirs(SUB_DIR, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(m, (torch.zeros(1, 112), torch.zeros(1, 44), torch.zeros(2, 1, H)),
                          os.path.join(SUB_DIR, "model.onnx"), opset_version=17, dynamo=False,
                          input_names=["state", "prev_prices", "h"], output_names=["pred", "h_out", "prices"])
    # Stats also stored on their own for reference; the graph carries them as constants.
    np.savez(os.path.join(SUB_DIR, "stats.npz"), mean=m.mean[0].numpy(), std=m.std[0].numpy(),
             price_idx=m.price_idx.numpy(), other_idx=m.other_idx.numpy(), flag_idx=m.flag_idx.numpy(),
             fill=np.float32(FILL))

    files = ["solution.py", "model.onnx", "stats.npz"]
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(SUB_DIR, f), arcname=f)
    for f in files:
        print(f"  {f}: {os.path.getsize(os.path.join(SUB_DIR, f)):,} bytes")
    print(f"zip {ZIP_PATH}: {os.path.getsize(ZIP_PATH):,} bytes")


if __name__ == "__main__":
    main()
