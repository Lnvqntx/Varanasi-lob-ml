"""Export a train_v1.py checkpoint (any hidden size) to one ONNX graph: preprocessing + GRU cells + head.

Per layer the GRU cell uses one fused MatMul over [x, h]:
  [x, h] @ [[W_ir W_iz | W_in |  0  ],
            [W_hr W_hz |  0   | W_hn]] + [b_ir+b_hr b_iz+b_hz | b_in | b_hn]
  -> r,z pre-activations | i_n | h_n,  n = tanh(i_n + r * h_n),  h' = n + z * (h - n)
which is torch.nn.GRU's math with fewer ORT nodes (per-node overhead dominates at batch 1).

  inputs   state (1,112) f32, prev_prices (1,44) f32 (= this row's prices at step 0 -> diff 0), h (L,1,H) f32
  outputs  pred (1,2) f32, h_out (L,1,H) f32, prices (1,44) f32 (pass back as prev_prices)

Usage (needs the `onnx` package for export only):
  PYTHONPATH=<dir with onnx> python export_fast.py --ckpt CKPT --out DIR --zip ZIP [--int8]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import shutil
import zipfile

import numpy as np
import torch
import torch.nn as nn

from export_v1 import FEATURE_COLUMNS, FILL

HERE = os.path.dirname(os.path.abspath(__file__))


class FastStep(nn.Module):
    def __init__(self, ck):
        super().__init__()
        sd = ck["model"]
        self.layers = sum(1 for k in sd if k.startswith("gru.weight_ih_l"))
        self.H = H = sd["gru.weight_hh_l0"].shape[1]
        self.register_buffer("price_idx", torch.tensor([FEATURE_COLUMNS.index(c) for c in ck["price_cols"]]))
        self.register_buffer("other_idx", torch.tensor([FEATURE_COLUMNS.index(c) for c in ck["other_cols"]]))
        self.register_buffer("flag_idx", torch.tensor([ck["other_cols"].index(c) for c in ck["flag_cols"]]))
        self.register_buffer("mean", ck["stats"]["mean"].cpu().float()[None])
        self.register_buffer("std", ck["stats"]["std"].cpu().float()[None])
        for l in range(self.layers):
            wih, whh = sd[f"gru.weight_ih_l{l}"], sd[f"gru.weight_hh_l{l}"]  # (3H, in), (3H, H), rows r,z,n
            bih, bhh = sd[f"gru.bias_ih_l{l}"], sd[f"gru.bias_hh_l{l}"]
            n_in = wih.shape[1]
            w = torch.zeros(n_in + H, 4 * H)
            w[:n_in, :2 * H] = wih[:2 * H].t()
            w[n_in:, :2 * H] = whh[:2 * H].t()
            w[:n_in, 2 * H:3 * H] = wih[2 * H:].t()
            w[n_in:, 3 * H:] = whh[2 * H:].t()
            b = torch.cat([bih[:2 * H] + bhh[:2 * H], bih[2 * H:], bhh[2 * H:]])
            self.register_buffer(f"w{l}", w)
            self.register_buffer(f"b{l}", b[None])
        self.register_buffer("wout", sd["head.weight"].t().contiguous())
        self.register_buffer("bout", sd["head.bias"][None])

    def cell(self, x, h, l):
        H = self.H
        g = torch.cat([x, h], dim=1) @ getattr(self, f"w{l}") + getattr(self, f"b{l}")
        rz = torch.sigmoid(g[:, :2 * H])
        n = torch.tanh(g[:, 2 * H:3 * H] + rz[:, :H] * g[:, 3 * H:])
        return n + rz[:, H:] * (h - n)

    def forward(self, state, prev_prices, h):
        prices = state.index_select(1, self.price_idx)
        other = state.index_select(1, self.other_idx).half().float()
        cont = torch.cat([prices, prices - prev_prices, other], dim=1)
        flags = (other.index_select(1, self.flag_idx).abs() == FILL).float()
        x = torch.cat([(cont - self.mean) / self.std, flags], dim=1)
        hs = []
        for l in range(self.layers):
            x = self.cell(x, h[l], l)
            hs.append(x)
        return x @ self.wout + self.bout, torch.stack(hs, 0), prices


def export(ck, out_dir, zip_path=None, int8=False):
    m = FastStep(ck).eval()
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    fp32_path = os.path.join(out_dir, "model_fp32.onnx") if int8 else onnx_path
    with torch.no_grad():
        torch.onnx.export(m, (torch.zeros(1, 112), torch.zeros(1, 44), torch.zeros(m.layers, 1, m.H)), fp32_path,
                          opset_version=17, dynamo=False,
                          input_names=["state", "prev_prices", "h"], output_names=["pred", "h_out", "prices"])
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32_path, onnx_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    # Stats also stored on their own for reference; the graph carries them as constants.
    np.savez(os.path.join(out_dir, "stats.npz"), mean=m.mean[0].numpy(), std=m.std[0].numpy(),
             price_idx=m.price_idx.numpy(), other_idx=m.other_idx.numpy(), flag_idx=m.flag_idx.numpy(),
             fill=np.float32(FILL))
    shutil.copy(os.path.join(HERE, "solution_fast.py"), os.path.join(out_dir, "solution.py"))

    files = ["solution.py", "model.onnx", "stats.npz"]
    for f in files:
        print(f"  {f}: {os.path.getsize(os.path.join(out_dir, f)):,} bytes")
    if zip_path:
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(os.path.join(out_dir, f), arcname=f)
        print(f"zip {zip_path}: {os.path.getsize(zip_path):,} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"checkpoint step {ck['step']}, valid wp {ck['valid']['weighted_pearson']:.4f}, hidden {ck['args'].get('hidden', 256)}")
    export(ck, args.out, args.zip, args.int8)


if __name__ == "__main__":
    main()
