"""export_v2.py plus 2-layer LSTM checkpoints from train_v4m.py --arch lstm (GRU export unchanged).

LSTM (--cell matmul only, so --int8 can quantize it): per layer one fused MatMul over [x, h], with the
torch gate rows (i, f, g, o) reordered to columns (i, f, o | g) so one Sigmoid and one Tanh cover them:
  [x, h] @ [[W_ii W_if W_io | W_ig], [W_hi W_hf W_ho | W_hg]] + (b_ih + b_hh)
  i, f, o = sigmoid(...), g = tanh(...);  c' = f * c + i * g;  h' = o * tanh(c')
Extra graph inputs c0, c1 and outputs c0_out, c1_out (1,1,H); the zip gets solution_v4.py, which carries
every recurrent input/output pair the graph declares. Same standardization handling as the GRU.

export_v2.py's description:

Export a train_v1.py checkpoint to a minimal single-step ONNX graph built directly with onnx.helper.

Graph (one session.run per row, 14 nodes; all tensors 3-D so no reshapes are needed):
  prices = Gather(state, price_idx)              other = Cast(Cast(Gather(state, other_idx), f16), f32)
  diff   = prices - prev_prices                  flags = Cast(Abs(Gather(other, flag_idx)) == 5.199, f32)
  x      = Concat[prices, diff, other, flags]                                              (1,1,168)
  h0_out = GRU(x, h0); h1_out = GRU(h0_out, h1); pred = h1_out @ W + b

--cell matmul replaces each GRU node with explicit cell math using one fused MatMul over [x, h]:
  [x, h] @ [[W_ir W_iz | W_in |  0  ],      -> r,z pre-activations | i_n | h_n
            [W_hr W_hz |  0   | W_hn]] + b,    n = tanh(i_n + r * h_n),  h' = n + z * (h - n)
which ORT's dynamic int8 quantization can handle (--int8); it cannot quantize the GRU op.
With --int8 the standardization stays as explicit Sub/Div nodes instead of being folded into layer 0:
dynamic quantization uses one scale per input row, and unstandardized inputs mix prices (~1) with
price diffs (~1e-3), which would lose the diffs.

Equivalences with train_v1.build_inputs + nn.GRU:
  - standardization (x - mean) / std is folded into layer-0 weights: W' = W / std, b' = b - W' @ mean
  - gate order is permuted from torch (r, z, n) to ONNX (z, r, h); linear_before_reset=1 matches torch

  inputs   state (1,1,112), prev_prices (1,1,44) (= this row's prices at step 0 -> diff 0), h0, h1 (1,1,H)
  outputs  pred (1,1,2), h0_out, h1_out (1,1,H), prices (1,1,44) (pass back as prev_prices)

Usage (needs the `onnx` package for export only):
  PYTHONPATH=<dir with onnx> python export_v2.py --ckpt CKPT --out DIR [--zip ZIP] [--cell matmul [--int8]]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import shutil
import zipfile

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from export_v1 import FEATURE_COLUMNS, FILL

HERE = os.path.dirname(os.path.abspath(__file__))


def to_onnx_gates(w, H):
    """torch rows (r, z, n) -> ONNX rows (z, r, h)."""
    return np.concatenate([w[H:2 * H], w[:H], w[2 * H:]], axis=0)


def gru_nodes(l, x, h, out, H):
    return [helper.make_node("GRU", [x, f"W{l}", f"R{l}", f"B{l}", "", h], ["", out], hidden_size=H,
                             linear_before_reset=1)]


def gru_inits(l, wih, bih, whh, bhh, H):
    return [
        numpy_helper.from_array(to_onnx_gates(wih, H)[None].astype(np.float32), f"W{l}"),
        numpy_helper.from_array(to_onnx_gates(whh, H)[None].astype(np.float32), f"R{l}"),
        numpy_helper.from_array(np.concatenate([to_onnx_gates(bih, H), to_onnx_gates(bhh, H)])[None]
                                .astype(np.float32), f"B{l}"),
    ]


def cell_nodes(l, x, h, out, H):
    p = f"l{l}_"
    return [
        helper.make_node("Concat", [x, h], [p + "xh"], axis=2),
        helper.make_node("MatMul", [p + "xh", f"Wf{l}"], [p + "mm"]),
        helper.make_node("Add", [p + "mm", f"bf{l}"], [p + "g"]),
        helper.make_node("Split", [p + "g", "split_g"], [p + "rz_pre", p + "i_n", p + "h_n"], axis=2),
        helper.make_node("Sigmoid", [p + "rz_pre"], [p + "rz"]),
        helper.make_node("Split", [p + "rz", "split_rz"], [p + "r", p + "z"], axis=2),
        helper.make_node("Mul", [p + "r", p + "h_n"], [p + "rh"]),
        helper.make_node("Add", [p + "i_n", p + "rh"], [p + "n_pre"]),
        helper.make_node("Tanh", [p + "n_pre"], [p + "n"]),
        helper.make_node("Sub", [h, p + "n"], [p + "hn"]),
        helper.make_node("Mul", [p + "z", p + "hn"], [p + "zhn"]),
        helper.make_node("Add", [p + "n", p + "zhn"], [out]),
    ]


def cell_inits(l, wih, bih, whh, bhh, H):
    n_in = wih.shape[1]
    w = np.zeros((n_in + H, 4 * H))
    w[:n_in, :2 * H] = wih[:2 * H].T
    w[n_in:, :2 * H] = whh[:2 * H].T
    w[:n_in, 2 * H:3 * H] = wih[2 * H:].T
    w[n_in:, 3 * H:] = whh[2 * H:].T
    b = np.concatenate([bih[:2 * H] + bhh[:2 * H], bih[2 * H:], bhh[2 * H:]])
    return [numpy_helper.from_array(w.astype(np.float32), f"Wf{l}"),
            numpy_helper.from_array(b.astype(np.float32), f"bf{l}")]


def lstm_cell_nodes(l, x, h, c, h_out, c_out):
    p = f"l{l}_"
    return [
        helper.make_node("Concat", [x, h], [p + "xh"], axis=2),
        helper.make_node("MatMul", [p + "xh", f"Wf{l}"], [p + "mm"]),
        helper.make_node("Add", [p + "mm", f"bf{l}"], [p + "g"]),
        helper.make_node("Split", [p + "g", "split_g"], [p + "ifo_pre", p + "g_pre"], axis=2),
        helper.make_node("Sigmoid", [p + "ifo_pre"], [p + "ifo"]),
        helper.make_node("Tanh", [p + "g_pre"], [p + "gg"]),
        helper.make_node("Split", [p + "ifo", "split_ifo"], [p + "i", p + "f", p + "o"], axis=2),
        helper.make_node("Mul", [p + "f", c], [p + "fc"]),
        helper.make_node("Mul", [p + "i", p + "gg"], [p + "ig"]),
        helper.make_node("Add", [p + "fc", p + "ig"], [c_out]),
        helper.make_node("Tanh", [c_out], [p + "tc"]),
        helper.make_node("Mul", [p + "o", p + "tc"], [h_out]),
    ]


def lstm_cell_inits(l, wih, bih, whh, bhh, H):
    order = np.concatenate([np.arange(0, 2 * H), np.arange(3 * H, 4 * H), np.arange(2 * H, 3 * H)])  # i f o g
    w = np.concatenate([wih[order].T, whh[order].T], axis=0)
    b = (bih + bhh)[order]
    return [numpy_helper.from_array(w.astype(np.float32), f"Wf{l}"),
            numpy_helper.from_array(b.astype(np.float32), f"bf{l}")]


def rnn_prefix(sd):
    return "lstm" if "lstm.weight_hh_l0" in sd else "gru"


def build(ck, cell="gru", fold=True):
    sd = {k: v.cpu().double().numpy() for k, v in ck["model"].items()}
    rp = rnn_prefix(sd)
    lstm = rp == "lstm"
    assert not lstm or cell == "matmul", "LSTM export needs --cell matmul"
    if lstm:  # rename to the gru.* keys used below; the layer functions are chosen separately
        sd = {k.replace("lstm.", "gru.", 1): v for k, v in sd.items()}
    H = sd["gru.weight_hh_l0"].shape[1]
    assert not any(k.startswith("gru.weight_ih_l2") for k in sd), "export supports 2 layers"
    price_idx = np.array([FEATURE_COLUMNS.index(c) for c in ck["price_cols"]], dtype=np.int64)
    other_idx = np.array([FEATURE_COLUMNS.index(c) for c in ck["other_cols"]], dtype=np.int64)
    flag_idx = [ck["other_cols"].index(c) for c in ck["flag_cols"]]
    mean = ck["stats"]["mean"].cpu().double().numpy()
    std = ck["stats"]["std"].cpu().double().numpy()
    n_p, n_o, n_cont = len(price_idx), len(other_idx), len(mean)
    assert sd["gru.weight_ih_l0"].shape[1] == n_cont + len(flag_idx)
    assert n_cont == 2 * n_p + n_o

    # Layer 0: same 168 input columns as training; standardization folded into the weights if fold.
    w0 = sd["gru.weight_ih_l0"].copy()
    b0 = sd["gru.bias_ih_l0"].copy()
    if fold:
        w0[:, :n_cont] /= std
        b0 -= w0[:, :n_cont] @ mean

    inits = [
        numpy_helper.from_array(price_idx, "price_idx"),
        numpy_helper.from_array(other_idx, "other_idx"),
        numpy_helper.from_array(np.array(FILL, dtype=np.float32), "fill"),
        numpy_helper.from_array(np.array(flag_idx, dtype=np.int64), "flag_idx"),
        numpy_helper.from_array(sd["head.weight"].T.astype(np.float32), "w_out"),
        numpy_helper.from_array(sd["head.bias"].astype(np.float32), "b_out"),
    ]
    layer_inits, layer_nodes = (gru_inits, gru_nodes) if cell == "gru" else (cell_inits, cell_nodes)
    if lstm:
        layer_inits = lstm_cell_inits
    if not fold:
        inits += [numpy_helper.from_array(mean.astype(np.float32), "mean"),
                  numpy_helper.from_array(std.astype(np.float32), "std")]
    if lstm:
        inits += [numpy_helper.from_array(np.array([3 * H, H], dtype=np.int64), "split_g"),
                  numpy_helper.from_array(np.array([H, H, H], dtype=np.int64), "split_ifo")]
    elif cell == "matmul":
        inits += [numpy_helper.from_array(np.array([2 * H, H, H], dtype=np.int64), "split_g"),
                  numpy_helper.from_array(np.array([H, H], dtype=np.int64), "split_rz")]
    for l, (wih, bih) in enumerate([(w0, b0), (sd["gru.weight_ih_l1"], sd["gru.bias_ih_l1"])]):
        inits += layer_inits(l, wih, bih, sd[f"gru.weight_hh_l{l}"], sd[f"gru.bias_hh_l{l}"], H)

    nodes = [
        helper.make_node("Gather", ["state", "price_idx"], ["prices"], axis=2),
        helper.make_node("Gather", ["state", "other_idx"], ["other_raw"], axis=2),
        helper.make_node("Cast", ["other_raw"], ["other16"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["other16"], ["other"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["prices", "prev_prices"], ["diff"]),
        helper.make_node("Gather", ["other", "flag_idx"], ["flag_vals"], axis=2),
        helper.make_node("Abs", ["flag_vals"], ["other_abs"]),
        helper.make_node("Equal", ["other_abs", "fill"], ["is_fill"]),
        helper.make_node("Cast", ["is_fill"], ["flags"], to=TensorProto.FLOAT),
        *([helper.make_node("Concat", ["prices", "diff", "other", "flags"], ["x"], axis=2)] if fold else [
            helper.make_node("Concat", ["prices", "diff", "other"], ["cont"], axis=2),
            helper.make_node("Sub", ["cont", "mean"], ["cont_c"]),
            helper.make_node("Div", ["cont_c", "std"], ["cont_z"]),
            helper.make_node("Concat", ["cont_z", "flags"], ["x"], axis=2)]),
        *(lstm_cell_nodes(0, "x", "h0", "c0", "h0_out", "c0_out") +
          lstm_cell_nodes(1, "h0_out", "h1", "c1", "h1_out", "c1_out") if lstm else
          layer_nodes(0, "x", "h0", "h0_out", H) + layer_nodes(1, "h0_out", "h1", "h1_out", H)),
        helper.make_node("MatMul", ["h1_out", "w_out"], ["pred_mm"]),
        helper.make_node("Add", ["pred_mm", "b_out"], ["pred"]),
    ]
    f32 = TensorProto.FLOAT
    rec = ["h0", "h1"] + (["c0", "c1"] if lstm else [])
    graph = helper.make_graph(
        nodes, f"{rp}_step",
        [helper.make_tensor_value_info("state", f32, [1, 1, 112]),
         helper.make_tensor_value_info("prev_prices", f32, [1, 1, n_p])] +
        [helper.make_tensor_value_info(r, f32, [1, 1, H]) for r in rec],
        [helper.make_tensor_value_info("pred", f32, [1, 1, 2])] +
        [helper.make_tensor_value_info(r + "_out", f32, [1, 1, H]) for r in rec] +
        [helper.make_tensor_value_info("prices", f32, [1, 1, n_p])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="export_v2")
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model, dict(mean=mean.astype(np.float32), std=std.astype(np.float32), price_idx=price_idx,
                       other_idx=other_idx, flag_idx=np.array(flag_idx, dtype=np.int64), fill=np.float32(FILL))


def export(ck, out_dir, zip_path=None, cell="gru", int8=False):
    assert not int8 or cell == "matmul", "--int8 needs --cell matmul"
    model, stats = build(ck, cell, fold=not int8)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        fp32_path = os.path.join(out_dir, "model_fp32.onnx")  # kept for reference, not zipped
        onnx.save(model, fp32_path)
        quantize_dynamic(fp32_path, onnx_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    else:
        onnx.save(model, onnx_path)
    # Stats also stored on their own for reference; the graph carries them folded into layer 0.
    np.savez(os.path.join(out_dir, "stats.npz"), **stats)
    sol = "solution_v4.py" if rnn_prefix(ck["model"]) == "lstm" else "solution_v2.py"
    shutil.copy(os.path.join(HERE, sol), os.path.join(out_dir, "solution.py"))
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
    ap.add_argument("--cell", choices=["gru", "matmul"], default="gru")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"checkpoint step {ck['step']}, valid wp {ck['valid']['weighted_pearson']:.4f}, "
          f"hidden {ck['args'].get('hidden', 256)}")
    export(ck, args.out, args.zip, args.cell, args.int8)


if __name__ == "__main__":
    main()
