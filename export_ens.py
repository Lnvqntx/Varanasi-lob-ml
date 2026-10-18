"""Export two train_v1.py checkpoints as ONE single-step ONNX graph that averages their outputs.

Shared front end (as export_v2.build): prices, price diffs, float16-rounded other features, fill flags.
Then per model m in (A, B): its own explicit standardization (Sub mean_m, Div std_m; never folded, see
export_v2 --int8), two explicit GRU cells (export_v2.cell_nodes/cell_inits, fused [x, h] MatMul) and
its linear head. pred = Clip((pred_A + pred_B) / 2, -2, 2): the raw outputs are averaged, then clipped
as the metric does. With --int8 (default) every MatMul is dynamically quantized to int8 exactly as
export_v2.py --int8 --cell matmul.

  inputs   state (1,1,112), prev_prices (1,1,44), hA0, hA1, hB0, hB1 (1,1,H)
  outputs  pred (1,1,2), hA0_out, hA1_out, hB0_out, hB1_out (1,1,H), prices (1,1,44)

Usage (needs the `onnx` package for export only):
  PYTHONPATH=<dir with onnx> python export_ens.py --ckpt-a A.pt --ckpt-b B.pt --out DIR --zip ZIP
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
from export_v2 import cell_inits, cell_nodes

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["A", "B"]


def build(cks):
    c0 = cks[0]
    for ck in cks[1:]:
        assert ck["price_cols"] == c0["price_cols"] and ck["other_cols"] == c0["other_cols"] \
            and ck["flag_cols"] == c0["flag_cols"], "checkpoints use different input columns"
    price_idx = np.array([FEATURE_COLUMNS.index(c) for c in c0["price_cols"]], dtype=np.int64)
    other_idx = np.array([FEATURE_COLUMNS.index(c) for c in c0["other_cols"]], dtype=np.int64)
    flag_idx = np.array([c0["other_cols"].index(c) for c in c0["flag_cols"]], dtype=np.int64)
    n_p = len(price_idx)
    Hs = {ck["model"]["gru.weight_hh_l0"].shape[1] for ck in cks}
    assert len(Hs) == 1, "both models must have the same hidden size (shared split constants)"
    H = Hs.pop()

    inits = [
        numpy_helper.from_array(price_idx, "price_idx"),
        numpy_helper.from_array(other_idx, "other_idx"),
        numpy_helper.from_array(np.array(FILL, dtype=np.float32), "fill"),
        numpy_helper.from_array(flag_idx, "flag_idx"),
        numpy_helper.from_array(np.array([2 * H, H, H], dtype=np.int64), "split_g"),
        numpy_helper.from_array(np.array([H, H], dtype=np.int64), "split_rz"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
        numpy_helper.from_array(np.array(-2.0, dtype=np.float32), "clip_lo"),
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), "clip_hi"),
    ]
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
        helper.make_node("Concat", ["prices", "diff", "other"], ["cont"], axis=2),
    ]
    stats = dict(price_idx=price_idx, other_idx=other_idx, flag_idx=flag_idx, fill=np.float32(FILL))
    for m, ck in zip(MODELS, cks):
        sd = {k: v.cpu().double().numpy() for k, v in ck["model"].items()}
        assert not any(k.startswith("gru.weight_ih_l2") for k in sd), "export supports 2 layers"
        mean = ck["stats"]["mean"].cpu().double().numpy()
        std = ck["stats"]["std"].cpu().double().numpy()
        assert len(mean) == 2 * n_p + len(other_idx) and sd["gru.weight_ih_l0"].shape[1] == len(mean) + len(flag_idx)
        inits += [numpy_helper.from_array(mean.astype(np.float32), f"mean{m}"),
                  numpy_helper.from_array(std.astype(np.float32), f"std{m}"),
                  numpy_helper.from_array(sd["head.weight"].T.astype(np.float32), f"w_out{m}"),
                  numpy_helper.from_array(sd["head.bias"].astype(np.float32), f"b_out{m}")]
        for l in range(2):
            inits += cell_inits(f"{m}{l}", sd[f"gru.weight_ih_l{l}"], sd[f"gru.bias_ih_l{l}"],
                                sd[f"gru.weight_hh_l{l}"], sd[f"gru.bias_hh_l{l}"], H)
        nodes += [
            helper.make_node("Sub", ["cont", f"mean{m}"], [f"cont_c{m}"]),
            helper.make_node("Div", [f"cont_c{m}", f"std{m}"], [f"cont_z{m}"]),
            helper.make_node("Concat", [f"cont_z{m}", "flags"], [f"x{m}"], axis=2),
            *cell_nodes(f"{m}0", f"x{m}", f"h{m}0", f"h{m}0_out", H),
            *cell_nodes(f"{m}1", f"h{m}0_out", f"h{m}1", f"h{m}1_out", H),
            helper.make_node("MatMul", [f"h{m}1_out", f"w_out{m}"], [f"pred_mm{m}"]),
            helper.make_node("Add", [f"pred_mm{m}", f"b_out{m}"], [f"pred{m}"]),
        ]
        stats[f"mean{m}"], stats[f"std{m}"] = mean.astype(np.float32), std.astype(np.float32)
    nodes += [
        helper.make_node("Add", ["predA", "predB"], ["pred_sum"]),
        helper.make_node("Mul", ["pred_sum", "half"], ["pred_avg"]),
        helper.make_node("Clip", ["pred_avg", "clip_lo", "clip_hi"], ["pred"]),
    ]
    f32 = TensorProto.FLOAT
    h_names = [f"h{m}{l}" for m in MODELS for l in range(2)]
    graph = helper.make_graph(
        nodes, "gru_ens_step",
        [helper.make_tensor_value_info("state", f32, [1, 1, 112]),
         helper.make_tensor_value_info("prev_prices", f32, [1, 1, n_p])]
        + [helper.make_tensor_value_info(h, f32, [1, 1, H]) for h in h_names],
        [helper.make_tensor_value_info("pred", f32, [1, 1, 2])]
        + [helper.make_tensor_value_info(f"{h}_out", f32, [1, 1, H]) for h in h_names]
        + [helper.make_tensor_value_info("prices", f32, [1, 1, n_p])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="export_ens")
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model, stats


def export(cks, out_dir, zip_path=None, int8=True):
    model, stats = build(cks)
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists; not overwriting")
    if zip_path and os.path.exists(zip_path):
        raise SystemExit(f"{zip_path} exists; not overwriting")
    os.makedirs(out_dir)
    onnx_path = os.path.join(out_dir, "model.onnx")
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        fp32_path = os.path.join(out_dir, "model_fp32.onnx")  # kept for reference, not zipped
        onnx.save(model, fp32_path)
        quantize_dynamic(fp32_path, onnx_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    else:
        onnx.save(model, onnx_path)
    np.savez(os.path.join(out_dir, "stats.npz"), **stats)  # the graph carries its own copy of the stats
    shutil.copy(os.path.join(HERE, "solution_ens.py"), os.path.join(out_dir, "solution.py"))
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
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--fp32", action="store_true", help="skip int8 quantization")
    args = ap.parse_args()
    cks = []
    for p in [args.ckpt_a, args.ckpt_b]:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{p}: step {ck['step']}, valid wp {ck['valid']['weighted_pearson']:.4f}, hidden {ck['args']['hidden']}")
        cks.append(ck)
    export(cks, args.out, args.zip, int8=not args.fp32)


if __name__ == "__main__":
    main()
