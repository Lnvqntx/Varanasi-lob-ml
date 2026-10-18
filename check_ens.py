"""Check a two-model ensemble submission against the fp32 PyTorch average of the same checkpoints.

  SUB=<extracted submission dir> python check_ens.py --ckpt-a A.pt --ckpt-b B.pt [--n 100]

The submission's solution.py runs on the first N valid sequences through check_v1's replica of the
ScorerStepByStep loop (CPU, GPUs hidden for it); the reference is the average of the two models' raw
outputs on the .npy cache (GPU 0, 1000-step chunks, as ensemble_check2.py). Both are scored with
utils.GlobalAccumulator on is_scored & need_prediction rows (it clips to [-2, 2]).
"""
import argparse
import os

import numpy as np

from check_v1 import load_valid, run_solution, score  # hides GPUs; imports $SUB/solution.py


def reference(seqs, ckpts):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 only, for the PyTorch reference
    import torch
    import train_v1
    from ensemble_check2 import load_model, predict
    from spread_v2 import GPU_MEM_CAP_GB
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    n = len(seqs)
    seq_ix = np.load("/home/jovyan/work/cache/valid_seq_ix.npy")
    assert [int(x) for x in seq_ix[:n]] == [s["seq"] for s in seqs], "cache order != parquet order"
    valid = train_v1.load_split("valid")
    out = []
    with torch.no_grad():
        for i0 in range(0, n, 25):
            b = train_v1.read_batch(valid, np.arange(i0, min(i0 + 25, n)))
            preds = [predict(*load_model(p, device), b, device) for p in ckpts]
            out.append(np.mean(preds, axis=0, dtype=np.float32))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    args = ap.parse_args()
    seqs = load_valid(args.n)
    print(f"submission {os.environ['SUB']}; {args.n} valid sequences", flush=True)
    preds, tp, _ = run_solution(seqs)
    print(f"ONNX run done: predict() {tp / (args.n * 20000) * 1e6:.1f} us/row (unpinned, indicative only)", flush=True)
    ref = reference(seqs, [args.ckpt_a, args.ckpt_b])
    need = np.stack([s["need"] for s in seqs])
    d = np.abs(preds[need] - np.clip(ref[need], -2, 2))
    print(f"int8 ensemble vs clipped fp32 average on {need.sum():,} prediction rows: max abs diff {d.max():.3e}, "
          f"mean {d.mean():.3e}")
    r_int8, r_fp32 = score(seqs, preds), score(seqs, ref)
    for name, r in [("int8 ensemble", r_int8), ("fp32 avg", r_fp32)]:
        print(f"{name:>14}: wp={r['weighted_pearson']:.6f} (t0 {r['t0']:.6f}, t1 {r['t1']:.6f}), "
              f"scored rows {r['selected_rows']:,}")
    print(f"int8 loss (fp32 - int8): {r_fp32['weighted_pearson'] - r_int8['weighted_pearson']:+.6f} (limit 0.0010)")


if __name__ == "__main__":
    main()
