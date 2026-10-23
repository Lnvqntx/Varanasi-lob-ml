"""Ensemble check on valid for the two mask-weighted models v7 + v9 (alpha 0.5, seeds 0 and 1).

Same method as ensemble_check2.py: one pass over all 1,873 valid sequences (GPU 0, 6 GB cap, batches of
25, 1000-step chunks); raw outputs averaged, then clipped to [-2, 2] by the metric (spread_v2.seq_stats);
per-sequence float64 sufficient statistics on two row sets:
  masked   is_scored & step >= 99   (official metric)
  all      step >= 99
and three sequence splits: all 1,873; held out = index % 4 == 0 (469 sequences the mask GRU never saw,
mask_gru.split_idx); fitting = the other 1,404. The submitted pair v4+v5 runs in the same pass as a
reference. Single-model masked scores must match each checkpoint's logged valid score within 0.0005,
otherwise no results are written.

Usage:
  nohup python ensemble_check3.py > ~/work/ensemble_check3.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time

import numpy as np

from ensemble_check2 import BATCH, load_model, predict
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt",
         "v4": f"{CKPT_DIR}/train_v4_h128_e5_fullval_best.pt",
         "v5": f"{CKPT_DIR}/train_v5_h128_e5_seed1_best.pt"}
SUBSETS = [("v7",), ("v9",), ("v7", "v9"), ("v4",), ("v5",), ("v4", "v5")]
STATS_OUT = "/home/jovyan/work/ensemble_check3_stats.npz"
RESULTS_OUT = "/home/jovyan/work/ensemble_check3_results.txt"
SETS = ["masked", "all"]
TOL = 0.0005


def split_idx(n):  # as mask_gru.split_idx (not imported: mask_gru sets up its own environment)
    i = np.arange(n)
    return i[i % 4 != 0], i[i % 4 == 0]


def main():
    for p in [STATS_OUT, RESULTS_OUT]:
        if os.path.exists(p):
            sys.exit(f"{p} exists; not overwriting")
    import torch
    import train_v1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)}, memory cap {GPU_MEM_CAP_GB} GB", flush=True)

    models = {k: load_model(v, device) for k, v in CKPTS.items()}
    refs = {k: torch.load(v, map_location="cpu", weights_only=False)["valid"]["weighted_pearson"]
            for k, v in CKPTS.items()}
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    need = np.arange(T) >= WARMUP
    S = np.zeros((len(SUBSETS), n, len(SETS), 2, 6), dtype=np.float64)
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            preds = {k: predict(m, st, b, device) for k, (m, st) in models.items()}
            for si, sub in enumerate(SUBSETS):
                avg = np.mean([preds[k] for k in sub], axis=0, dtype=np.float32)
                for k, i in enumerate(batches[bi]):
                    S[si, i, 0] = seq_stats(b["targets"][k], avg[k], b["is_scored"][k] & need)
                    S[si, i, 1] = seq_stats(b["targets"][k], avg[k], need)
            if (bi + 1) % 15 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True)
    sub_names = ["+".join(s) for s in SUBSETS]
    np.savez(STATS_OUT, stats=S, subsets=sub_names, sets=SETS, stat_names=STAT_NAMES,
             ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)

    fit, held = split_idx(n)
    splits = {"all 1873": np.arange(n), f"held out {len(held)}": held, f"fitting {len(fit)}": fit}
    R = {name: wp_from_stats(S[:, idx].sum(1)) for name, idx in splits.items()}  # (n_subsets, n_sets, 3)
    bad = []
    for k in CKPTS:
        r = R["all 1873"][sub_names.index(k), 0, 2]
        print(f"sanity {k}: masked {r:.6f} vs logged {refs[k]:.6f} | diff {r - refs[k]:+.6f}", flush=True)
        if abs(r - refs[k]) > TOL:
            bad.append(k)
    if bad:
        sys.exit(f"STOP: single-model masked score off by more than {TOL} for {bad}; no results written")
    report(sub_names, R, refs)


def report(sub_names, R, refs):
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("Ensemble check on valid (steps >= 99), fp32 PyTorch, GPU 0: v7 + v9 (mask-weighted, alpha 0.5, seeds 0/1)")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    out("v4 + v5 (plain recipe, seeds 0/1; the submitted ensemble) is shown as a reference.")
    for k, v in CKPTS.items():
        out(f"  {k}: {v}")
    out("  All four picked their checkpoint on all 1,873 valid sequences (masked rows), so full-valid masked scores")
    out("  are slightly optimistic. Held out = valid index % 4 == 0, the sequences the mask GRU never saw.")
    out(f"\nSingle-model sanity vs logged valid score (tolerance {TOL}):")
    for k in CKPTS:
        r = R["all 1873"][sub_names.index(k), 0, 2]
        out(f"  {k}: {r:.6f} vs {refs[k]:.6f} ({r - refs[k]:+.6f})")
    for split, r in R.items():
        out(f"\n{split} sequences; gain = pair minus its better member")
        out(f"  {'subset':<7} {'masked t0':>10} {'masked t1':>10} {'masked':>8} {'all rows':>9} {'gain':>8} {'gain all':>9}")
        for si, name in enumerate(sub_names):
            g = ga = ""
            if "+" in name:
                mem = [sub_names.index(k) for k in name.split("+")]
                g = f"{r[si, 0, 2] - max(r[m, 0, 2] for m in mem):+8.4f}"
                ga = f"{r[si, 1, 2] - max(r[m, 1, 2] for m in mem):+9.4f}"
            out(f"  {name:<7} {r[si, 0, 0]:>10.4f} {r[si, 0, 1]:>10.4f} {r[si, 0, 2]:>8.4f} {r[si, 1, 2]:>9.4f} "
                f"{g:>8} {ga:>9}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
