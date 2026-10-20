"""Ensemble check on the 469 held-out valid sequences for v11 + v12 (train + valid, mask-weighted), with v7 + v9.

ensemble_check4.py's method (GPU 0, 6 GB cap, batches of 25, 1000-step chunks; raw outputs averaged, then
clipped to [-2, 2] by the metric in spread_v2.seq_stats; per-sequence float64 sufficient statistics on
is_scored & step >= 99 and on all steps >= 99), but the pass covers only the held-out sequences
(valid index % 4 == 0): v11 and v12 trained on the other 1,404. v11/v12 come as best and final
checkpoints; their "best" was picked on these same 469 sequences, so the final ones are the fair comparison.
Single-model sanity: v11/v12 against the held-out score stored in each checkpoint, v7/v9 against
ensemble_check4's held-out result (4 decimals); tolerance 0.0005, otherwise no results are written.

Usage:
  nohup python ensemble_check5.py > ~/work/ensemble_check5.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time

import numpy as np

from ensemble_check2 import BATCH, load_model, predict
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v11b": f"{CKPT_DIR}/train_v11_maskw_a05_adv_h128_e5_seed0_best.pt",
         "v11f": f"{CKPT_DIR}/train_v11_maskw_a05_adv_h128_e5_seed0_final.pt",
         "v12b": f"{CKPT_DIR}/train_v12_maskw_a05_adv_h128_e5_seed1_best.pt",
         "v12f": f"{CKPT_DIR}/train_v12_maskw_a05_adv_h128_e5_seed1_final.pt",
         "v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt"}
SUBSETS = [("v11b",), ("v12b",), ("v11b", "v12b"), ("v11f",), ("v12f",), ("v11f", "v12f"),
           ("v7",), ("v9",), ("v7", "v9")]
HELD_REFS = {"v7": 0.6892, "v9": 0.6889}  # ensemble_check4_results.txt, held out 469
STATS_OUT = "/home/jovyan/work/ensemble_check5_stats.npz"
RESULTS_OUT = "/home/jovyan/work/ensemble_check5_results.txt"
SETS = ["masked", "all"]
TOL = 0.0005


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
    cks = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in CKPTS.items()}
    refs = dict(HELD_REFS)
    for k, ck in cks.items():
        if k not in refs:
            assert ck["args"]["add_valid"] and len(ck["valid_eval_idx"]) == 469
            refs[k] = ck["valid"]["weighted_pearson"]
    steps = {k: ck["step"] for k, ck in cks.items()}
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    held = np.arange(n)[np.arange(n) % 4 == 0]
    assert all(np.array_equal(cks[k]["valid_eval_idx"], held) for k in cks if k not in HELD_REFS)
    need = np.arange(T) >= WARMUP
    S = np.zeros((len(SUBSETS), len(held), len(SETS), 2, 6), dtype=np.float64)
    batches = [held[i:i + BATCH] for i in range(0, len(held), BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            preds = {k: predict(m, st, b, device) for k, (m, st) in models.items()}
            for si, sub in enumerate(SUBSETS):
                avg = np.mean([preds[k] for k in sub], axis=0, dtype=np.float32)
                for k in range(len(batches[bi])):
                    j = bi * BATCH + k
                    S[si, j, 0] = seq_stats(b["targets"][k], avg[k], b["is_scored"][k] & need)
                    S[si, j, 1] = seq_stats(b["targets"][k], avg[k], need)
            if (bi + 1) % 5 == 0:
                print(f"  {min((bi + 1) * BATCH, len(held))}/{len(held)} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    sub_names = ["+".join(s) for s in SUBSETS]
    np.savez(STATS_OUT, stats=S, subsets=sub_names, sets=SETS, stat_names=STAT_NAMES, held=held,
             ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)

    R = wp_from_stats(S.sum(1))  # (n_subsets, n_sets, 3)
    bad = []
    for k in CKPTS:
        r = R[sub_names.index(k), 0, 2]
        print(f"sanity {k}: masked {r:.6f} vs reference {refs[k]:.6f} | diff {r - refs[k]:+.6f}", flush=True)
        if abs(r - refs[k]) > TOL:
            bad.append(k)
    if bad:
        sys.exit(f"STOP: single-model masked score off by more than {TOL} for {bad}; no results written")

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"Ensemble check on the {len(held)} held-out valid sequences (index % 4 == 0, steps >= 99), fp32 PyTorch, GPU 0")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        out(f"  {k}: {v} (step {steps[k]})")
    out("  v11/v12 trained on train + the other 1,404 valid sequences; their 'best' (b) was picked on these 469")
    out("  sequences, so it is optimistic; final (f) is the end of epoch 5. v7/v9 picked on all 1,873 valid.")
    out(f"\nSingle-model sanity (tolerance {TOL}): v11/v12 vs the held-out score in the checkpoint, "
        f"v7/v9 vs ensemble_check4")
    for k in CKPTS:
        r = R[sub_names.index(k), 0, 2]
        out(f"  {k}: {r:.6f} vs {refs[k]:.6f} ({r - refs[k]:+.6f})")
    out("\ngain = pair minus its better member")
    out(f"  {'subset':<10} {'masked t0':>10} {'masked t1':>10} {'masked':>8} {'all rows':>9} {'gain':>8} {'gain all':>9}")
    for si, name in enumerate(sub_names):
        g = ga = ""
        if "+" in name:
            mem = [sub_names.index(k) for k in name.split("+")]
            g = f"{R[si, 0, 2] - max(R[m, 0, 2] for m in mem):+8.4f}"
            ga = f"{R[si, 1, 2] - max(R[m, 1, 2] for m in mem):+9.4f}"
        out(f"  {name:<10} {R[si, 0, 0]:>10.4f} {R[si, 0, 1]:>10.4f} {R[si, 0, 2]:>8.4f} {R[si, 1, 2]:>9.4f} "
            f"{g:>8} {ga:>9}")
    ref = sub_names.index("v7+v9")
    for name in ["v11b+v12b", "v11f+v12f"]:
        si = sub_names.index(name)
        out(f"  {name} vs v7+v9: masked {R[si, 0, 2] - R[ref, 0, 2]:+.4f}, all rows {R[si, 1, 2] - R[ref, 1, 2]:+.4f}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
