"""Ensemble check for v13 (2-layer LSTM, train_v4m.py) with the GRUs v7 and v9.

ensemble_check5.py's method (GPU 0, 6 GB cap, batches of 25, 1000-step chunks; raw outputs averaged, then
clipped to [-2, 2] by the metric in spread_v2.seq_stats; per-sequence float64 sufficient statistics on
is_scored & step >= 99 and on all steps >= 99), with one pass over all 1,873 valid sequences; the 469
held-out sequences (valid index % 4 == 0) are a subset of the same statistics.
GRUs load with ensemble_check2.load_model (train_v1.GRUModel); the LSTM with train_v4m.LSTMModel.
Residual diversity: per target, the correlation of the residuals (clipped y - clipped p) of v7-v9 and
v7-v13 on the scored rows of the 469, unweighted and weighted by |clipped y|; averaged over t0 and t1.
Single-model sanity (tolerance 0.00001): full-valid masked and all-rows scores against those stored in
each checkpoint (the logged eval); held-out 469 v7, v9 and v7+v9 against ensemble_check5_stats.npz.

Usage:
  nohup python ensemble_check6.py > ~/work/ensemble_check6.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time

import numpy as np

from ensemble_check2 import BATCH, load_model, predict
from spread_v2 import GPU_MEM_CAP_GB, METRIC_CLIP, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt",
         "v13": f"{CKPT_DIR}/train_v13_lstm_maskw_a05_h128_e5_seed0_best.pt"}
SUBSETS = [("v7",), ("v9",), ("v13",), ("v7", "v13"), ("v9", "v13"), ("v7", "v9"), ("v7", "v9", "v13")]
PAIRS = [("v7", "v9"), ("v7", "v13"), ("v9", "v13")]
CHECK5 = "/home/jovyan/work/ensemble_check5_stats.npz"
STATS_OUT = "/home/jovyan/work/ensemble_check6_stats.npz"
RESULTS_OUT = "/home/jovyan/work/ensemble_check6_results.txt"
SETS = ["masked", "all"]
TOL = 0.00001


def load_lstm(path, device):
    import torch
    import train_v4m
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["args"]["arch"] == "lstm"
    model = train_v4m.LSTMModel(hidden=ck["args"]["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    print(f"{path}: step {ck['step']}, LSTM hidden {ck['args']['hidden']}, seed {ck['args']['seed']}, "
          f"n_val {ck['args']['n_val']}", flush=True)
    return model, stats


def resid_sums(y, pa, pb):
    """(2 weightings, 2 targets, 6) float64 sums [n, sa, sb, saa, sbb, sab] of residuals a, b."""
    y = np.clip(y, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    a = y - np.clip(pa, -METRIC_CLIP, METRIC_CLIP)
    b = y - np.clip(pb, -METRIC_CLIP, METRIC_CLIP)
    out = []
    for w in [np.ones_like(y), np.abs(y)]:
        out.append(np.stack([w.sum(0), (w * a).sum(0), (w * b).sum(0), (w * a * a).sum(0),
                             (w * b * b).sum(0), (w * a * b).sum(0)], axis=-1))
    return np.stack(out)


def resid_corr(s):
    n, sa, sb, saa, sbb, sab = np.moveaxis(s, -1, 0)
    ma, mb = sa / n, sb / n
    return (sab / n - ma * mb) / np.sqrt((saa / n - ma * ma) * (sbb / n - mb * mb))


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

    models = {k: (load_lstm if k == "v13" else load_model)(v, device) for k, v in CKPTS.items()}
    cks = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in CKPTS.items()}
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    is_held = np.arange(n) % 4 == 0
    need = np.arange(T) >= WARMUP
    S = np.zeros((len(SUBSETS), n, len(SETS), 2, 6), dtype=np.float64)
    RS = np.zeros((len(PAIRS), 2, 2, 6), dtype=np.float64)  # held-out scored rows only
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            preds = {k: predict(m, st, b, device) for k, (m, st) in models.items()}
            for si, sub in enumerate(SUBSETS):
                avg = np.mean([preds[k] for k in sub], axis=0, dtype=np.float32)
                for k, j in enumerate(batches[bi]):
                    S[si, j, 0] = seq_stats(b["targets"][k], avg[k], b["is_scored"][k] & need)
                    S[si, j, 1] = seq_stats(b["targets"][k], avg[k], need)
            for k, j in enumerate(batches[bi]):
                if is_held[j]:
                    mask = b["is_scored"][k] & need
                    for pi, (pa, pb) in enumerate(PAIRS):
                        RS[pi] += resid_sums(b["targets"][k][mask], preds[pa][k][mask], preds[pb][k][mask])
            if (bi + 1) % 10 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    sub_names = ["+".join(s) for s in SUBSETS]
    np.savez(STATS_OUT, stats=S, resid=RS, subsets=sub_names, pairs=["-".join(p) for p in PAIRS], sets=SETS,
             stat_names=STAT_NAMES, held=np.flatnonzero(is_held), ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)

    R_full = wp_from_stats(S.sum(1))  # (n_subsets, n_sets, 3)
    R_held = wp_from_stats(S[:, is_held].sum(1))
    c5 = np.load(CHECK5)
    c5_names = list(c5["subsets"])
    R_c5 = wp_from_stats(c5["stats"].sum(1))
    checks = []
    for k, ck in cks.items():
        si = sub_names.index(k)
        checks.append((f"{k} full masked vs checkpoint", R_full[si, 0, 2], ck["valid"]["weighted_pearson"]))
        checks.append((f"{k} full all rows vs checkpoint", R_full[si, 1, 2], ck["valid_all"]["weighted_pearson"]))
    for k in ["v7", "v9", "v7+v9"]:
        for s, set_name in enumerate(SETS):
            checks.append((f"{k} held {set_name} vs ensemble_check5", R_held[sub_names.index(k), s, 2],
                           R_c5[c5_names.index(k), s, 2]))
    bad = []
    for name, r, ref in checks:
        print(f"sanity {name}: {r:.7f} vs {ref:.7f} | diff {r - ref:+.2e}", flush=True)
        if abs(r - ref) > TOL:
            bad.append(name)
    if bad:
        sys.exit(f"STOP: sanity off by more than {TOL} for {bad}; no results written")

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("Ensemble check: v13 (2-layer LSTM) with the GRUs v7 and v9, fp32 PyTorch, GPU 0, steps >= 99")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        out(f"  {k}: {v} (step {cks[k]['step']})")
    out("  all three picked their checkpoint on all 1,873 valid; held out = valid index % 4 == 0 (469 sequences)")
    out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
    for name, r, ref in checks:
        out(f"  {name}: {r:.7f} vs {ref:.7f} ({r - ref:+.1e})")
    out("\ngain = subset minus its best member; v7+v9+v13 is a reference only (too slow for the time limit)")
    out(f"  {'subset':<12} {'held masked':>11} {'gain':>8} {'held all':>9} {'gain':>8} {'full masked':>11} {'gain':>8}")
    cols = [(R_held, 0), (R_held, 1), (R_full, 0)]
    for si, name in enumerate(sub_names):
        cells = []
        for R, s in cols:
            g = ""
            if "+" in name:
                g = f"{R[si, s, 2] - max(R[sub_names.index(m), s, 2] for m in name.split('+')):+.4f}"
            cells += [f"{R[si, s, 2]:.4f}", g]
        out(f"  {name:<12} {cells[0]:>11} {cells[1]:>8} {cells[2]:>9} {cells[3]:>8} {cells[4]:>11} {cells[5]:>8}")
    ref = sub_names.index("v7+v9")
    for name in ["v7+v13", "v9+v13", "v7+v9+v13"]:
        si = sub_names.index(name)
        out(f"  {name} vs v7+v9: held masked {R_held[si, 0, 2] - R_held[ref, 0, 2]:+.4f}, "
            f"held all {R_held[si, 1, 2] - R_held[ref, 1, 2]:+.4f}, full masked {R_full[si, 0, 2] - R_full[ref, 0, 2]:+.4f}")
    out("\nResidual correlation (y - p, both clipped) on the scored rows of the 469; lower = more diverse")
    out(f"  {'pair':<8} {'t0':>7} {'t1':>7} {'mean':>7} | {'|y|-weighted t0':>15} {'t1':>7} {'mean':>7}")
    for pi, (pa, pb) in enumerate(PAIRS):
        c = resid_corr(RS[pi])  # (2 weightings, 2 targets)
        out(f"  {pa + '-' + pb:<8} {c[0, 0]:>7.4f} {c[0, 1]:>7.4f} {c[0].mean():>7.4f} | "
            f"{c[1, 0]:>15.4f} {c[1, 1]:>7.4f} {c[1].mean():>7.4f}")
    gain = R_held[sub_names.index("v7+v13"), 0, 2] - R_held[ref, 0, 2]
    full = R_full[sub_names.index("v7+v13"), 0, 2] - R_full[ref, 0, 2]
    go = gain >= 0.002 and full >= 0
    out(f"\nDecision rule (v7+v13 >= v7+v9 + 0.002 on the 469 masked AND not worse on all 1,873 masked): "
        f"{gain:+.4f}, {full:+.4f} -> {'CONTINUE' if go else 'STOP'}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
