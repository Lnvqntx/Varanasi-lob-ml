"""Ensemble check for the v16 models (train_multi.py) with v7, v9 and v13.

ensemble_check6.py's method (6 GB cap, batches of 25, 1000-step chunks; raw outputs averaged, then clipped to
[-2, 2] by the metric in spread_v2.seq_stats; per-sequence float64 sufficient statistics on is_scored & step >= 99
and on all steps >= 99), one pass over all 1,873 valid sequences; the 469 held-out sequences (valid index % 4 == 0)
are a subset of the same statistics. Runs on GPU 1 (one process).
Checkpoints: the final (step 64500) weights of every model. v7 and v13 saved their best at step 64500, so their
_best.pt is the final; v9's trainer kept only its best (step 51200, the checkpoint behind the v7+v9 reference),
so v9 is that file. v16a/b/c are their _final.pt.
GRUs load in train_v1.GRUModel with the checkpoint's layer count (v16c: 3; v16a's dropout is inactive in eval);
v13 with train_v4m.LSTMModel.
Subsets: every single model, every pair among the 2-layer GRUs (v7, v9, v16a, v16b), v7+v13 and v16a+v13.
Residual diversity: per target, the correlation of the residuals (clipped y - clipped p) on the scored rows of the
469, unweighted and weighted by |clipped y|; averaged over t0 and t1.
Sanity (tolerance 0.00001): full-valid masked and all-rows scores of each single model against those stored in its
checkpoint; held-out 469 v7, v9 and v7+v9 against ensemble_check5_stats.npz.

Usage:
  nohup nice -n 10 python ensemble_check7.py > ~/work/ensemble_check7.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import sys
import time

import numpy as np

from ensemble_check2 import BATCH, predict
from ensemble_check6 import load_lstm, resid_corr, resid_sums
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

# ensemble_check2/6 and spread_v2 set CUDA_VISIBLE_DEVICES = "0" on import; put GPU 1 back before CUDA starts
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt",
         "v13": f"{CKPT_DIR}/train_v13_lstm_maskw_a05_h128_e5_seed0_best.pt",
         "v16a": f"{CKPT_DIR}/train_v16a_drop01_wd_final.pt",
         "v16b": f"{CKPT_DIR}/train_v16b_alpha075_final.pt",
         "v16c": f"{CKPT_DIR}/train_v16c_3layers_final.pt"}
GRU2 = ["v7", "v9", "v16a", "v16b"]
SUBSETS = [(k,) for k in CKPTS] + [(a, b) for i, a in enumerate(GRU2) for b in GRU2[i + 1:]] + \
    [("v7", "v13"), ("v16a", "v13")]
PAIRS = [("v16a", "v7"), ("v16a", "v9"), ("v7", "v9")]
CHECK5 = "/home/jovyan/work/ensemble_check5_stats.npz"
STATS_OUT = "/home/jovyan/work/ensemble_check7_stats.npz"
RESULTS_OUT = "/home/jovyan/work/varanasi-lob-ml/ensemble_check7_results.txt"
SETS = ["masked", "all"]
TOL = 0.00001
REF = "v7+v9"


def load_gru(path, device):
    import torch
    import train_v1
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["args"].get("arch", "gru") == "gru" and not ck["args"].get("aux")
    layers = ck["args"].get("layers") or 2
    model = train_v1.GRUModel(hidden=ck["args"]["hidden"], layers=layers).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    print(f"{path}: step {ck['step']}, GRU hidden {ck['args']['hidden']}, layers {layers}, "
          f"dropout {ck['args'].get('dropout', 0)}, wd {ck['args'].get('wd')}, alpha {ck['args'].get('alpha')}, "
          f"seed {ck['args']['seed']}, n_val {ck['args']['n_val']}", flush=True)
    return model, stats


def main():
    for p in [STATS_OUT, RESULTS_OUT]:
        if os.path.exists(p):
            sys.exit(f"{p} exists; not overwriting")
    import torch
    import train_v1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1" and not torch.cuda.is_initialized()
    device = torch.device("cuda:0")  # physical GPU 1
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)} (physical GPU 1), memory cap {GPU_MEM_CAP_GB} GB", flush=True)

    models = {k: (load_lstm if k == "v13" else load_gru)(v, device) for k, v in CKPTS.items()}
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

    out("Ensemble check: v16a/b/c (train_multi.py) with v7, v9 and v13, fp32 PyTorch, GPU 1, steps >= 99")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        out(f"  {k}: {v} (step {cks[k]['step']})")
    out("  final weights of each model; v9's file is its best (step 51200): its trainer saved no final")
    out("  all picked/scored on all 1,873 valid; held out = valid index % 4 == 0 (469 sequences)")
    out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
    for name, r, ref in checks:
        out(f"  {name}: {r:.7f} vs {ref:.7f} ({r - ref:+.1e})")
    out(f"\ngain = subset minus its best member; vs {REF} = subset minus {REF}")
    out(f"  {'subset':<10} {'held masked':>11} {'gain':>8} {'vs ' + REF:>8} {'held all':>9} {'gain':>8} {'vs ' + REF:>8}"
        f" {'full masked':>11} {'gain':>8} {'vs ' + REF:>8}")
    cols = [(R_held, 0), (R_held, 1), (R_full, 0)]
    ref = sub_names.index(REF)
    for si, name in enumerate(sub_names):
        cells = []
        for R, s in cols:
            g = ""
            if "+" in name:
                g = f"{R[si, s, 2] - max(R[sub_names.index(m), s, 2] for m in name.split('+')):+.4f}"
            cells += [f"{R[si, s, 2]:.4f}", g, f"{R[si, s, 2] - R[ref, s, 2]:+.4f}"]
        out(f"  {name:<10} {cells[0]:>11} {cells[1]:>8} {cells[2]:>8} {cells[3]:>9} {cells[4]:>8} {cells[5]:>8}"
            f" {cells[6]:>11} {cells[7]:>8} {cells[8]:>8}")
    pairs = [i for i, s in enumerate(SUBSETS) if len(s) == 2]
    best = max(pairs, key=lambda i: R_held[i, 0, 2])
    out(f"\nBest pair on the 469 masked: {sub_names[best]} {R_held[best, 0, 2]:.4f}, "
        f"{R_held[best, 0, 2] - R_held[ref, 0, 2]:+.4f} vs {REF} ({R_held[ref, 0, 2]:.4f}); "
        f"all 1,873 masked {R_full[best, 0, 2]:.4f}, {R_full[best, 0, 2] - R_full[ref, 0, 2]:+.4f} vs {REF} "
        f"({R_full[ref, 0, 2]:.4f}); held all rows {R_held[best, 1, 2] - R_held[ref, 1, 2]:+.4f}")
    out("\nResidual correlation (y - p, both clipped) on the scored rows of the 469; lower = more diverse")
    out(f"  {'pair':<10} {'t0':>7} {'t1':>7} {'mean':>7} | {'|y|-weighted t0':>15} {'t1':>7} {'mean':>7}")
    for pi, (pa, pb) in enumerate(PAIRS):
        c = resid_corr(RS[pi])  # (2 weightings, 2 targets)
        out(f"  {pa + '-' + pb:<10} {c[0, 0]:>7.4f} {c[0, 1]:>7.4f} {c[0].mean():>7.4f} | "
            f"{c[1, 0]:>15.4f} {c[1, 1]:>7.4f} {c[1].mean():>7.4f}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
