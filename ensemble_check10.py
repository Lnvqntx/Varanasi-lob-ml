"""Ensemble check with a third data seed: the wd models of data seeds 0, 1 and 2 (v24) and the 7-epoch wd models (v23).

ensemble_check9.py's method (GPU 0, 6 GB cap, batches of 25, 1000-step chunks; raw outputs averaged, then clipped
to [-2, 2] by the metric in spread_v2.seq_stats; per-sequence float64 sufficient statistics on is_scored & step
>= 99 and on all steps >= 99). ONE pass over all 1,873 valid sequences with every model run on each batch, so the
disk is read once; the 469 held-out sequences (valid index % 4 == 0) are a subset of the same statistics.
Models (final checkpoints, all 2-layer GRUs hidden 128, loaded in train_v1.GRUModel):
  data seed 0: v18c (wd 0.05), v18a (wd 0.1), v23b (7 epochs, wd 0.07), v23c (7 epochs, wd 0.1)
  data seed 1: v19a (wd 0.1), v19c (wd 0.05)
  data seed 2: v24a (wd 0.05), v24b (wd 0.07), v24c (wd 0.1)
  references:  v7 (_best.pt = its final, data seed 0, wd 0.01), v9 (_best.pt, step 51200, data seed 1, wd 0.01)
Subsets: every single model and every pair (no triples). Reported: all singles; all pairs whose members have
different data seeds, sorted by the all-1,873 masked score, with the gain over the better member, over v18c+v19a
(the current best submission) and over v7+v9; the best pair overall; residual correlations (y - p, both clipped)
on the scored rows of the 469, unweighted and |y|-weighted, mean over t0 and t1.
Sanity (tolerance 0.00001), each subset against its own reference: full-valid masked and all-rows scores of each
single model against its checkpoint; v7, v9, v7+v9 on the 469 against ensemble_check5_stats.npz (which stores
only the 469); every subset shared with ensemble_check9_stats.npz (which stores all 1,873, held-out rows selected
by its "held" indices), full and held-out, both sets.

  nohup nice -n 10 python ensemble_check10.py > ~/work/ensemble_check10.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import itertools
import sys
import time

import numpy as np

from ensemble_check2 import BATCH, predict
from ensemble_check7 import load_gru
from ensemble_check9 import resid_corr_matrix, resid_moments
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ensemble_check7 sets "1" on import

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt",
         "v18c": f"{CKPT_DIR}/train_v18c_wd05_nodrop_final.pt",
         "v18a": f"{CKPT_DIR}/train_v18a_wd10_final.pt",
         "v23b": f"{CKPT_DIR}/train_v23b_e7_wd07_final.pt",
         "v23c": f"{CKPT_DIR}/train_v23c_e7_wd10_final.pt",
         "v19a": f"{CKPT_DIR}/train_v19a_wd10_ds1_final.pt",
         "v19c": f"{CKPT_DIR}/train_v19c_wd05_nodrop_ds1_final.pt",
         "v24a": f"{CKPT_DIR}/train_v24a_ds2_wd05_final.pt",
         "v24b": f"{CKPT_DIR}/train_v24b_ds2_wd07_final.pt",
         "v24c": f"{CKPT_DIR}/train_v24c_ds2_wd10_final.pt"}
NAMES = list(CKPTS)
REFS = ["v7", "v9"]
DATA_SEED = {"v7": 0, "v9": 1, "v18c": 0, "v18a": 0, "v23b": 0, "v23c": 0, "v19a": 1, "v19c": 1,
             "v24a": 2, "v24b": 2, "v24c": 2}
PAIRS = list(itertools.combinations(NAMES, 2))
CROSS = [(a, b) for a, b in PAIRS if DATA_SEED[a] != DATA_SEED[b]]
SUBSETS = [(k,) for k in NAMES] + PAIRS
CHECK5 = "/home/jovyan/work/ensemble_check5_stats.npz"
CHECK9 = "/home/jovyan/work/ensemble_check9_stats.npz"
STATS_OUT = "/home/jovyan/work/ensemble_check10_stats.npz"
RESULTS_OUT = "/home/jovyan/work/varanasi-lob-ml/ensemble_check10_results.txt"
SETS = ["masked", "all"]
TOL = 0.00001
REF = "v18c+v19a"   # current best submission
REF2 = "v7+v9"      # previous submission


def main():
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    saved = np.load(STATS_OUT) if os.path.exists(STATS_OUT) else None  # an earlier pass: compare, never overwrite
    missing = [v for v in CKPTS.values() if not os.path.exists(v)]
    if missing:
        sys.exit(f"missing checkpoints: {missing}")
    import torch
    import train_v1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0" and not torch.cuda.is_initialized()
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)} (GPU 0), memory cap {GPU_MEM_CAP_GB} GB; {len(NAMES)} models, "
          f"{len(SUBSETS)} subsets ({len(PAIRS)} pairs, {len(CROSS)} with different data seeds)", flush=True)

    models = {k: load_gru(v, device) for k, v in CKPTS.items()}
    cks = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in CKPTS.items()}
    for k, ck in cks.items():
        a = ck["args"]
        assert a.get("data_seed", a["seed"]) == DATA_SEED[k], f"{k}: data seed {a.get('data_seed', a['seed'])}"
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    is_held = np.arange(n) % 4 == 0
    need = np.arange(T) >= WARMUP
    m = len(NAMES)
    S = np.zeros((len(SUBSETS), n, len(SETS), 2, 6), dtype=np.float64)
    RM = np.zeros((2, 2, 1 + m + m * m), dtype=np.float64)  # held-out scored rows only
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            preds = {k: predict(mod, st, b, device) for k, (mod, st) in models.items()}
            for si, sub in enumerate(SUBSETS):
                avg = np.mean([preds[k] for k in sub], axis=0, dtype=np.float32)
                for k, j in enumerate(batches[bi]):
                    S[si, j, 0] = seq_stats(b["targets"][k], avg[k], b["is_scored"][k] & need)
                    S[si, j, 1] = seq_stats(b["targets"][k], avg[k], need)
            for k, j in enumerate(batches[bi]):
                if is_held[j]:
                    mask = b["is_scored"][k] & need
                    RM += resid_moments(b["targets"][k][mask], np.stack([preds[q][k][mask] for q in NAMES]))
            if (bi + 1) % 10 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    sub_names = ["+".join(s) for s in SUBSETS]
    if saved is None:
        np.savez(STATS_OUT, stats=S, resid_moments=RM, subsets=sub_names, models=NAMES, sets=SETS,
                 stat_names=STAT_NAMES, held=np.flatnonzero(is_held), ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
        print(f"saved {STATS_OUT}", flush=True)
    else:
        assert list(saved["subsets"]) == sub_names and list(saved["ckpts"]) == [f"{k}={v}" for k, v in CKPTS.items()]
        d_s, d_r = np.abs(saved["stats"] - S).max(), np.abs(saved["resid_moments"] - RM).max()
        print(f"{STATS_OUT} exists (earlier pass); kept. Max abs difference of this pass vs it: stats {d_s:.3e}, "
              f"residual moments {d_r:.3e}", flush=True)
        if d_s > 0 or d_r > 0:
            sys.exit("STOP: this pass differs from the saved stats; no results written")

    R_full = wp_from_stats(S.sum(1))  # (n_subsets, n_sets, 3)
    R_held = wp_from_stats(S[:, is_held].sum(1))
    idx = {s: i for i, s in enumerate(sub_names)}

    def find(name):
        """subset index of 'a+b' in either order"""
        if name in idx:
            return idx[name]
        parts = name.split("+")
        return idx.get("+".join(reversed(parts))) if len(parts) == 2 else None

    checks = []
    for k, ck in cks.items():
        checks.append((f"{k} full masked vs checkpoint", R_full[idx[k], 0, 2], ck["valid"]["weighted_pearson"]))
        checks.append((f"{k} full all rows vs checkpoint", R_full[idx[k], 1, 2], ck["valid_all"]["weighted_pearson"]))
    # ensemble_check5_stats.npz holds only the 469 held-out sequences, ensemble_check9_stats.npz all 1,873 (held-out
    # indices in "held"); select the held-out rows explicitly instead of summing everything stored.
    for path, label in [(CHECK5, "ensemble_check5"), (CHECK9, "ensemble_check9")]:
        c = np.load(path)
        assert np.array_equal(c["held"], np.flatnonzero(is_held)), f"{label}: different held-out sequences"
        n_c = c["stats"].shape[1]
        assert n_c in (n, len(c["held"])), f"{label}: stats cover {n_c} sequences"
        refs = {"held": wp_from_stats((c["stats"][:, c["held"]] if n_c == n else c["stats"]).sum(1))}
        if n_c == n:
            refs["full"] = wp_from_stats(c["stats"].sum(1))
        for ci, name in enumerate(c["subsets"]):
            i = find(str(name))
            if i is None or (label == "ensemble_check5" and str(name) not in ("v7", "v9", "v7+v9")):
                continue
            for part, R in [("held", R_held), ("full", R_full)]:
                if part in refs:
                    for s, set_name in enumerate(SETS):
                        checks.append((f"{name} {part} {set_name} vs {label}", R[i, s, 2], refs[part][ci, s, 2]))
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

    ref, ref2 = idx[REF], idx[REF2]
    out("Ensemble check with three data seeds: wd models of data seeds 0 / 1 / 2 and the 7-epoch wd models, with v7 "
        "and v9; fp32 PyTorch, GPU 0, steps >= 99")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        a = cks[k]["args"]
        out(f"  {k:<5} data seed {DATA_SEED[k]}, init seed {a['seed']}, wd {a.get('wd', 0.01)}, epochs "
            f"{a.get('epochs')}: {v} (step {cks[k]['step']})")
    out("  final weights, except v9 (its best, step 51200: its trainer saved no final)")
    out("  held out = valid index % 4 == 0 (469 sequences); masked = is_scored rows")
    out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
    for name, r, ref_v in checks:
        out(f"  {name}: {r:.7f} vs {ref_v:.7f} ({r - ref_v:+.1e})")

    def row(name, tag=""):
        si = idx[name]
        cells = []
        for R, s in [(R_held, 0), (R_held, 1), (R_full, 0)]:
            g = ""
            if "+" in name:
                g = f"{R[si, s, 2] - max(R[idx[q], s, 2] for q in name.split('+')):+.4f}"
            cells += [f"{R[si, s, 2]:.4f}", g]
        return (f"  {name:<10} {tag:<7} {cells[0]:>11} {cells[1]:>8} {cells[2]:>9} {cells[3]:>8} {cells[4]:>11} "
                f"{cells[5]:>8} {R_held[si, 0, 2] - R_held[ref, 0, 2]:>+9.4f} {R_full[si, 0, 2] - R_full[ref, 0, 2]:>+9.4f}"
                f" {R_held[si, 0, 2] - R_held[ref2, 0, 2]:>+9.4f} {R_full[si, 0, 2] - R_full[ref2, 0, 2]:>+9.4f}")

    header = (f"  {'subset':<10} {'seeds':<7} {'held masked':>11} {'gain':>8} {'held all':>9} {'gain':>8} {'full masked':>11} "
              f"{'gain':>8} {'held-ref':>9} {'full-ref':>9} {'held-r2':>9} {'full-r2':>9}")
    legend = (f"  gain = pair minus its better member; ref = {REF} (held masked {R_held[ref, 0, 2]:.4f}, full masked "
              f"{R_full[ref, 0, 2]:.4f}); r2 = {REF2} (held masked {R_held[ref2, 0, 2]:.4f}, full masked "
              f"{R_full[ref2, 0, 2]:.4f}); differences on masked rows")
    out("\nSingle models")
    out(legend)
    out(header)
    for k in NAMES:
        out(row(k, f"ds{DATA_SEED[k]}"))
    out(f"\nAll {len(CROSS)} pairs with members from different data seeds, sorted by full masked "
        f"(REF = involves v7 or v9)")
    out(legend)
    out(header)
    for a, b in sorted(CROSS, key=lambda p: -R_full[idx["+".join(p)], 0, 2]):
        tag = f"{DATA_SEED[a]}x{DATA_SEED[b]}" + (" REF" if a in REFS or b in REFS else "")
        out(row(f"{a}+{b}", tag))
    best = max(PAIRS, key=lambda p: R_full[idx["+".join(p)], 0, 2])
    bn = "+".join(best)
    bi_ = idx[bn]
    out(f"\nBest pair overall (all {len(PAIRS)} pairs, by full masked): {bn} (data seeds {DATA_SEED[best[0]]} x "
        f"{DATA_SEED[best[1]]}): full masked {R_full[bi_, 0, 2]:.4f} ({R_full[bi_, 0, 2] - R_full[ref, 0, 2]:+.4f} vs "
        f"{REF}, {R_full[bi_, 0, 2] - R_full[ref2, 0, 2]:+.4f} vs {REF2}); 469 masked {R_held[bi_, 0, 2]:.4f} "
        f"({R_held[bi_, 0, 2] - R_held[ref, 0, 2]:+.4f} vs {REF}, {R_held[bi_, 0, 2] - R_held[ref2, 0, 2]:+.4f} vs "
        f"{REF2}); 469 all rows {R_held[bi_, 1, 2] - R_held[ref, 1, 2]:+.4f} vs {REF}")
    best_h = max(PAIRS, key=lambda p: R_held[idx["+".join(p)], 0, 2])
    out(f"  best pair on the 469 masked: {'+'.join(best_h)} {R_held[idx['+'.join(best_h)], 0, 2]:.4f}")
    if bn != REF:
        out(f"  {REF} itself: full masked {R_full[ref, 0, 2]:.4f}, 469 masked {R_held[ref, 0, 2]:.4f}; rank among all "
            f"pairs by full masked: {sorted(sub_names[len(NAMES):], key=lambda s: -R_full[idx[s], 0, 2]).index(REF) + 1}")
    gain_full = R_full[bi_, 0, 2] - R_full[ref, 0, 2]
    gain_held = R_held[bi_, 0, 2] - R_held[ref, 0, 2]
    met = gain_full >= 0.0015 and gain_held >= 0
    out(f"Build rule (best pair >= +0.0015 over {REF} on all 1,873 and >= 0 on the 469): "
        f"{'MET' if met else 'NOT MET'} ({gain_full:+.4f} / {gain_held:+.4f})")
    wd = [k for k in NAMES if k not in REFS]
    for combo in [(0, 1), (0, 2), (1, 2)]:
        g = np.array([R_full[idx["+".join(p)], 0, 2] - max(R_full[idx[q], 0, 2] for q in p) for p in CROSS
                      if p[0] in wd and p[1] in wd and {DATA_SEED[p[0]], DATA_SEED[p[1]]} == set(combo)])
        out(f"  wd pairs, data seeds {combo[0]} x {combo[1]}: {len(g)}, mean gain over the better member on all 1,873 "
            f"masked {g.mean():+.4f} (range {g.min():+.4f} .. {g.max():+.4f})")
    same = [p for p in PAIRS if p not in CROSS and p[0] in wd and p[1] in wd]
    g = np.array([R_full[idx["+".join(p)], 0, 2] - max(R_full[idx[q], 0, 2] for q in p) for p in same])
    out(f"  wd pairs, same data seed: {len(g)}, mean gain {g.mean():+.4f} (range {g.min():+.4f} .. {g.max():+.4f})")
    RC = resid_corr_matrix(RM, m)
    for w, label in [(0, "unweighted"), (1, "|y|-weighted")]:
        out(f"\nResidual correlation (y - p, both clipped) on the scored rows of the 469, {label}, mean of t0 and t1")
        out("  " + " " * 6 + "".join(f"{k:>7}" for k in NAMES))
        for i, k in enumerate(NAMES):
            out(f"  {k:<6}" + "".join(f"{RC[w, i, j]:>7.4f}" for j in range(m)))
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
