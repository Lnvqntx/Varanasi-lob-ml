"""Ensemble check across data seeds: the weight-decay models of data seed 0 and data seed 1, with v7 and v9.

ensemble_check8.py's method (GPU 0, 6 GB cap, batches of 25, 1000-step chunks; raw outputs averaged, then clipped
to [-2, 2] by the metric in spread_v2.seq_stats; per-sequence float64 sufficient statistics on is_scored & step
>= 99 and on all steps >= 99). ONE pass over all 1,873 valid sequences with every model run on each batch, so the
disk is read once; the 469 held-out sequences (valid index % 4 == 0) are a subset of the same statistics.
Models (final checkpoints, step 64500; all 2-layer GRUs, loaded in train_v1.GRUModel, dropout inactive in eval):
  data seed 0: v16a (drop 0.1, wd 0.05), v17a (drop 0.2, wd 0.05), v17c (drop 0.1, wd 0.05, init seed 1),
               v18a (wd 0.1), v18c (wd 0.05)
  data seed 1: v19a (wd 0.1), v19c (wd 0.05)
  references:  v7 (_best.pt = its final, data seed 0, wd 0.01), v9 (_best.pt, step 51200; no final saved; data
               seed 1, wd 0.01)
Subsets: every single model, every pair (gain over the better member), and the triples made of a cross-seed pair
(one wd model of data seed 0 + one of data seed 1) plus any third model (reference only: too slow for the limit).
Residual correlation matrix (y - p, both clipped) of all models on the scored rows of the 469, unweighted and
weighted by |clipped y|, mean over t0 and t1.
Sanity (tolerance 0.00001): full-valid masked and all-rows scores of each single model against its checkpoint;
held-out 469 v7, v9, v7+v9 against ensemble_check5_stats.npz and the subsets shared with ensemble_check8_stats.npz.

Usage (queue_check9.sh starts it after v19):
  nohup nice -n 10 python ensemble_check9.py > ~/work/ensemble_check9.log 2>&1 &
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
from spread_v2 import GPU_MEM_CAP_GB, METRIC_CLIP, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ensemble_check7 sets "1" on import

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt",
         "v16a": f"{CKPT_DIR}/train_v16a_drop01_wd_final.pt",
         "v17a": f"{CKPT_DIR}/train_v17a_drop02_wd05_final.pt",
         "v17c": f"{CKPT_DIR}/train_v17c_drop01_wd05_initseed1_final.pt",
         "v18a": f"{CKPT_DIR}/train_v18a_wd10_final.pt",
         "v18c": f"{CKPT_DIR}/train_v18c_wd05_nodrop_final.pt",
         "v19a": f"{CKPT_DIR}/train_v19a_wd10_ds1_final.pt",
         "v19c": f"{CKPT_DIR}/train_v19c_wd05_nodrop_ds1_final.pt"}
NAMES = list(CKPTS)
REFS = ["v7", "v9"]
WD_DS0 = ["v16a", "v17a", "v17c", "v18a", "v18c"]
WD_DS1 = ["v19a", "v19c"]
DATA_SEED = {"v7": 0, "v9": 1, **{k: 0 for k in WD_DS0}, **{k: 1 for k in WD_DS1}}
PAIRS = list(itertools.combinations(NAMES, 2))
CROSS = [(a, b) for a, b in PAIRS if (a in WD_DS0 and b in WD_DS1) or (a in WD_DS1 and b in WD_DS0)]
TRIPLES = sorted({tuple(sorted((a, b, c), key=NAMES.index)) for a, b in CROSS for c in NAMES if c not in (a, b)},
                 key=lambda t: [NAMES.index(m) for m in t])
SUBSETS = [(k,) for k in NAMES] + PAIRS + TRIPLES
CHECK5 = "/home/jovyan/work/ensemble_check5_stats.npz"
CHECK8 = "/home/jovyan/work/ensemble_check8_stats.npz"
STATS_OUT = "/home/jovyan/work/ensemble_check9_stats.npz"
RESULTS_OUT = "/home/jovyan/work/varanasi-lob-ml/ensemble_check9_results.txt"
SETS = ["masked", "all"]
TOL = 0.00001
REF = "v7+v9"


def resid_moments(y, preds):
    """y (r, 2), preds (m, r, 2) -> (2 weightings, 2 targets, m + 1 + m*m) float64 sums [w | w r_i | w r_i r_j]."""
    y = np.clip(y, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    r = y[None] - np.clip(preds, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)  # (m, rows, 2)
    out = []
    for w in [np.ones_like(y), np.abs(y)]:
        s1 = np.einsum("rt,mrt->tm", w, r)
        s2 = np.einsum("rt,irt,jrt->tij", w, r, r).reshape(2, -1)
        out.append(np.concatenate([w.sum(0)[:, None], s1, s2], axis=1))
    return np.stack(out)


def resid_corr_matrix(s, m):
    """(2 weightings, 2 targets, 1 + m + m*m) -> (2 weightings, m, m) correlation, mean over the two targets."""
    n = s[..., :1]
    mu = s[..., 1:1 + m] / n
    cov = s[..., 1 + m:].reshape(*s.shape[:2], m, m) / n[..., None] - mu[..., :, None] * mu[..., None, :]
    sd = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    return (cov / (sd[..., :, None] * sd[..., None, :])).mean(1)


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
          f"{len(SUBSETS)} subsets ({len(PAIRS)} pairs, {len(CROSS)} cross-seed, {len(TRIPLES)} triples)", flush=True)

    models = {k: load_gru(v, device) for k, v in CKPTS.items()}
    cks = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in CKPTS.items()}
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
    checks = []
    for k, ck in cks.items():
        checks.append((f"{k} full masked vs checkpoint", R_full[idx[k], 0, 2], ck["valid"]["weighted_pearson"]))
        checks.append((f"{k} full all rows vs checkpoint", R_full[idx[k], 1, 2], ck["valid_all"]["weighted_pearson"]))
    # ensemble_check5_stats.npz holds only the 469 held-out sequences, ensemble_check8_stats.npz all 1,873 (with the
    # held-out indices in "held"); select the held-out rows explicitly instead of summing everything stored.
    for path, label in [(CHECK5, "ensemble_check5"), (CHECK8, "ensemble_check8")]:
        c = np.load(path)
        assert np.array_equal(c["held"], np.flatnonzero(is_held)), f"{label}: different held-out sequences"
        n_c = c["stats"].shape[1]
        assert n_c in (n, len(c["held"])), f"{label}: stats cover {n_c} sequences"
        refs = {"held": wp_from_stats((c["stats"][:, c["held"]] if n_c == n else c["stats"]).sum(1))}
        if n_c == n:
            refs["full"] = wp_from_stats(c["stats"].sum(1))
        for ci, name in enumerate(c["subsets"]):
            if name in idx and (label == "ensemble_check8" or name in ("v7", "v9", "v7+v9")):
                for part, R in [("held", R_held), ("full", R_full)]:
                    if part in refs:
                        for s, set_name in enumerate(SETS):
                            checks.append((f"{name} {part} {set_name} vs {label}", R[idx[name], s, 2],
                                           refs[part][ci, s, 2]))
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

    ref = idx[REF]
    out("Ensemble check across data seeds: wd models of data seed 0 and 1 with v7 and v9, fp32 PyTorch, GPU 0, steps >= 99")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        a = cks[k]["args"]
        out(f"  {k:<5} data seed {DATA_SEED[k]}, init seed {a['seed']}, wd {a.get('wd', 0.01)}, dropout "
            f"{a.get('dropout', 0)}: {v} (step {cks[k]['step']})")
    out("  final weights, except v9 (its best, step 51200: its trainer saved no final; the v7+v9 reference uses it)")
    out("  held out = valid index % 4 == 0 (469 sequences); cross-seed = one wd model of data seed 0 + one of data seed 1")
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
        return (f"  {name:<16} {tag:<5} {cells[0]:>11} {cells[1]:>8} {cells[2]:>9} {cells[3]:>8} {cells[4]:>11} "
                f"{cells[5]:>8} {R_held[si, 0, 2] - R_held[ref, 0, 2]:>+9.4f} {R_full[si, 0, 2] - R_full[ref, 0, 2]:>+9.4f}")

    header = (f"  {'subset':<16} {'':<5} {'held masked':>11} {'gain':>8} {'held all':>9} {'gain':>8} {'full masked':>11} "
              f"{'gain':>8} {'held vs ref':>9} {'full vs ref':>9}")
    out(f"\nSingle models (ref = {REF}: held masked {R_held[ref, 0, 2]:.4f}, full masked {R_full[ref, 0, 2]:.4f})")
    out(header)
    for k in NAMES:
        out(row(k, f"ds{DATA_SEED[k]}"))
    out(f"\nAll {len(PAIRS)} pairs, sorted by full masked; gain = pair minus its better member; CROSS = cross-seed wd pair")
    out(header)
    for a, b in sorted(PAIRS, key=lambda p: -R_full[idx["+".join(p)], 0, 2]):
        out(row(f"{a}+{b}", "CROSS" if (a, b) in CROSS else ""))
    best = max(CROSS, key=lambda p: R_full[idx["+".join(p)], 0, 2])
    bn = "+".join(best)
    out(f"\nBest cross-seed pair (on all 1,873 masked): {bn} {R_full[idx[bn], 0, 2]:.4f}, "
        f"{R_full[idx[bn], 0, 2] - R_full[ref, 0, 2]:+.4f} vs {REF} ({R_full[ref, 0, 2]:.4f}); 469 masked "
        f"{R_held[idx[bn], 0, 2]:.4f}, {R_held[idx[bn], 0, 2] - R_held[ref, 0, 2]:+.4f} vs {REF} "
        f"({R_held[ref, 0, 2]:.4f}); 469 all rows {R_held[idx[bn], 1, 2] - R_held[ref, 1, 2]:+.4f}")
    best_h = max(CROSS, key=lambda p: R_held[idx["+".join(p)], 0, 2])
    if best_h != best:
        out(f"  (best cross-seed pair on the 469 masked instead: {'+'.join(best_h)} {R_held[idx['+'.join(best_h)], 0, 2]:.4f})")
    same = [p for p in PAIRS if p not in CROSS and p[0] not in REFS and p[1] not in REFS]
    for label, group in [("cross-seed wd pairs", CROSS), ("same-seed wd pairs", same)]:
        g = np.array([R_full[idx["+".join(p)], 0, 2] - max(R_full[idx[q], 0, 2] for q in p) for p in group])
        out(f"  {label}: {len(group)}, mean gain over the better member on all 1,873 masked {g.mean():+.4f} "
            f"(range {g.min():+.4f} .. {g.max():+.4f})")
    trip = sorted(TRIPLES, key=lambda s: -R_full[idx["+".join(s)], 0, 2])
    out(f"\nTriples with a cross-seed pair plus one other model (reference only: three models do not fit the time "
        f"limit); top 5 of {len(TRIPLES)} by full masked")
    out(header)
    for s in trip[:5]:
        out(row("+".join(s)))
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
