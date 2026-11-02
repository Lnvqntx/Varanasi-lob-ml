"""Three-model ensembles: the three wd models of data seeds 0 / 1 / 2 with the older recipes (v7, v9, v13 LSTM).

ensemble_check10.py's method (GPU 0, 6 GB cap, batches of 25, 1000-step chunks; raw outputs averaged, then clipped
to [-2, 2] by the metric in spread_v2.seq_stats; per-sequence float64 sufficient statistics on is_scored & step
>= 99 and on all steps >= 99). ONE pass over all 1,873 valid sequences with every model run on each batch, so the
disk is read once; the 469 held-out sequences (valid index % 4 == 0) are a subset of the same statistics.
Models:
  wd recipe (wd 0.05-0.1, 2-layer GRU hidden 128, final checkpoints): v18c (data seed 0), v19a (1), v24a (2)
  older recipe (wd 0.01, mask weighting alpha 0.5): v7 GRU (data seed 0, _best.pt = its final, step 64500),
  v9 GRU (data seed 1, _best.pt, step 51200: its trainer saved no final), v13 LSTM (data seed 0, _best.pt = its
  final, step 64500; loaded with train_v4m.LSTMModel)
Subsets: every single model, every pair and every triple. Reported: singles; pairs; every triple whose members
come from at least two different data seeds or recipes (recipe = wd / wd01 GRU / LSTM), sorted by the all-1,873
masked score, with the gain over its best member and over v18c+v19a (the current best submission); the best
triple of GRUs only and the best triple overall, and the build rule (>= +0.0015 over v18c+v19a on all 1,873 and
>= 0 on the 469); residual correlations (y - p, both clipped) on the scored rows of the 469, unweighted and
|y|-weighted, mean over t0 and t1.
Sanity (tolerance 0.00001): full-valid masked and all-rows scores of each single model against its checkpoint;
every subset also stored in ensemble_check6/7/9/10_stats.npz (all store the 1,873 sequences, held-out indices in
"held"), full and held-out, both sets.

  nohup nice -n 10 python ensemble_check11.py > ~/work/ensemble_check11.log 2>&1 &
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
from ensemble_check6 import load_lstm
from ensemble_check7 import load_gru
from ensemble_check9 import resid_corr_matrix, resid_moments
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ensemble_check7 sets "1" on import

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v18c": f"{CKPT_DIR}/train_v18c_wd05_nodrop_final.pt",
         "v19a": f"{CKPT_DIR}/train_v19a_wd10_ds1_final.pt",
         "v24a": f"{CKPT_DIR}/train_v24a_ds2_wd05_final.pt",
         "v7": f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         "v9": f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt",
         "v13": f"{CKPT_DIR}/train_v13_lstm_maskw_a05_h128_e5_seed0_best.pt"}
NAMES = list(CKPTS)
DATA_SEED = {"v18c": 0, "v19a": 1, "v24a": 2, "v7": 0, "v9": 1, "v13": 0}
RECIPE = {"v18c": "wd", "v19a": "wd", "v24a": "wd", "v7": "wd01", "v9": "wd01", "v13": "lstm"}
GRUS = [k for k in NAMES if RECIPE[k] != "lstm"]
PAIRS = list(itertools.combinations(NAMES, 2))
TRIPLES = list(itertools.combinations(NAMES, 3))
MIXED = [t for t in TRIPLES if len({DATA_SEED[k] for k in t}) > 1 or len({RECIPE[k] for k in t}) > 1]
SUBSETS = [(k,) for k in NAMES] + PAIRS + TRIPLES
REF_STATS = [f"/home/jovyan/work/ensemble_check{i}_stats.npz" for i in (6, 7, 9, 10)]
STATS_OUT = "/home/jovyan/work/ensemble_check11_stats.npz"
RESULTS_OUT = "/home/jovyan/work/varanasi-lob-ml/ensemble_check11_results.txt"
SETS = ["masked", "all"]
TOL = 0.00001
REF = "v18c+v19a"   # current best submission


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
          f"{len(SUBSETS)} subsets ({len(PAIRS)} pairs, {len(TRIPLES)} triples, {len(MIXED)} mixed)", flush=True)

    models = {k: (load_lstm if RECIPE[k] == "lstm" else load_gru)(v, device) for k, v in CKPTS.items()}
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
    idx = {frozenset(s): i for i, s in enumerate(SUBSETS)}

    def at(name):
        return idx[frozenset(name.split("+"))]

    checks = []
    for k, ck in cks.items():
        checks.append((f"{k} full masked vs checkpoint", R_full[at(k), 0, 2], ck["valid"]["weighted_pearson"]))
        checks.append((f"{k} full all rows vs checkpoint", R_full[at(k), 1, 2], ck["valid_all"]["weighted_pearson"]))
    # every reference file stores all 1,873 sequences; the held-out rows are selected by its "held" indices
    for path in REF_STATS:
        label = os.path.basename(path).replace("_stats.npz", "")
        c = np.load(path)
        assert np.array_equal(c["held"], np.flatnonzero(is_held)), f"{label}: different held-out sequences"
        assert c["stats"].shape[1] == n, f"{label}: stats cover {c['stats'].shape[1]} sequences"
        refs = {"held": wp_from_stats(c["stats"][:, c["held"]].sum(1)), "full": wp_from_stats(c["stats"].sum(1))}
        for ci, name in enumerate(c["subsets"]):
            key = frozenset(str(name).split("+"))
            if key not in idx:
                continue
            for part, R in [("held", R_held), ("full", R_full)]:
                for s, set_name in enumerate(SETS):
                    checks.append((f"{name} {part} {set_name} vs {label}", R[idx[key], s, 2], refs[part][ci, s, 2]))
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

    ref = at(REF)
    out("Three-model ensembles: wd models of data seeds 0 / 1 / 2 with v7, v9 (wd 0.01 GRUs) and v13 (LSTM); fp32 "
        "PyTorch, GPU 0, steps >= 99")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        a = cks[k]["args"]
        out(f"  {k:<5} {RECIPE[k]:<5} data seed {DATA_SEED[k]}, init seed {a['seed']}, wd {a.get('wd', 0.01)}: {v} "
            f"(step {cks[k]['step']})")
    out("  final weights; v7 and v13 _best.pt are their finals (step 64500); v9 is its best (step 51200, no final saved)")
    out("  held out = valid index % 4 == 0 (469 sequences); masked = is_scored rows")
    out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
    for name, r, ref_v in checks:
        out(f"  {name}: {r:.7f} vs {ref_v:.7f} ({r - ref_v:+.1e})")

    def row(sub):
        name = "+".join(sub)
        si = at(name)
        cells = []
        for R, s in [(R_held, 0), (R_held, 1), (R_full, 0), (R_full, 1)]:
            g = f"{R[si, s, 2] - max(R[at(q), s, 2] for q in sub):+.4f}" if len(sub) > 1 else ""
            cells += [f"{R[si, s, 2]:.4f}", g]
        tag = "/".join(f"{RECIPE[k]}{DATA_SEED[k]}" for k in sub)
        return (f"  {name:<15} {tag:<20} {cells[0]:>11} {cells[1]:>8} {cells[2]:>9} {cells[3]:>8} {cells[4]:>11} "
                f"{cells[5]:>8} {cells[6]:>9} {cells[7]:>8} {R_held[si, 0, 2] - R_held[ref, 0, 2]:>+9.4f} "
                f"{R_full[si, 0, 2] - R_full[ref, 0, 2]:>+9.4f}")

    header = (f"  {'subset':<15} {'recipe/seed':<20} {'held masked':>11} {'gain':>8} {'held all':>9} {'gain':>8} "
              f"{'full masked':>11} {'gain':>8} {'full all':>9} {'gain':>8} {'held-ref':>9} {'full-ref':>9}")
    legend = (f"  gain = subset minus its best member; ref = {REF} (held masked {R_held[ref, 0, 2]:.4f}, full masked "
              f"{R_full[ref, 0, 2]:.4f}); held = the 469, full = all 1,873; ref differences on masked rows")

    def by_full(subs):
        return sorted(subs, key=lambda s: -R_full[at("+".join(s)), 0, 2])

    out("\nSingle models")
    out(legend)
    out(header)
    for k in NAMES:
        out(row((k,)))
    out(f"\nAll {len(PAIRS)} pairs, sorted by full masked")
    out(header)
    for p in by_full(PAIRS):
        out(row(p))
    out(f"\n{len(MIXED)} of {len(TRIPLES)} triples with members from at least two data seeds or recipes, sorted by "
        f"full masked")
    out(legend)
    out(header)
    for tr in by_full(MIXED):
        out(row(tr))

    def describe(label, tr):
        si = at("+".join(tr))
        gf, gh = R_full[si, 0, 2] - R_full[ref, 0, 2], R_held[si, 0, 2] - R_held[ref, 0, 2]
        out(f"{label}: {'+'.join(tr)}: full masked {R_full[si, 0, 2]:.4f} ({gf:+.4f} vs {REF}), 469 masked "
            f"{R_held[si, 0, 2]:.4f} ({gh:+.4f}); all rows full {R_full[si, 1, 2]:.4f} "
            f"({R_full[si, 1, 2] - R_full[ref, 1, 2]:+.4f}), 469 {R_held[si, 1, 2]:.4f} "
            f"({R_held[si, 1, 2] - R_held[ref, 1, 2]:+.4f})")
        return gf, gh

    out("")
    best_gru = by_full([t for t in MIXED if all(k in GRUS for k in t)])[0]
    best_all = by_full(MIXED)[0]
    describe("Best triple of GRUs only", best_gru)
    gf, gh = describe("Best triple overall", best_all)
    best_h = max(MIXED, key=lambda s: R_held[at("+".join(s)), 0, 2])
    out(f"  best triple on the 469 masked: {'+'.join(best_h)} {R_held[at('+'.join(best_h)), 0, 2]:.4f}")
    best_pair = by_full(PAIRS)[0]
    out(f"  best pair (by full masked): {'+'.join(best_pair)} {R_full[at('+'.join(best_pair)), 0, 2]:.4f}")
    met = gf >= 0.0015 and gh >= 0
    out(f"Build rule (best triple >= +0.0015 over {REF} on all 1,873 and >= 0 on the 469): "
        f"{'MET' if met else 'NOT MET'} ({gf:+.4f} / {gh:+.4f})")
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
