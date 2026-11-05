"""Causal predicted unit class per sequence, for train_multi.py unitpred=1 (no target leakage).

The unit (0.125433 / 0.241330 / 0.468536 / 0.656868, index 0-3) is predicted from the first K = 20 steps only, with
unit_probe.py's best classifier at K = 20: LightGBM (same parameters as unit_probe.fit_lgbm) on the per-column
mean, standard deviation and mean |step-to-step change| of the 168 unstandardised model inputs over steps 0..19
(unit_probe.inputs / window_features; 504 features). Labels for fitting are unit_labels.py's (from the targets).
  train: 5-fold OUT-OF-FOLD predictions (unit_probe.folds: stratified by unit, seed 0): every train sequence is
         predicted by a classifier that never saw its label, as a test sequence would be.
  valid: the classifier fit on all 10,607 train sequences (never on valid), as it would be used at test time.
That classifier is also saved (LightGBM text model) for a later export. Only the first 20 steps of the price and
feature caches are read (pread). Outputs, new files only (never overwritten; written as .tmp, then renamed):
  ~/work/cache/unit_pred_train.npy, unit_pred_valid.npy  int8 class per sequence (cache row order)
  ~/work/checkpoints/unit_pred_lgbm_k20.txt                the all-train classifier
  unit_pred_results.txt (repo)                              accuracies and confusion matrices
CPU only, 2 threads:  nice -n 10 python unit_pred.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "2"

import sys
import time

import numpy as np
import pyarrow.parquet as pq

from make_cache import DATA_DIR, price_and_other_cols
from unit_labels import unit_labels
from unit_probe import CACHE, FLAG_COLS, N_FOLD, SEED, UNITS, confusion, folds, inputs, read_head, window_features

K = 20
OUT = "/home/jovyan/work/varanasi-lob-ml/unit_pred_results.txt"
MODEL_OUT = "/home/jovyan/work/checkpoints/unit_pred_lgbm_k20.txt"
PARAMS = dict(objective="multiclass", num_class=len(UNITS), learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=20, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.5, seed=SEED,
              num_threads=2, deterministic=True, force_row_wise=True, verbose=-1)  # unit_probe.fit_lgbm
ROUNDS = 200


def pred_path(split):
    return f"{CACHE}/unit_pred_{split}.npy"


def unit_pred(split):
    """int8 predicted unit class per sequence of split (train: out-of-fold)."""
    return np.load(pred_path(split))


def features(split, flag_idx):
    ph = read_head(f"{CACHE}/{split}_prices.npy", K)
    fh = read_head(f"{CACHE}/{split}_features.npy", K)
    return window_features(inputs(ph, fh, flag_idx), K)


def main():
    outs = [pred_path("train"), pred_path("valid"), MODEL_OUT, OUT]
    if any(os.path.exists(p) for p in outs):
        sys.exit(f"an output exists; not overwriting: {[p for p in outs if os.path.exists(p)]}")
    import lightgbm as lgb
    t0 = time.time()
    _, other_cols = price_and_other_cols(pq.ParquetFile(f"{DATA_DIR}/train.parquet").schema_arrow.names)
    flag_idx = [other_cols.index(c) for c in FLAG_COLS]
    Xtr, Xva = features("train", flag_idx), features("valid", flag_idx)
    ytr, yva = unit_labels("train").astype(int), unit_labels("valid").astype(int)
    assert len(Xtr) == len(ytr) and len(Xva) == len(yva)
    print(f"features {Xtr.shape} / {Xva.shape} from the first {K} steps, {time.time() - t0:.0f}s", flush=True)
    fo = folds(ytr)
    ptr = np.full(len(ytr), -1)
    for f in range(N_FOLD):
        m = lgb.train(PARAMS, lgb.Dataset(Xtr[fo != f], ytr[fo != f]), num_boost_round=ROUNDS)
        ptr[fo == f] = m.predict(Xtr[fo == f]).argmax(1)
        print(f"  fold {f} done, {time.time() - t0:.0f}s", flush=True)
    assert (ptr >= 0).all()
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=ROUNDS)
    pva = m.predict(Xva).argmax(1)
    lines = [f"Predicted unit class from the first {K} steps (LightGBM, unit_probe.py parameters, {ROUNDS} rounds); "
             f"labels from unit_labels.py"]
    for name, yy, pp in [(f"train, {N_FOLD}-fold out-of-fold (stratified by unit, seed {SEED})", ytr, ptr),
                         ("valid, classifier fit on all train", yva, pva)]:
        lines.append(f"\n{name}: accuracy {np.mean(yy == pp):.5f} ({np.sum(yy != pp)} wrong of {len(yy)}); "
                     f"wrong sequences {np.flatnonzero(yy != pp).tolist()}")
        lines.append("    true\\pred " + " ".join(f"{u:>8}" for u in UNITS))
        c = confusion(yy, pp)
        for k, u in enumerate(UNITS):
            lines.append(f"    {u:>9} " + " ".join(f"{v:>8}" for v in c[k]))
    lines.append(f"\n(run time {time.time() - t0:.0f}s)")
    for p, a in [(pred_path("train"), ptr), (pred_path("valid"), pva)]:
        np.save(p + ".tmp.npy", a.astype(np.int8))
        os.rename(p + ".tmp.npy", p)
    m.save_model(MODEL_OUT + ".tmp")
    os.rename(MODEL_OUT + ".tmp", MODEL_OUT)
    with open(OUT + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.rename(OUT + ".tmp", OUT)
    print("\n".join(lines))
    print(f"saved {outs}")


if __name__ == "__main__":
    main()
