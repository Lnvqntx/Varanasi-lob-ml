"""Is the v7+v9 ensemble's error on the valid scored rows predictable from causal information? (CPU only)

Rows: is_scored & step >= 99 of all 1,873 valid sequences; y and pred clipped to [-2, 2] as in the metric.
Prediction: ~/work/calib_v7v9_pred_t{0,1}.npy (fp32 v7+v9 average saved as float16, see error_analysis.py).
Residual per target: r = y - b pred, b fitted by weighted least squares (weights |y|, no intercept) on all
scored rows. Features (all causal, at the current step):
  inputs    the 168 model inputs (44 prices, 44 price diffs, 68 other features, 12 fill-value flags; not
            standardized, which trees do not need)
  derived   the probe_features.py series at the current step: f0, f1 (i0 / i1 mid move), sp0 (i0 spread
            proxy), imb0, imb1 (i0 / i1 imbalance proxies)
  rolling   causal rolling mean and std of those five series over windows 5, 20, 100, 500
  step      step index;  pred  the ensemble prediction for t0 and t1
LightGBM regressor of r per target, sample weight |y| + 0.05, 4-fold CV grouped by sequence (sequence
folds from a seeded permutation; a fold's rows are predicted by a model that never saw those sequences).
Inside each training fold 10% of the sequences are held out for early stopping (weighted l2), and training
rows are subsampled to TRAIN_ROWS to keep each fit to minutes. Out-of-fold: weighted Pearson of pred
alone versus b pred + r_hat (clipped by the metric), per target and averaged, overall and per fold.
Importance: gain, normalized per model, averaged over the 8 models (4 folds x 2 targets).

  OMP_NUM_THREADS=4 nice -n 10 nohup python residual_probe.py > ~/work/residual_probe.log 2>&1 &
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time

import lightgbm as lgb
import numpy as np

import probe_features as pf
from error_analysis import CLIP, PRED, WARMUP, wp

CACHE = "/home/jovyan/work/cache"
RESULTS_OUT = "/home/jovyan/work/residual_probe_results.txt"
OOF_OUT = "/home/jovyan/work/residual_probe_oof.npz"
N_FOLDS = 4
SEED = 0
TRAIN_ROWS = 1_200_000
ES_ROWS = 300_000
THREADS = 4
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 2000,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0,
          "max_bin": 255, "num_threads": THREADS, "deterministic": True, "force_row_wise": True,
          "seed": SEED, "verbose": -1}
MAX_ROUNDS, ES_PATIENCE = 2000, 100


def feature_names():
    import train_v1
    names = pf.input_names(train_v1.PRICE_COLS, train_v1.OTHER_COLS)
    groups = []
    for c in names:
        if c.startswith("flag_"):
            groups.append("flag")
        elif c.startswith("d_"):
            groups.append(f"{c[2:4]} price diff")
        elif c[:3] in ("i0_", "i1_"):
            kind = c[3:].rstrip("0123456789")
            groups.append(f"{c[:2]} " + {"p": "price", "v": "volume", "dp": "trade price", "dv": "trade volume"}[kind])
        else:
            groups.append("additional a*")
    for s in pf.SERIES:
        names.append(s)
        groups.append("derived (current)")
    for s in pf.SERIES:
        for w in pf.WINDOWS:
            names += [f"{s}_mean{w}", f"{s}_std{w}"]
            groups += ["rolling mean", "rolling std"]
    names += ["step", "pred_t0", "pred_t1"]
    groups += ["step", "pred", "pred"]
    return names, groups


def build():
    fac = pf.fit_factors()
    P = np.load(f"{CACHE}/valid_prices.npy", mmap_mode="r")
    F = np.load(f"{CACHE}/valid_features.npy", mmap_mode="r")
    Y = np.load(f"{CACHE}/valid_targets.npy", mmap_mode="r")
    M = np.load(f"{CACHE}/valid_is_scored.npy", mmap_mode="r")
    pr = [np.load(p, mmap_mode="r") for p in PRED]
    n, T = M.shape
    names, _ = feature_names()
    mask = np.asarray(M).copy()
    mask[:, :WARMUP] = False
    N = int(mask.sum())
    X = np.empty((N, len(names)), dtype=np.float32)
    y = np.empty((N, 2), dtype=np.float64)
    p = np.empty((N, 2), dtype=np.float64)
    seq = np.empty(N, dtype=np.int32)
    t, o = time.time(), 0
    for j in range(n):
        steps = np.flatnonzero(mask[j])
        k = len(steps)
        prices, other = np.asarray(P[j]), np.asarray(F[j])
        s = pf.seq_series(prices, other, fac)
        cols = [pf.input_matrix(prices, other)[steps]]
        cols.append(np.stack([s[c][steps] for c in pf.SERIES], 1))
        for c in pf.SERIES:
            for w in pf.WINDOWS:
                m, sd = pf.rolling(s[c], w)
                cols.append(np.stack([m[steps], sd[steps]], 1))
        pj = np.stack([np.asarray(pr[i][j])[steps - WARMUP] for i in (0, 1)], 1).astype(np.float64)
        cols.append(np.column_stack([steps, pj]))
        X[o:o + k] = np.concatenate(cols, 1)
        y[o:o + k] = np.asarray(Y[j])[steps]
        p[o:o + k] = pj
        seq[o:o + k] = j
        o += k
        if (j + 1) % 200 == 0:
            print(f"  features {j + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    assert o == N
    return X, np.clip(y, -CLIP, CLIP), np.clip(p, -CLIP, CLIP), seq, names


def main():
    for f in [RESULTS_OUT, OOF_OUT]:
        if os.path.exists(f):
            sys.exit(f"{f} exists; not overwriting")
    X, y, p, seq, names = build()
    _, groups = feature_names()
    N, nf = X.shape
    print(f"X {X.shape} {X.nbytes / 2**30:.2f} GB", flush=True)
    w = np.abs(y)
    b = (w * y * p).sum(0) / (w * p * p).sum(0)
    r = y - b * p
    sw = w + 0.05
    n_seq = seq.max() + 1
    rng = np.random.default_rng(SEED)
    fold_of_seq = rng.permutation(n_seq) % N_FOLDS
    fold = fold_of_seq[seq]
    rhat = np.zeros_like(r)
    imp = np.zeros(nf)
    info = []
    t0 = time.time()
    for f in range(N_FOLDS):
        tr_seqs = np.flatnonzero(fold_of_seq != f)
        es_seqs = rng.choice(tr_seqs, size=len(tr_seqs) // 10, replace=False)
        is_es = np.isin(seq, es_seqs)
        tr_rows = np.flatnonzero((fold != f) & ~is_es)
        es_rows = np.flatnonzero(is_es)
        tr_rows = np.sort(rng.choice(tr_rows, size=min(TRAIN_ROWS, len(tr_rows)), replace=False))
        es_rows = np.sort(rng.choice(es_rows, size=min(ES_ROWS, len(es_rows)), replace=False))
        te_rows = np.flatnonzero(fold == f)
        assert not np.isin(seq[tr_rows], seq[te_rows]).any() and not np.isin(seq[es_rows], seq[te_rows]).any()
        Xtr, Xes = X[tr_rows], X[es_rows]
        for k in (0, 1):
            ts = time.time()
            dtr = lgb.Dataset(Xtr, r[tr_rows, k], weight=sw[tr_rows, k], feature_name=names, free_raw_data=True)
            des = lgb.Dataset(Xes, r[es_rows, k], weight=sw[es_rows, k], reference=dtr)
            m = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[des],
                          callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
            for s in range(0, len(te_rows), 500_000):
                idx = te_rows[s:s + 500_000]
                rhat[idx, k] = m.predict(X[idx], num_iteration=m.best_iteration, num_threads=THREADS)
            g = m.feature_importance("gain", iteration=m.best_iteration)
            imp += g / max(g.sum(), 1e-12)
            a = wp(y[te_rows, k], p[te_rows, k])
            c = wp(y[te_rows, k], np.clip(b[k] * p[te_rows, k] + rhat[te_rows, k], -CLIP, CLIP))
            info.append((f, k, m.best_iteration, a, c, len(tr_rows), len(te_rows), time.time() - ts))
            print(f"fold {f} t{k}: best iteration {m.best_iteration}, train rows {len(tr_rows):,}, "
                  f"test rows {len(te_rows):,} | wp pred {a:.4f} -> corrected {c:.4f} ({c - a:+.4f}) | "
                  f"{time.time() - ts:.0f}s, total {time.time() - t0:.0f}s", flush=True)
            del dtr, des, m
    np.savez(OOF_OUT, rhat=rhat.astype(np.float32), fold=fold, b=b)
    imp /= 2 * N_FOLDS
    corr = np.clip(b * p + rhat, -CLIP, CLIP)
    base = [wp(y[:, k], p[:, k]) for k in (0, 1)]
    new = [wp(y[:, k], corr[:, k]) for k in (0, 1)]
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("Residual probe: is the v7+v9 ensemble's error on valid scored rows predictable from causal inputs?")
    out(f"rows {N:,} (is_scored & step >= 99, 1,873 sequences), features {nf}; r = y - b pred, "
        f"b (WLS, weights |y|) t0 {b[0]:.4f}, t1 {b[1]:.4f}")
    out(f"LightGBM {lgb.__version__}, {THREADS} threads, params {PARAMS}, early stopping {ES_PATIENCE} on 10% of the "
        f"training sequences; training rows subsampled to {TRAIN_ROWS:,} per fit; {N_FOLDS}-fold CV grouped by sequence")
    out("\nOut-of-fold weighted Pearson on all scored rows (pooled over the 4 folds)")
    out(f"  {'':<10} {'t0':>8} {'t1':>8} {'mean':>8}")
    out(f"  {'pred':<10} {base[0]:>8.4f} {base[1]:>8.4f} {np.mean(base):>8.4f}")
    out(f"  {'pred + GBM':<10} {new[0]:>8.4f} {new[1]:>8.4f} {np.mean(new):>8.4f}")
    out(f"  {'gain':<10} {new[0] - base[0]:>+8.4f} {new[1] - base[1]:>+8.4f} {np.mean(new) - np.mean(base):>+8.4f}")
    out("\nPer fold (test rows predicted by a model that never saw their sequences)")
    out(f"  {'fold':>4} {'target':>6} {'best iter':>9} {'wp pred':>8} {'corrected':>9} {'gain':>8} {'test rows':>10} {'fit s':>6}")
    for f, k, it, a, c, ntr, nte, sec in info:
        out(f"  {f:>4} {'t' + str(k):>6} {it:>9} {a:>8.4f} {c:>9.4f} {c - a:>+8.4f} {nte:>10,} {sec:>6.0f}")
    out("\nTop 20 features by gain importance (normalized per model, mean over 4 folds x 2 targets)")
    out(f"  {'rank':>4} {'feature':<16} {'group':<20} {'share':>7}")
    for i, j in enumerate(np.argsort(-imp)[:20]):
        out(f"  {i + 1:>4} {names[j]:<16} {groups[j]:<20} {imp[j]:>7.4f}")
    gi = {}
    for j, gname in enumerate(groups):
        gi[gname] = gi.get(gname, 0.0) + imp[j]
    out("\nImportance by group")
    for gname, v in sorted(gi.items(), key=lambda kv: -kv[1]):
        out(f"  {gname:<20} {v:>7.4f}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
