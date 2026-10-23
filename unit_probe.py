"""Can the per-sequence target unit (0.1254 / 0.2413 / 0.4685 / 0.6569) be recognised from the warm-up inputs?

Labels: unit = smallest nonzero |target| in the sequence (both targets), matched to the four measured units
(0.125433 / 0.241330 / 0.468536 / 0.656868, relative tolerance 1e-4). To keep the disk load small the first
LABEL_ROWS steps are read for every sequence; when their smallest nonzero |target| is not a unit (the sequence never
moves by a single unit early on; 48 of 10,607 train sequences) the whole sequence is read instead. Every nonzero
|target| read must be an integer multiple of the label; on valid the label is checked against the full-sequence one.
The heads read from the caches are kept in SCRATCH (another disk) so a rerun does not read them again.
Inputs: the model's 168 inputs as in train_v1.build_inputs, before standardisation: 44 prices, 44 price
differences (0 at step 0), 68 other columns, 12 fill flags (|x| == FILL). Only the first READ_ROWS = 120 steps of
each sequence are read from the price and feature caches (pread with readahead disabled).
Per sequence and K in (20, 50, 99): per-column mean, standard deviation and mean |step-to-step change| over steps
0..K-1 -> 504 features (causal: only warm-up steps).
Classifiers: multinomial logistic regression (softmax, standardised features, L2, full-batch L-BFGS in torch on CPU;
scikit-learn is not installed) and LightGBM (small). 5-fold CV on train, stratified by unit, seed 0; one sample per
sequence, so folds are grouped by sequence. Then fit on all train, test on all 1,873 valid.
CPU only, 2 threads; run with ionice -c 3 nice -n 10.

  ionice -c 3 nice -n 10 python unit_probe.py > ~/work/unit_probe.log 2>&1
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

CACHE = "/home/jovyan/work/cache"
OUT = "/home/jovyan/work/varanasi-lob-ml/unit_probe_results.txt"
UNITS = [0.125433, 0.241330, 0.468536, 0.656868]
COARSE = 3
REL_TOL = 1e-4
SCRATCH = "/tmp/claude-1000/-home-jovyan-work-varanasi-lob-ml/83b783f2-117b-43c4-a9dc-95382dadc578/scratchpad"
READ_ROWS = 120
LABEL_ROWS = 2000
KS = [20, 50, 99]
FILL = 5.19921875  # train_v1.FILL
FLAG_COLS = ["a0", "a5", "a6", "a7"] + [f"i{i}_dv{j}" for i in range(2) for j in range(4)]  # train_v1.FLAG_COLS
N_FOLD = 5
SEED = 0
L2 = 1e-3


def read_head(path, rows):
    """First `rows` steps of every sequence of a (n, T, C) .npy file, reading only those bytes."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(fh)
        offset = fh.tell()
    assert not fortran
    n, T = shape[:2]
    row = int(np.prod(shape[2:], dtype=np.int64)) * dtype.itemsize
    out = np.empty((n, rows) + tuple(shape[2:]), dtype=dtype)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        for i in range(n):
            buf = os.pread(fd, rows * row, offset + i * T * row)
            out[i] = np.frombuffer(buf, dtype=dtype).reshape(out.shape[1:])
    finally:
        os.close(fd)
    return out


def match(y):
    """unit index of the smallest nonzero |y| (-1 if none or not a unit) and the integer-multiple check."""
    a = np.abs(y[y != 0]).astype(np.float64)
    if not len(a):
        return -1, False
    k = int(np.argmin([abs(a.min() / u - 1) for u in UNITS]))
    if abs(a.min() / UNITS[k] - 1) > REL_TOL:
        return -1, False
    return k, bool(np.abs(a / UNITS[k] - np.round(a / UNITS[k])).max() < 1e-3)


def label_units(y, full):
    """y (n, R, 2) heads, full (n, T, 2) memmap -> unit index per sequence, multiples flag, fallback indices."""
    n = len(y)
    lab, ok, fallback = np.full(n, -1), np.zeros(n, bool), []
    for i in range(n):
        lab[i], ok[i] = match(y[i])
        if lab[i] < 0:
            fallback.append(i)
            lab[i], ok[i] = match(np.asarray(full[i]))
    return lab, ok, fallback


def inputs(prices, other, flag_idx):
    """(n, R, 44) f32, (n, R, 68) f16 -> (n, R, 168) f32, train_v1.build_inputs without standardisation."""
    prices = prices.astype(np.float32)
    other = other.astype(np.float32)
    dp = np.zeros_like(prices)
    dp[:, 1:] = prices[:, 1:] - prices[:, :-1]
    flags = (np.abs(other[:, :, flag_idx]) == FILL).astype(np.float32)
    return np.concatenate([prices, dp, other, flags], axis=2)


def window_features(x, K):
    w = x[:, :K].astype(np.float64)
    return np.concatenate([w.mean(1), w.std(1), np.abs(np.diff(w, axis=1)).mean(1)], axis=1)


def fit_logreg(X, y, Xt):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float64)
    Xts = torch.tensor((Xt - mu) / sd, dtype=torch.float64)
    yt = torch.tensor(y)
    W = torch.zeros(X.shape[1], len(UNITS), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(len(UNITS), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=500, history_size=20, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xs @ W + b, yt) + L2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (Xts @ W + b).argmax(1).numpy(), W.detach().numpy() * 1.0


def fit_lgbm(X, y, Xt):
    import lightgbm as lgb  # native API: the sklearn wrapper needs scikit-learn
    params = dict(objective="multiclass", num_class=len(UNITS), learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=20, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.5, seed=SEED,
                  num_threads=2, deterministic=True, force_row_wise=True, verbose=-1)
    m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=200)
    return m.predict(Xt).argmax(1), m.feature_importance("gain")


def folds(y):
    rng = np.random.default_rng(SEED)
    f = np.zeros(len(y), int)
    for k in range(len(UNITS)):
        idx = rng.permutation(np.flatnonzero(y == k))
        f[idx] = np.arange(len(idx)) % N_FOLD
    return f


def confusion(y, p):
    c = np.zeros((len(UNITS), len(UNITS)), int)
    np.add.at(c, (y, p), 1)
    return c


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; not overwriting")
    t0 = time.time()
    price_cols, other_cols = price_and_other_cols(pq.ParquetFile(f"{DATA_DIR}/train.parquet").schema_arrow.names)
    flag_idx = [other_cols.index(c) for c in FLAG_COLS]
    names = price_cols + [f"d_{c}" for c in price_cols] + other_cols + [f"flag_{c}" for c in FLAG_COLS]
    assert len(names) == 168

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    data = {}
    for split in ["train", "valid"]:
        cache = f"{SCRATCH}/unit_probe_heads_{split}.npz"
        if os.path.exists(cache):
            z = np.load(cache)
            yh, ph, fh_ = z["targets"], z["prices"], z["features"]
        else:
            yh = read_head(f"{CACHE}/{split}_targets.npy", LABEL_ROWS)
            ph = read_head(f"{CACHE}/{split}_prices.npy", READ_ROWS)
            fh_ = read_head(f"{CACHE}/{split}_features.npy", READ_ROWS)
            np.savez(cache, targets=yh, prices=ph, features=fh_)
        lab, ok, fallback = label_units(yh, np.load(f"{CACHE}/{split}_targets.npy", mmap_mode="r"))
        x = inputs(ph, fh_, flag_idx)
        del ph, fh_
        data[split] = (lab, x)
        print(f"{split}: read {len(lab)} sequences, {time.time() - t0:.0f}s", flush=True)
        out(f"{split}: {len(lab)} sequences: " + ", ".join(f"{u} {np.sum(lab == k)}" for k, u in enumerate(UNITS))
            + f"; {len(fallback)} labelled from the full sequence (first {LABEL_ROWS} steps never at 1 unit); "
            f"unmatched {np.sum(lab < 0)}; integer-multiple check fails {np.sum(~ok)}")
        assert (lab >= 0).all() and ok.all(), f"{split}: bad labels {np.flatnonzero((lab < 0) | ~ok)[:20]}"

    # valid: compare with the full-sequence label (all 20,000 steps)
    tv = np.load(f"{CACHE}/valid_targets.npy", mmap_mode="r")
    full = np.array([match(np.asarray(tv[i]))[0] for i in range(len(tv))])
    out(f"valid: first-{LABEL_ROWS}-step label equals the full-sequence label for "
        f"{np.sum(full == data['valid'][0])}/{len(full)} sequences "
        f"(unit_analysis.py counts 398 / 347 / 431 / 697: {[int(np.sum(full == k)) for k in range(4)]})")

    (ytr, xtr), (yva, xva) = data["train"], data["valid"]
    fo = folds(ytr)
    out(f"\nFeatures: per-column mean, std and mean |change| of the 168 inputs over steps 0..K-1 (504 features). "
        f"CV: {N_FOLD} folds stratified by unit, seed {SEED}. Confusion rows = true unit, columns = predicted.")
    importances = {}
    for K in KS:
        Xtr, Xva = window_features(xtr, K), window_features(xva, K)
        for model, fit in [("logreg", fit_logreg), ("lgbm", fit_lgbm)]:
            cv = np.zeros(len(ytr), int)
            for f in range(N_FOLD):
                tr, te = fo != f, fo == f
                cv[te] = fit(Xtr[tr], ytr[tr], Xtr[te])[0]
            pv, imp = fit(Xtr, ytr, Xva)
            importances[(K, model)] = imp
            for name, yy, pp in [("train 5-fold CV", ytr, cv), ("train all -> valid 1,873", yva, pv)]:
                c = confusion(yy, pp)
                coarse_miss = c[COARSE].sum() - c[COARSE, COARSE]
                into_coarse = c[:, COARSE].sum() - c[COARSE, COARSE]
                out(f"\nK={K} {model}, {name}: accuracy {np.mean(yy == pp):.4f} ({np.sum(yy != pp)} wrong of {len(yy)}); "
                    f"0.6569 sequences called another unit {coarse_miss}/{c[COARSE].sum()}, "
                    f"other units called 0.6569 {into_coarse}/{len(yy) - c[COARSE].sum()}")
                out("    true\\pred " + " ".join(f"{u:>7}" for u in UNITS))
                for k, u in enumerate(UNITS):
                    out(f"    {u:>9} " + " ".join(f"{v:>7}" for v in c[k]))
            print(f"  K={K} {model} done, {time.time() - t0:.0f}s", flush=True)

    # which input columns carry the signal: LightGBM gain at K=99 (fit on all train), summed over the three stats
    gain = importances[(99, "lgbm")].reshape(3, 168)
    col = gain.sum(0)
    order = np.argsort(-col)[:10]
    out(f"\nTop 10 input columns, K=99, LightGBM gain summed over mean/std/mean|change| (share of total gain):")
    for j in order:
        st = ", ".join(f"{s} {g / gain.sum():.1%}" for s, g in zip(["mean", "std", "|chg|"], gain[:, j]))
        out(f"  {names[j]:<16} {col[j] / gain.sum():>6.1%}  ({st})")
    # single-column check: logistic regression on only that column's three stats, valid accuracy
    Xtr, Xva = window_features(xtr, 99), window_features(xva, 99)
    out("Single-column logistic regression (that column's 3 stats only, K=99), train all -> valid accuracy:")
    for j in order:
        cols = [j, 168 + j, 336 + j]
        pv = fit_logreg(Xtr[:, cols], ytr, Xva[:, cols])[0]
        out(f"  {names[j]:<16} {np.mean(pv == yva):.4f}")
    out(f"\n(run time {time.time() - t0:.0f}s)")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
