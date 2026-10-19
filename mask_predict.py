"""What does the valid is_scored mask look like, and can it be predicted from the past? CPU only.

Part 1 (all 1,873 valid sequences, steps >= 99, no model): scored fraction per sequence and per step
bin, run lengths of scored rows and gaps vs an independent Bernoulli mask with each sequence's rate,
mask autocorrelation and periodicity, and how the mask relates to |target| (future info: upper bound).

Part 2 (600 random valid sequences, steps >= 99, every 10th step): LightGBM classifier for is_scored
from causal features only: the 168 standardized model inputs at the current step (train_v1.build_inputs
recipe, rebuilt in numpy with the v2 checkpoint's stats), rolling activity over the last 20 / 200 steps
(mean |standardized change| across the 44 prices and across the 68 other features), and the step.
Split by sequence: 400 train (350 fit + 50 early stopping), 200 test.

  CUDA_VISIBLE_DEVICES= nohup nice -n 10 python mask_predict.py > ~/work/mask_predict.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "8"

import sys
import time

import numpy as np
from scipy.stats import rankdata

CACHE = "/home/jovyan/work/cache"
CKPT = "/home/jovyan/work/checkpoints/train_v2_best.pt"  # source of the standardization stats only
RESULTS_OUT = os.environ.get("RESULTS_OUT", "/home/jovyan/work/mask_predict_results.txt")
FILL = 5.19921875  # as train_v1.FILL
WARMUP, T = 99, 20000
BINS = [(99, 1000), (1000, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
N_SAMPLE, N_TRAIN, N_ES, EVERY = 600, 400, 50, 10
WINDOWS = [20, 200]
THREADS = 8
SEED = 0

LINES = []
T0 = time.time()


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def log(s):
    print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)


def runs(m):
    """Lengths of runs of True, and of False runs strictly between two True runs."""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts, starts[1:] - ends[:-1]


def summ(x):
    x = np.asarray(x)
    return f"mean {x.mean():7.2f}  median {np.median(x):5.0f}  p90 {np.percentile(x, 90):5.0f}  max {x.max():6,}"


def auc(score, y):
    """ROC AUC with ties counted half (rank formulation)."""
    y = y.astype(bool)
    npos, nneg = y.sum(), (~y).sum()
    r = rankdata(score)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def auc_discrete(score, y):
    """Same AUC for 37M rows of a discretized score: counts per unique value instead of ranks."""
    vals, inv = np.unique(score, return_inverse=True)
    pos = np.bincount(inv, weights=y, minlength=len(vals))
    neg = np.bincount(inv, minlength=len(vals)) - pos
    neg_below = np.cumsum(neg) - neg
    return (pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())


def avg_precision(score, y):
    o = np.argsort(-score, kind="stable")
    ys = y[o].astype(np.float64)
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return (prec * ys).sum() / ys.sum()


def top_decile_capture(score, y):
    k = int(np.ceil(0.1 * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].sum() / y.sum()


def part1(sc, tg):
    n = len(sc)
    m = sc[:, WARMUP:]
    out("== 2a. Scored fraction per sequence (steps >= 99) ==")
    f = m.mean(1)
    q = np.percentile(f, [0, 25, 50, 75, 100])
    out(f"  overall {m.mean():.4%} ({m.sum():,} of {m.size:,});  min {q[0]:.4%}  q1 {q[1]:.4%}  median {q[2]:.4%}  "
        f"q3 {q[3]:.4%}  max {q[4]:.4%};  sequences with no scored row: {(f == 0).sum()}")
    sd_binom = np.sqrt(m.mean() * (1 - m.mean()) / m.shape[1])
    out(f"  SD of per-sequence fraction {f.std():.4%} vs {sd_binom:.4%} expected if every sequence had the same rate")

    out("\n== 2b. Scored share per step bin ==")
    for lo, hi in BINS:
        b = sc[:, lo:hi]
        out(f"  {lo:>5}-{hi - 1:<5} {b.mean():.4%}  ({b.sum():,} of {b.size:,})")

    out("\n== 2c. Run lengths (per sequence, steps >= 99) vs independent Bernoulli with each sequence's own rate ==")
    rng = np.random.default_rng(SEED)
    on, gap, on_b, gap_b = [], [], [], []
    for i in range(n):
        a, g = runs(m[i])
        on.append(a)
        gap.append(g)
        a, g = runs(rng.random(m.shape[1]) < f[i])
        on_b.append(a)
        gap_b.append(g)
    on, gap, on_b, gap_b = map(np.concatenate, [on, gap, on_b, gap_b])
    out(f"  scored runs    actual    n {len(on):>9,}  {summ(on)}")
    out(f"                 Bernoulli n {len(on_b):>9,}  {summ(on_b)}")
    out(f"  gaps           actual    n {len(gap):>9,}  {summ(gap)}")
    out(f"                 Bernoulli n {len(gap_b):>9,}  {summ(gap_b)}")
    for L in [1, 2, 5, 10, 50]:
        out(f"  share of scored rows in runs of length >= {L:>2}: actual {on[on >= L].sum() / on.sum():.4f}  "
            f"Bernoulli {on_b[on_b >= L].sum() / on_b.sum():.4f}")
    hist = np.bincount(np.minimum(on, 30))
    out("  scored-run length histogram (actual; 30 = 30+): "
        + ", ".join(f"{L}:{hist[L]:,}" for L in range(1, len(hist)) if hist[L]))

    out("\n== 2d. Mask autocorrelation (pooled within sequences) and periodicity ==")
    x = m.astype(np.float32)
    mu, var = x.mean(), x.var()
    for lag in [1, 10, 100, 1000]:
        c = ((x[:, lag:] - mu) * (x[:, :-lag] - mu)).mean() / var
        out(f"  lag {lag:>4}: {c:+.4f}")
    steps = np.arange(WARMUP, T)
    for k in [10, 50, 100, 1000]:
        share = np.array([m[:, (steps % k) == r].mean() for r in range(k)])
        cnt = np.array([((steps % k) == r).sum() * n for r in range(k)])
        z = (share - m.mean()) / np.sqrt(m.mean() * (1 - m.mean()) / cnt)
        out(f"  step mod {k:>4}: share min {share.min():.4%} max {share.max():.4%}  max |z| {np.abs(z).max():.1f} "
            f"(binomial; ~{np.sqrt(2 * np.log(k)) + 1:.1f} expected by chance at most)")

    out("\n== 2e. Mask vs |target| (future information: an upper bound on what is knowable) ==")
    y = m.ravel()
    a = np.abs(np.clip(tg[:, WARMUP:], -2, 2))
    for j, name in enumerate(["|t0|", "|t1|"]):
        v = a[:, :, j].ravel()
        out(f"  {name} clipped: AUC for is_scored {auc_discrete(v, y):.4f};  inside mask min {v[y].min():.4f} "
            f"median {np.median(v[y]):.4f};  outside min {v[~y].min():.4f} median {np.median(v[~y]):.4f};  "
            f"exact zeros inside {(v[y] == 0).mean():.2%} outside {(v[~y] == 0).mean():.2%}")
    s = a.sum(2).ravel()
    out(f"  |t0|+|t1| clipped: AUC {auc_discrete(s, y):.4f}")
    # Targets are discretized (one value holds about a third of rows), so bins are quantile edges
    # with ties kept together: a hard threshold would show 0% below it and 100% above.
    edges = np.unique(np.percentile(s, np.arange(0, 101, 10), method="lower"))
    out("  scored share by |t0|+|t1| bin (quantile edges, ties kept together):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (s >= lo) & ((s < hi) if hi < edges[-1] else (s <= hi))
        out(f"    [{lo:.3f}, {hi:.3f}{')' if hi < edges[-1] else ']'}: {y[sel].mean():8.4%} of {sel.sum():>10,} rows "
            f"({sel.mean():6.2%} of all)")


def seq_features(prices, other, mean, std, flag_idx):
    """(T, 168 + 4) causal features for one sequence: standardized inputs + rolling activity."""
    prices = prices.astype(np.float32)
    other = other.astype(np.float32)
    dp = np.zeros_like(prices)
    dp[1:] = prices[1:] - prices[:-1]
    flags = (np.abs(other[:, flag_idx]) == FILL).astype(np.float32)
    cont = (np.concatenate([prices, dp, other], 1) - mean) / std
    x = np.concatenate([cont, flags], 1)
    n_p = prices.shape[1]
    dpz = cont[:, n_p:2 * n_p]  # standardized price diffs (0 at step 0)
    oz = cont[:, 2 * n_p:]
    do = np.zeros_like(oz)
    do[1:] = oz[1:] - oz[:-1]
    act = [np.abs(dpz).mean(1), np.abs(do).mean(1)]
    roll = []
    for w in WINDOWS:
        for a in act:  # mean over steps t-w+1 .. t (fewer at the start)
            cs = np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])
            t = np.arange(len(a))
            lo = np.maximum(t + 1 - w, 0)
            roll.append(((cs[t + 1] - cs[lo]) / (t + 1 - lo)).astype(np.float32))
    return np.concatenate([x, np.stack(roll, 1)], 1)


def fit_eval(name, Xtr, ytr, Xes, yes, Xte, yte, cols):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=THREADS, seed=SEED, verbose=-1, deterministic=True, force_row_wise=True)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=cols, free_raw_data=False)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    t = time.time()
    b = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[des],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = b.predict(Xte, num_iteration=b.best_iteration)
    r = dict(auc=auc(p, yte), ap=avg_precision(p, yte), lift=top_decile_capture(p, yte), iters=b.best_iteration)
    log(f"{name}: AUC {r['auc']:.4f}, AP {r['ap']:.4f}, top-decile capture {r['lift']:.4f}, "
        f"{b.best_iteration} trees, {time.time() - t:.0f}s")
    return r, b


def part2(valid, sc, mean, std, flag_idx, price_cols, other_cols, flag_cols):
    n_price, n_other = len(price_cols), len(other_cols)
    rng = np.random.default_rng(SEED)
    n = len(sc)
    pick = np.sort(rng.choice(n, N_SAMPLE, replace=False))
    perm = rng.permutation(N_SAMPLE)
    role = np.empty(N_SAMPLE, dtype=np.int8)  # 0 fit, 1 early stopping, 2 test
    role[perm[:N_TRAIN - N_ES]], role[perm[N_TRAIN - N_ES:N_TRAIN]], role[perm[N_TRAIN:]] = 0, 1, 2
    steps = np.arange(WARMUP, T, EVERY)
    cols = (list(price_cols) + [f"d_{c}" for c in price_cols] + list(other_cols)
            + [f"flag_{c}" for c in flag_cols]
            + [f"act_{k}_{w}" for w in WINDOWS for k in ["price", "other"]] + ["step"])
    X = np.zeros((N_SAMPLE * len(steps), len(cols)), dtype=np.float32)
    y = np.zeros(len(X), dtype=np.int8)
    g = np.repeat(role, len(steps))
    for k, i in enumerate(pick):
        f = seq_features(np.asarray(valid["prices"][i]), np.asarray(valid["features"][i]), mean, std, flag_idx)
        s = slice(k * len(steps), (k + 1) * len(steps))
        X[s, :-1] = f[steps]
        X[s, -1] = steps
        y[s] = sc[i, steps]
        if (k + 1) % 100 == 0:
            log(f"features {k + 1}/{N_SAMPLE} sequences")
    nc = 2 * n_price + n_other
    zc = X[:, :nc]
    log(f"standardized inputs on the sample: mean of column means {zc.mean(0).mean():+.3f}, "
        f"median column std {np.median(zc.std(0)):.3f} (expect ~0 and ~1)")
    tr, es, te = g == 0, g == 1, g == 2
    out(f"\n== 3. Predicting is_scored from causal features ==")
    out(f"  sample: {N_SAMPLE} valid sequences (seed {SEED}), steps 99..19999 every {EVERY}th -> {len(X):,} rows, "
        f"scored rate {y.mean():.4%}")
    out(f"  split by sequence: {N_TRAIN - N_ES} fit + {N_ES} early stopping = {N_TRAIN} train, "
        f"{N_SAMPLE - N_TRAIN} test ({te.sum():,} rows, {y[te].sum():,} scored)")
    out("  model: LightGBM binary, lr 0.05, 63 leaves, early stopping on the 50 held-out train sequences")
    out("  top-decile capture = share of all scored test rows in the top 10% of predicted probability (random: 0.10)")
    out(f"  {'feature set':<34} {'AUC':>7} {'AP':>7} {'top10%':>7} {'trees':>6}")
    base = y[te].mean()
    sets = {"all 173 features": np.arange(len(cols)),
            "step index only": np.array([len(cols) - 1]),
            "rolling activity only (4)": np.arange(len(cols) - 5, len(cols) - 1)}
    res = {}
    for name, idx in sets.items():
        c = [cols[j] for j in idx]
        r, b = fit_eval(name, X[tr][:, idx], y[tr], X[es][:, idx], y[es], X[te][:, idx], y[te], c)
        res[name] = (r, b, c)
        out(f"  {name:<34} {r['auc']:>7.4f} {r['ap']:>7.4f} {r['lift']:>7.4f} {r['iters']:>6}")
    ysh = y[tr].copy()
    rng.shuffle(ysh)
    yes_sh = y[es].copy()
    rng.shuffle(yes_sh)
    r, _ = fit_eval("shuffled labels", X[tr], ysh, X[es], yes_sh, X[te], y[te], cols)
    out(f"  {'all 173, shuffled train labels':<34} {r['auc']:>7.4f} {r['ap']:>7.4f} {r['lift']:>7.4f} {r['iters']:>6}")
    out(f"  (AP of a random ranking = scored rate in test = {base:.4f})")
    r, b, c = res["all 173 features"]
    imp = b.feature_importance("gain")
    o = np.argsort(-imp)[:15]
    out("  top 15 features by gain (all-features model; d_ = first difference, act_ = rolling activity):")
    for j in o:
        out(f"    {c[j]:<16} {imp[j] / imp.sum():.2%}")
    groups = {"a0-a7": [k for k, n in enumerate(c) if n.startswith("a") and not n.startswith("act")],
              "prices": list(range(n_price)), "price diffs": list(range(n_price, 2 * n_price)),
              "volumes / other book cols": [k for k, n in enumerate(c[2 * n_price:2 * n_price + n_other], 2 * n_price)
                                            if not n.startswith("a")],
              "fill flags": [k for k, n in enumerate(c) if n.startswith("flag_")],
              "rolling activity": [k for k, n in enumerate(c) if n.startswith("act_")], "step": [len(c) - 1]}
    out("  gain share by feature group: " + ", ".join(f"{g} {imp[ix].sum() / imp.sum():.1%}" for g, ix in groups.items()))
    return res


def main():
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mean = ck["stats"]["mean"].numpy().astype(np.float32)
    std = ck["stats"]["std"].numpy().astype(np.float32)
    flag_idx = [ck["other_cols"].index(c) for c in ck["flag_cols"]]
    valid = {k: np.load(f"{CACHE}/valid_{k}.npy", mmap_mode="r") for k in ["prices", "features", "targets"]}
    sc = np.load(f"{CACHE}/valid_is_scored.npy")
    assert sc.shape == (1873, T) and not sc[:, :WARMUP].any()
    out("Valid is_scored mask diagnostic (CPU only); valid cache in /home/jovyan/work/cache, read-only")
    out(f"standardization stats from {CKPT}; inputs follow train_v1.build_inputs\n")
    log("part 1: mask structure")
    tg = np.asarray(valid["targets"], dtype=np.float32)
    part1(sc, tg)
    del tg
    log("part 2: predictability")
    part2(valid, sc, mean, std, flag_idx, ck["price_cols"], ck["other_cols"], ck["flag_cols"])
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    log(f"saved {RESULTS_OUT}")


if __name__ == "__main__":
    main()
