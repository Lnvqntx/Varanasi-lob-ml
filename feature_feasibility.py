"""Can standard order-book features be derived from the anonymised columns, and would they add information?

Read-only, CPU only (2 threads), tiny data: the first 200 train sequences, first 120 rows each, sliced from the
memmapped caches (prices / features / targets); column names from make_cache.price_and_other_cols (parquet schema
only); the GRU input standardisation from the v18c checkpoint's stats (the 168 model inputs = train_v5m.build_inputs).
Writes feature_feasibility_results.txt only.

1. Structure of the 112 raw columns and the 168 inputs: value ranges and saturation, which columns change on the
   same rows, the principal components of the per-step price changes and their loadings (bid 0-10 vs ask 11-21),
   whether price changes sit on a tick grid (an affine scaling keeps a grid, a non-linear per-column transform does
   not), how the two instruments relate, and what the 8 anonymous columns look like.
2. For mid, spread, level-1 imbalance, total depth imbalance, microprice and the cross-instrument mid difference:
   whether they can be computed, and the proxy used when not exactly.
3. For each proxy: weighted Pearson (weights |clipped target| at the current step) with the current targets and with
   the 20-step-ahead change y[t+20] - y[t]; the R^2 of predicting it linearly from the 168 inputs at the same step
   (5-fold CV grouped by sequence; also with the inputs' mean over the previous 20 steps added, a crude stand-in for
   the history the GRU sees); and the change in the |y|-weighted CV correlation of a ridge from the 168 inputs to the
   target when the proxy is added (the new linear information). Standard errors: bootstrap over sequences.

  nice -n 10 python feature_feasibility.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "2"

from statistics import NormalDist

import numpy as np
import pyarrow.parquet as pq
import torch

from make_cache import DATA_DIR, price_and_other_cols

torch.set_num_threads(2)
CACHE = "/home/jovyan/work/cache"
CKPT = "/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt"
OUT = "/home/jovyan/work/varanasi-lob-ml/feature_feasibility_results.txt"
N_SEQ, N_ROW = 200, 120
AHEAD = 20
HIST = 20
N_BOOT = 200
SEED = 0
FILL = 5.19921875
Q99 = NormalDist().inv_cdf(0.99)  # 2.3263
Q1E7 = NormalDist().inv_cdf(1 - 1e-7)  # 5.1993, sklearn QuantileTransformer's output bound

lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def wcorr(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def boot_se(fn, n_seq, rng):
    """Bootstrap standard error over sequences; fn(seq index array) -> scalar."""
    v = [fn(rng.integers(0, n_seq, n_seq)) for _ in range(N_BOOT)]
    return float(np.nanstd(v))


def cv_r2(X, y, groups, w=None, lam=1e-3, k=5):
    """Out-of-fold R^2 (weighted if w) of a ridge from X to y, folds grouped by sequence."""
    pred = np.zeros_like(y)
    for f in range(k):
        te = groups % k == f
        A = X[~te]
        mu, sd = A.mean(0), A.std(0) + 1e-9
        Z = np.c_[(A - mu) / sd, np.ones(len(A))]
        ww = np.ones(len(A)) if w is None else w[~te]
        G = Z.T @ (Z * ww[:, None]) + lam * len(A) * np.eye(Z.shape[1])
        beta = np.linalg.solve(G, Z.T @ (ww * y[~te]))
        pred[te] = np.c_[(X[te] - mu) / sd, np.ones(te.sum())] @ beta
    ww = np.ones(len(y)) if w is None else w
    m = (ww * y).sum() / ww.sum()
    return 1 - (ww * (y - pred) ** 2).sum() / (ww * (y - m) ** 2).sum(), pred


def main():
    rng = np.random.default_rng(SEED)
    names = pq.ParquetFile(f"{DATA_DIR}/train.parquet").schema_arrow.names
    price_cols, other_cols = price_and_other_cols(names)
    P = np.asarray(np.load(f"{CACHE}/train_prices.npy", mmap_mode="r")[:N_SEQ, :N_ROW], np.float64)
    F = np.asarray(np.load(f"{CACHE}/train_features.npy", mmap_mode="r")[:N_SEQ, :N_ROW], np.float64)
    Y = np.asarray(np.load(f"{CACHE}/train_targets.npy", mmap_mode="r")[:N_SEQ, :N_ROW], np.float64)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert ck["price_cols"] == price_cols and ck["other_cols"] == other_cols
    from train_v5m import FLAG_COLS, build_inputs
    X168 = build_inputs(torch.from_numpy(P.astype(np.float32)), torch.from_numpy(F.astype(np.float16)),
                        ck["stats"]).double().numpy()
    raw = np.concatenate([P, F], axis=2)
    raw_names = price_cols + other_cols
    col = {n: raw[:, :, j] for j, n in enumerate(raw_names)}
    in_names = price_cols + [f"d_{c}" for c in price_cols] + other_cols + [f"flag_{c}" for c in FLAG_COLS]
    assert X168.shape[2] == len(in_names) == 168

    out(f"Feature feasibility: first {N_SEQ} train sequences x first {N_ROW} rows ({N_SEQ * N_ROW:,} rows), CPU, read-only")
    out(f"Column names: make_cache.price_and_other_cols (parquet schema); model inputs: train_v5m.build_inputs with "
        f"the v18c checkpoint's input stats")

    # ---------------------------------------------------------------- 1. structure
    out("\n=== 1. Structure of the 112 raw columns ===")
    groups = [("i0 bid price-like p0-p10", [f"i0_p{k}" for k in range(11)]),
              ("i0 ask price-like p11-p21", [f"i0_p{k}" for k in range(11, 22)]),
              ("i0 bid volume-like v0-v10", [f"i0_v{k}" for k in range(11)]),
              ("i0 ask volume-like v11-v21", [f"i0_v{k}" for k in range(11, 22)]),
              ("i0 trade price dp0-3", [f"i0_dp{k}" for k in range(4)]),
              ("i0 trade volume dv0-3", [f"i0_dv{k}" for k in range(4)]),
              ("i1 bid price-like p0-p10", [f"i1_p{k}" for k in range(11)]),
              ("i1 ask price-like p11-p21", [f"i1_p{k}" for k in range(11, 22)]),
              ("i1 bid volume-like v0-v10", [f"i1_v{k}" for k in range(11)]),
              ("i1 ask volume-like v11-v21", [f"i1_v{k}" for k in range(11, 22)]),
              ("i1 trade price dp0-3", [f"i1_dp{k}" for k in range(4)]),
              ("i1 trade volume dv0-3", [f"i1_dv{k}" for k in range(4)])] + \
             [(f"additional {c}", [c]) for c in [f"a{k}" for k in range(8)]]
    assert sorted(sum((g for _, g in groups), [])) == sorted(raw_names)
    out(f"Group stats (values pooled over the group's columns). Bounds: Phi^-1(0.99) = {Q99:.4f}; "
        f"Phi^-1(1 - 1e-7) = {Q1E7:.4f} (sklearn QuantileTransformer's clip; the cache fill value is {FILL})")
    out(f"  {'group':<28} {'mean':>6} {'sd':>5} {'in-seq sd':>9} {'min':>7} {'max':>6} {'|x|>2.32':>8} "
        f"{'|x|=fill':>8} {'rows changed':>12} {'all cols change together':>24}")
    for gname, cols in groups:
        A = np.stack([col[c] for c in cols], 2)
        ch = np.diff(A, axis=1) != 0
        together = (ch.all(2) | ~ch.any(2)).mean()
        out(f"  {gname:<28} {A.mean():>6.2f} {A.std():>5.2f} {A.std(1).mean():>9.3f} {A.min():>7.3f} {A.max():>6.3f} "
            f"{np.mean(np.abs(A) > 2.32):>8.3f} {np.mean(np.isclose(np.abs(A), FILL)):>8.3f} {ch.mean():>12.2f} "
            f"{together:>24.3f}")

    out("\nCo-change of groups (fraction of steps on which a group changes, given another changes):")
    chg = {g: (np.diff(np.stack([col[c] for c in cols], 2), axis=1) != 0).any(2) for g, cols in groups}
    keys = ["i0 bid price-like p0-p10", "i0 bid volume-like v0-v10", "i0 trade price dp0-3", "i0 trade volume dv0-3",
            "i1 bid price-like p0-p10", "i1 bid volume-like v0-v10", "i1 trade price dp0-3"]
    out(f"  {'P(row | column)':<28} " + " ".join(f"{k.split(' ')[0] + ' ' + k.split(' ')[2][:3] + ('p' if 'price' in k else 'v'):>9}" for k in keys))
    for a in keys:
        out(f"  {a:<28} " + " ".join(f"{(chg[a] & chg[b]).sum() / max(chg[b].sum(), 1):>9.2f}" for b in keys))
    ip0 = np.stack([col[f"i0_p{k}"] for k in range(22)], 2)
    ip1 = np.stack([col[f"i1_p{k}"] for k in range(22)], 2)
    out(f"  i0 bid and ask price columns change on the same rows: {np.mean(chg['i0 bid price-like p0-p10'] == chg['i0 ask price-like p11-p21']):.3f}"
        f"; i1: {np.mean(chg['i1 bid price-like p0-p10'] == chg['i1 ask price-like p11-p21']):.3f}")

    def pca_diffs(A):
        d = np.diff(A, axis=1).reshape(-1, A.shape[2])
        d = d[(d != 0).any(1)]
        w, v = np.linalg.eigh(np.cov(d.T))
        w, v = w[::-1], v[:, ::-1]
        u = v[:, 0] * (1 if v[:, 0].sum() >= 0 else -1)
        corr = np.array([np.corrcoef(d[:, k], d @ u)[0, 1] for k in range(A.shape[2])])
        return w / w.sum(), u, corr, d

    out("\nPrincipal components of the per-step price-column changes (rows where any column changed):")
    pcs = {}
    for inst, A in [("i0", ip0), ("i1", ip1)]:
        ev, u, corr, d = pca_diffs(A)
        pcs[inst] = u
        out(f"  {inst}: variance share PC1 {ev[0]:.3f}, PC2 {ev[1]:.3f}, PC3 {ev[2]:.3f}; effective rank "
            f"{1 / (ev ** 2).sum():.2f}")
        out(f"    corr of each column's change with PC1, bid p0-p10: " + " ".join(f"{c:+.2f}" for c in corr[:11]))
        out(f"    ask p11-p21:                                    " + " ".join(f"{c:+.2f}" for c in corr[11:]))
    # tick grid: per sequence and column, nonzero |changes| / their smallest value
    def grid_score(A):
        res = []
        for s in range(A.shape[0]):
            for k in range(A.shape[2]):
                d = np.abs(np.diff(A[s, :, k]))
                d = d[d > 1e-6]
                if len(d) >= 4:
                    r = d / d.min()
                    res.append(np.mean(np.abs(r - np.round(r)) < 0.05))
        return np.mean(res), len(res)
    g0, n0 = grid_score(ip0)
    out(f"  Tick-grid test: within a sequence and column, the fraction of nonzero |changes| that are integer multiples "
        f"(+-5%) of the smallest one: i0 {g0:.2f} ({n0} sequence-columns); a raw or affinely scaled tick price "
        f"gives about 1.0, a chance level for smooth values is about 0.1 + the share of repeats of the smallest step")
    k_uniq = np.mean([len(np.unique(np.round(np.abs(np.diff(ip0[s, :, 0]))[np.abs(np.diff(ip0[s, :, 0])) > 1e-6], 4)))
                      for s in range(N_SEQ)])
    out(f"    i0_p0: on average {k_uniq:.1f} distinct nonzero |change| values per sequence")
    mean0 = np.array([ip0[:, :, k].mean() for k in range(22)])
    mean1 = np.array([ip1[:, :, k].mean() for k in range(22)])
    lvl = np.array([np.corrcoef(ip0[:, :, k].ravel(), ip1[:, :, k].ravel())[0, 1] for k in range(22)])
    out(f"  Same-index columns of the two instruments: correlation of the 22 column means i0 vs i1 "
        f"{np.corrcoef(mean0, mean1)[0, 1]:.4f} (max |mean diff| {np.abs(mean0 - mean1).max():.3f}); row-level "
        f"correlation of i0_pk with i1_pk: median {np.median(lvl):.2f}, range {lvl.min():.2f} to {lvl.max():.2f}")
    f0 = (np.diff(ip0, axis=1) @ pcs["i0"])
    f1 = (np.diff(ip1, axis=1) @ pcs["i1"])
    both = (f0 != 0) & (f1 != 0)
    out(f"  i0 vs i1 mid-move proxies (PC1 of changes): corr on rows where both move {np.corrcoef(f0[both], f1[both])[0, 1]:+.3f} "
        f"({both.sum()} rows); 20-step sums: {np.corrcoef(np.add.reduceat(f0, np.arange(0, f0.shape[1], 20), axis=1).ravel(), np.add.reduceat(f1, np.arange(0, f1.shape[1], 20), axis=1).ravel())[0, 1]:+.3f}")

    out("\nVolume-like columns: bid/ask and price relation")
    V0 = np.stack([col[f"i0_v{k}"] for k in range(22)], 2)
    dv0 = np.diff(V0, axis=1).reshape(-1, 22)
    cvol = np.corrcoef(V0.reshape(-1, 22).T)
    out(f"  i0 volume level correlations: mean within bid {cvol[:11, :11][np.triu_indices(11, 1)].mean():+.2f}, "
        f"within ask {cvol[11:, 11:][np.triu_indices(11, 1)].mean():+.2f}, bid vs ask {cvol[:11, 11:].mean():+.2f}; "
        f"same-index bid k vs ask k+11: " + " ".join(f"{cvol[k, k + 11]:+.2f}" for k in range(11)))
    out(f"  i0 volume changes vs the mid-move proxy (same step): " + " ".join(
        f"{np.corrcoef(dv0[:, k], f0.ravel())[0, 1]:+.2f}" for k in range(22)))

    out("\nTrade columns dp / dv:")
    for inst in ["i0", "i1"]:
        mid = np.concatenate([np.zeros((N_SEQ, 1)), np.cumsum(f0 if inst == "i0" else f1, axis=1)], 1)
        pc1lvl = (ip0 if inst == "i0" else ip1) @ pcs[inst]
        for k in range(4):
            dp, dv = col[f"{inst}_dp{k}"], col[f"{inst}_dv{k}"]
            cdp = np.corrcoef(np.diff(dp, axis=1).ravel(), np.diff(pc1lvl, axis=1).ravel())[0, 1]
            out(f"  {inst}_dp{k}: changes on {np.mean(np.diff(dp, axis=1) != 0):.2f} of steps, with {inst}_dv{k} on "
                f"{np.mean((np.diff(dp, axis=1) != 0) & (np.diff(dv, axis=1) != 0)) / max(np.mean(np.diff(dp, axis=1) != 0), 1e-9):.2f} of those; "
                f"corr of its changes with the book PC1 level change {cdp:+.2f}; dv{k} fill share {np.mean(np.isclose(np.abs(dv), FILL)):.3f}")

    out("\nAdditional columns a0-a7:")
    for k in range(8):
        a = col[f"a{k}"]
        ok = ~np.isclose(np.abs(a), FILL)
        ac1 = np.corrcoef(a[:, 1:][ok[:, 1:] & ok[:, :-1]], a[:, :-1][ok[:, 1:] & ok[:, :-1]])[0, 1]
        cy = [wcorr(a[ok], np.clip(Y[:, :, t], -2, 2)[ok], np.abs(np.clip(Y[:, :, t], -2, 2))[ok] + 1e-12) for t in range(2)]
        cm = np.corrcoef(np.diff(a, axis=1)[ok[:, 1:] & ok[:, :-1]], f0[ok[:, 1:] & ok[:, :-1]])[0, 1]
        out(f"  a{k}: fill {1 - ok.mean():.3f}, lag-1 autocorr {ac1:+.3f}, distinct values {len(np.unique(np.round(a, 4)))}, "
            f"corr with t0 {cy[0]:+.3f} / t1 {cy[1]:+.3f} (|y|-weighted), its change vs i0 mid move {cm:+.3f}")

    out("\nThe 168 model inputs (train_v5m.build_inputs): 0-43 the 44 price-like columns (i0_p0-21, i1_p0-21), 44-87 their "
        "one-step differences, 88-155 the 68 other columns (i0_v0-21, i0_dp0-3, i0_dv0-3, i1_v0-21, i1_dp0-3, i1_dv0-3, "
        "a0-a7), all standardised with fixed train means / stds, then 156-167 twelve flags |x| == fill for a0, a5, a6, "
        "a7, i0_dv0-3, i1_dv0-3.")

    # ---------------------------------------------------------------- 2./3. proxies
    out("\n=== 2./3. Order-book quantities: proxies, target correlations, overlap with the 168 inputs ===")
    T = N_ROW
    mid = {}
    for inst, A, u in [("i0", ip0, pcs["i0"]), ("i1", ip1, pcs["i1"])]:
        mid[inst] = np.concatenate([np.zeros((N_SEQ, 1)), np.cumsum(np.diff(A, axis=1) @ u, axis=1)], 1)
    # spread proxy: slope-normalised columns (each column's change / its PC1 slope moves 1:1 with the mid)
    d0 = np.diff(ip0, axis=1).reshape(-1, 22)
    fm = (d0 @ pcs["i0"])
    slope = (d0 * fm[:, None]).sum(0) / (fm * fm).sum()
    use = np.abs(slope) >= 0.05 * np.abs(slope).max()
    norm = ip0 / slope
    spread = norm[:, :, 11:][:, :, use[11:]].mean(2) - norm[:, :, :11][:, :, use[:11]].mean(2)
    # volumes in rank units: undo the normal quantile transform's last step (Phi), giving the column's global quantile
    Phi = np.vectorize(NormalDist().cdf)
    U0 = Phi(V0)
    V1 = np.stack([col[f"i1_v{k}"] for k in range(22)], 2)
    # level-1 pair: the bid/ask volume pair (k, k + 11) whose rank imbalance best predicts the next-20-step i0 mid move
    fut = mid["i0"][:, AHEAD:] - mid["i0"][:, :-AHEAD]
    pair_r = []
    for k in range(11):
        imb = (U0[:, :, k] - U0[:, :, k + 11]) / (U0[:, :, k] + U0[:, :, k + 11] + 1e-9)
        pair_r.append(np.corrcoef(imb[:, :-AHEAD].ravel(), fut.ravel())[0, 1])
    kbest = int(np.nanargmax(np.abs(pair_r)))
    imb1 = (U0[:, :, kbest] - U0[:, :, kbest + 11]) / (U0[:, :, kbest] + U0[:, :, kbest + 11] + 1e-9)
    depth = (U0[:, :, :11].sum(2) - U0[:, :, 11:].sum(2)) / (U0.sum(2) + 1e-9)
    micro_off = spread * imb1 / 2
    micro = mid["i0"] + micro_off
    beta = (np.diff(mid["i0"], axis=1) * np.diff(mid["i1"], axis=1)).sum() / (np.diff(mid["i1"], axis=1) ** 2).sum()
    xdiff = mid["i0"] - beta * mid["i1"]
    out(f"Proxies (all causal; mid in PC1 units of the instrument's price-column changes, relative to step 0):")
    out(f"  mid0, mid1          cumulative PC1 of the price-column changes (level relative to step 0); also dmid0_20 = "
        f"mid0[t] - mid0[t-20]")
    out(f"  spread0             mean of slope-normalised ask columns minus bid columns ({use.sum()} of 22 columns with "
        f"|slope| >= 5% of the largest); a linear combination of the current price inputs")
    out(f"  imb_L1              rank imbalance (Phi(v_bid) - Phi(v_ask)) / sum of the pair i0_v{kbest} / i0_v{kbest + 11}, "
        f"the pair whose imbalance best predicts the next-20-step mid move (corr {pair_r[kbest]:+.3f}; all pairs "
        + " ".join(f"{r:+.2f}" for r in pair_r) + ")")
    out(f"  imb_depth           (sum Phi(bid volumes) - sum Phi(ask volumes)) / sum of all 22, rank units")
    out(f"  micro_off, micro0   microprice offset spread0 * imb_L1 / 2 and micro0 = mid0 + offset")
    out(f"  xmid                mid0 - beta * mid1, beta {beta:.3f} (regression of i0 on i1 PC1 changes)")

    feats = {"mid0": mid["i0"], "mid1": mid["i1"], "dmid0_20": np.concatenate([np.full((N_SEQ, AHEAD), np.nan),
                                                                                   mid["i0"][:, AHEAD:] - mid["i0"][:, :-AHEAD]], 1),
             "spread0": spread, "imb_L1": imb1, "imb_depth": depth, "micro_off": micro_off, "micro0": micro,
             "xmid": xdiff}
    # references: one existing input and the inputs' best single column
    yc = np.clip(Y, -2, 2)
    seq_id = np.repeat(np.arange(N_SEQ), T).reshape(N_SEQ, T)
    hist = np.full_like(X168, np.nan)
    cs = np.cumsum(X168, axis=1)
    hist[:, HIST:] = (cs[:, HIST - 1:-1] - np.concatenate([np.zeros((N_SEQ, 1, 168)), cs[:, :-HIST - 1]], 1)) / HIST
    rows_cur = np.zeros((N_SEQ, T), bool)
    rows_cur[:, HIST:] = True  # rows with a 20-step history (for R^2 and the incremental test)
    rows_fut = np.zeros((N_SEQ, T), bool)
    rows_fut[:, HIST:T - AHEAD] = True
    dY = np.full_like(yc, np.nan)
    dY[:, :-AHEAD] = yc[:, AHEAD:] - yc[:, :-AHEAD]
    Xc = X168[rows_cur]
    Xh = np.c_[Xc, hist[rows_cur]]
    g = seq_id[rows_cur]

    # base ridge from the 168 inputs to each target (|y|-weighted), out of fold
    base_pred = {}
    for t in range(2):
        y = yc[:, :, t][rows_cur]
        _, base_pred[t] = cv_r2(Xc, y, g, w=np.abs(y) + 1e-6)
    base_corr = [wcorr(base_pred[t], yc[:, :, t][rows_cur], np.abs(yc[:, :, t][rows_cur]) + 1e-12) for t in range(2)]
    best_in = max(range(168), key=lambda j: np.nanmean([abs(wcorr(X168[:, :, j][rows_cur], yc[:, :, t][rows_cur], np.abs(yc[:, :, t][rows_cur]) + 1e-12)) for t in range(2)]))
    out(f"\nReference: |y|-weighted CV correlation of a ridge from the 168 current inputs to the target on these rows "
        f"(steps {HIST}-{T - 1}): t0 {base_corr[0]:.3f}, t1 {base_corr[1]:.3f}; best single input column {in_names[best_in]}")
    feats["ref:" + in_names[best_in]] = X168[:, :, best_in]

    out(f"\n  {'quantity':<22} {'corr y now (t0/t1, se)':>26} {'corr y[t+20]-y[t] (t0/t1, se)':>32} "
        f"{'R2 from 168':>11} {'+hist':>6} {'gain in ridge corr t0/t1':>25}")
    summary = {}
    for name, f in feats.items():
        def c_now(sel, t):
            m = rows_cur[sel]
            return wcorr(f[sel][m], yc[sel][:, :, t][m], np.abs(yc[sel][:, :, t][m]) + 1e-12)

        def c_fut(sel, t):
            m = rows_fut[sel]
            return wcorr(f[sel][m], dY[sel][:, :, t][m], np.abs(yc[sel][:, :, t][m]) + 1e-12)
        allseq = np.arange(N_SEQ)
        cn = [c_now(allseq, t) for t in range(2)]
        cf = [c_fut(allseq, t) for t in range(2)]
        sn = boot_se(lambda s: np.mean([c_now(s, t) for t in range(2)]), N_SEQ, rng)
        sf = boot_se(lambda s: np.mean([c_fut(s, t) for t in range(2)]), N_SEQ, rng)
        fv = f[rows_cur]
        r2, _ = cv_r2(Xc, fv, g)
        r2h, _ = cv_r2(Xh, fv, g)
        gain = []
        for t in range(2):
            y = yc[:, :, t][rows_cur]
            _, p = cv_r2(np.c_[Xc, fv], y, g, w=np.abs(y) + 1e-6)
            gain.append(wcorr(p, y, np.abs(y) + 1e-12) - base_corr[t])
        summary[name] = (cn, cf, r2, r2h, gain)
        out(f"  {name:<22} {cn[0]:+.3f} / {cn[1]:+.3f} ({sn:.3f}) {cf[0]:+.3f} / {cf[1]:+.3f} ({sf:.3f})        "
            f"{r2:>11.3f} {r2h:>6.3f} {gain[0]:>+12.4f} / {gain[1]:+.4f}")
    out("  se: bootstrap over the 200 sequences of the mean over t0 / t1. R2: 5-fold CV by sequence, rows "
        f"{HIST}-{T - 1}; '+hist' adds the inputs' mean over the previous {HIST} steps. Gain: |y|-weighted CV correlation "
        "of the ridge with the quantity added minus without (same folds).")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
