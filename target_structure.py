"""How are t0 and t1 built? Target structure on N_SEQ train sequences (CPU only, numpy, <= 4 threads).

Train caches, full sequences (all 20,000 steps, no mask). Sequences: a seeded random choice of N_SEQ.
  (a) distinct values of t0 and t1, the top 15 values with frequencies, the joint frequency table over
      the top 8 values of each
  (b) autocorrelation of each target within a sequence at LAGS (pooled over sequences), plus the share
      of exact zeros near the end of a sequence (a forward window runs out there)
  (c) correlation of each target at step t with the FUTURE change price[t + h] - price[t] of each of the
      22 i0 price columns and of f0 (i0 mid proxy: projection on the first principal component of the 22
      i0 price diffs, probe_features.py), h in HORIZONS; best (column, h) pairs. Also the correlation
      ratio eta^2 of each target on the binned best future change (catches a non-linear map such as a
      sign or a rounding)
  (d) linear regression of each target on future i0 price changes at all horizons: (1) f0 only,
      (2) the best column per horizon, (3) all 22 columns + f0 at all horizons; fitted on 80% of the
      sequences, R^2 on the other 20% (whole sequences), weights per horizon for (1) and (2). Rows with
      t + max(h) < T only, so every horizon exists
  (e) t0 versus t1: R^2 of t1 on t0, and whether t0 + t1 or t0 - t1 is more discrete or has lower variance
  (f) units: the smallest nonzero |target| of each sequence, whether every target is an integer multiple of
      it, the targets in those units, and the linear R^2 of t0, t1, t0 + t1, t0 - t1 from the current row

  OMP_NUM_THREADS=4 nice -n 10 nohup python target_structure.py > ~/work/target_structure.log 2>&1 &
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import sys
import time

import numpy as np

import probe_features as pf

CACHE = "/home/jovyan/work/cache"
RESULTS_OUT = "/home/jovyan/work/target_structure_results.txt"
N_SEQ = 1000
SEED = 0
LAGS = [1, 2, 5, 10, 20, 50, 100, 200]
HORIZONS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
N_BINS = 50
COLS = [f"i0_p{k}" for k in range(22)] + ["f0"]


def moments():
    return {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}


def corr_from(n, sx, sy, sxx, syy, sxy):
    return (sxy / n - sx / n * sy / n) / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))


def main():
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    fac = pf.fit_factors()
    u0 = fac["u0"]
    P = np.load(f"{CACHE}/train_prices.npy", mmap_mode="r")
    Yall = np.load(f"{CACHE}/train_targets.npy", mmap_mode="r")
    n_all, T = Yall.shape[:2]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_all, N_SEQ, replace=False))
    is_test = np.arange(N_SEQ) % 5 == 0
    H, C, L = len(HORIZONS), len(COLS), len(LAGS)
    hmax = max(HORIZONS)
    # (c) pooled moments, shape (H, C, 2)
    cn = np.zeros((H, 1, 1))
    csx, csxx = np.zeros((H, C, 1)), np.zeros((H, C, 1))
    csy, csyy = np.zeros((H, 1, 2)), np.zeros((H, 1, 2))
    csxy = np.zeros((H, C, 2))
    # (b) autocorrelation moments, shape (L, 2)
    an, asx, asy, asxx, asyy, asxy = (np.zeros((L, 2)) for _ in range(6))
    # (d) all-horizon design: H x C future changes + intercept, accumulated for fit / test sequences
    nfeat = H * C + 1
    XtX = np.zeros((2, nfeat, nfeat))
    Xty = np.zeros((2, nfeat, 2))
    yty = np.zeros((2, 2))
    ysum = np.zeros((2, 2))
    nrow = np.zeros(2)
    Ys = np.empty((N_SEQ, T, 2), dtype=np.float32)
    first = {}  # first 50 sequences' future changes, for eta^2 bin edges
    t = time.time()
    for i, j in enumerate(idx):
        p = np.asarray(P[j][:, :22], dtype=np.float64)
        p = np.column_stack([p, p @ u0])  # (T, 23)
        y = np.asarray(Yall[j], dtype=np.float64)
        Ys[i] = y
        for li, lag in enumerate(LAGS):
            a, b = y[:-lag], y[lag:]
            an[li] += len(a)
            asx[li] += a.sum(0)
            asy[li] += b.sum(0)
            asxx[li] += (a * a).sum(0)
            asyy[li] += (b * b).sum(0)
            asxy[li] += (a * b).sum(0)
        D = []
        for hi, h in enumerate(HORIZONS):
            d = p[h:] - p[:-h]
            yy = y[:-h]
            cn[hi] += len(d)
            csx[hi, :, 0] += d.sum(0)
            csxx[hi, :, 0] += (d * d).sum(0)
            csy[hi, 0] += yy.sum(0)
            csyy[hi, 0] += (yy * yy).sum(0)
            csxy[hi] += d.T @ yy
            D.append(d[:T - hmax])
            if i < 50:
                first.setdefault(h, []).append(d[:T - hmax])
        X = np.column_stack(D + [np.ones(T - hmax)])
        yy = y[:T - hmax]
        k = int(is_test[i])
        XtX[k] += X.T @ X
        Xty[k] += X.T @ yy
        yty[k] += (yy * yy).sum(0)
        ysum[k] += yy.sum(0)
        nrow[k] += len(yy)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_SEQ} seqs, {time.time() - t:.0f}s", flush=True)

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out(f"Target structure on {N_SEQ} train sequences (seeded random choice, seed {SEED}), all {T:,} steps each")
    out(f"f0 = projection of the 22 i0 price columns on their first principal component of diffs (probe_features.py)")

    # (a)
    out("\n(a) Distinct values")
    for k in (0, 1):
        v, c = np.unique(Ys[:, :, k], return_counts=True)
        o = np.argsort(-c)
        out(f"  t{k}: {len(v):,} distinct values over {c.sum():,} rows; top 15 cover {c[o[:15]].sum() / c.sum():.1%}; "
            f"range {v[0]:.4f} to {v[-1]:.4f}")
        out("    " + ", ".join(f"{v[q]:.6g} ({c[q] / c.sum():.2%})" for q in o[:15]))
    top = []
    for k in (0, 1):
        v, c = np.unique(Ys[:, :, k], return_counts=True)
        top.append(v[np.argsort(-c)[:8]])
    a0 = np.full(Ys.shape[:2], 8)
    a1 = np.full(Ys.shape[:2], 8)
    for q in range(8):
        a0[Ys[:, :, 0] == top[0][q]] = q
        a1[Ys[:, :, 1] == top[1][q]] = q
    J = np.zeros((9, 9))
    np.add.at(J, (a0.ravel(), a1.ravel()), 1)
    J /= J.sum()
    out("  joint frequency (% of rows), rows t0 top 8 values + other, columns t1 top 8 values + other")
    labels1 = [f"{x:.4g}" for x in top[1]] + ["other"]
    corner = "t0 \\ t1"
    out("    " + f"{corner:>9}" + "".join(f"{s:>8}" for s in labels1))
    for q in range(9):
        lab = f"{top[0][q]:.4g}" if q < 8 else "other"
        out("    " + f"{lab:>9}" + "".join(f"{100 * J[q, r]:>8.2f}" for r in range(9)))

    # (b)
    out("\n(b) Autocorrelation within a sequence (pooled)")
    ac = corr_from(an, asx, asy, asxx, asyy, asxy)
    out(f"  {'lag':>5} {'t0':>8} {'t1':>8}")
    for li, lag in enumerate(LAGS):
        out(f"  {lag:>5} {ac[li, 0]:>8.4f} {ac[li, 1]:>8.4f}")
    out("  share of exact zeros by distance from the end of the sequence (a forward window runs out there)")
    for lo, hi in [(0, 1), (1, 10), (10, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 20000)]:
        seg = Ys[:, T - hi:T - lo]
        out(f"    last steps {lo:>5}-{hi - 1:>5} from the end: t0 {np.mean(seg[:, :, 0] == 0):.3f}, "
            f"t1 {np.mean(seg[:, :, 1] == 0):.3f}")
    out(f"    first 100 steps: t0 {np.mean(Ys[:, :100, 0] == 0):.3f}, t1 {np.mean(Ys[:, :100, 1] == 0):.3f}")

    # (c)
    out("\n(c) Correlation of the target at t with the future change price[t + h] - price[t]")
    R = corr_from(cn, csx, csy, csxx, csyy, csxy)  # (H, C, 2)
    for k in (0, 1):
        out(f"  t{k}: per horizon, f0 and the best i0 price column (by |corr|)")
        out(f"    {'h':>5} {'f0':>8} {'best col':>9} {'corr':>8}")
        for hi, h in enumerate(HORIZONS):
            b = np.argmax(np.abs(R[hi, :22, k]))
            out(f"    {h:>5} {R[hi, 22, k]:>8.4f} {COLS[b]:>9} {R[hi, b, k]:>8.4f}")
        flat = np.argsort(-np.abs(R[:, :, k]).ravel())[:10]
        out("    top 10 (column, h): " + "; ".join(
            f"{COLS[f % C]} h={HORIZONS[f // C]} {R[f // C, f % C, k]:+.4f}" for f in flat))
        out("    full |corr| table, rows = columns, columns = horizons")
        out("      " + f"{'col':>6}" + "".join(f"{h:>7}" for h in HORIZONS))
        for c in range(C):
            out("      " + f"{COLS[c]:>6}" + "".join(f"{R[hi, c, k]:>7.3f}" for hi in range(H)))

    # eta^2 of each target on the binned best (column, h) future change (first 50 sequences; nonlinear fit)
    out("  correlation ratio eta^2 of the target on the binned best future change (first 50 sequences,")
    out(f"  {N_BINS} quantile bins; eta^2 >= corr^2, a large gap means a non-linear map)")
    for k in (0, 1):
        hb, cb = np.unravel_index(np.argmax(np.abs(R[:, :, k])), (H, C))
        x = np.concatenate([d[:, cb] for d in first[HORIZONS[hb]]])
        yk = Ys[:50, :T - hmax, k].ravel().astype(np.float64)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1))[1:-1])
        bi = np.searchsorted(edges, x)
        cnt = np.bincount(bi)
        mb = np.bincount(bi, yk) / np.maximum(cnt, 1)
        eta2 = (cnt * (mb - yk.mean()) ** 2).sum() / ((yk - yk.mean()) ** 2).sum()
        out(f"    t{k}: {COLS[cb]} h={HORIZONS[hb]}: corr^2 {np.corrcoef(x, yk)[0, 1] ** 2:.4f}, eta^2 {eta2:.4f} "
            f"({len(edges) + 1} bins)")

    # (d)
    out(f"\n(d) Linear regression of each target on future i0 price changes at all horizons {HORIZONS}")
    out(f"  rows with t + {hmax} < T; fitted on {int((~is_test).sum())} sequences, R^2 on the other "
        f"{int(is_test.sum())} (whole sequences); in-sample R^2 in brackets")
    fidx = lambda hi, c: hi * C + c

    def fit(cols_idx, k):
        ci = list(cols_idx) + [nfeat - 1]
        A = XtX[0][np.ix_(ci, ci)]
        beta = np.linalg.solve(A + 1e-9 * np.eye(len(ci)), Xty[0][ci, k])
        r2 = []
        for s in (1, 0):
            sse = yty[s, k] - 2 * beta @ Xty[s][ci, k] + beta @ XtX[s][np.ix_(ci, ci)] @ beta
            sst = yty[s, k] - ysum[s, k] ** 2 / nrow[s]
            r2.append(1 - sse / sst)
        return beta, r2

    for k in (0, 1):
        best = [int(np.argmax(np.abs(R[hi, :22, k]))) for hi in range(H)]
        specs = [("f0 at all horizons", [fidx(hi, 22) for hi in range(H)]),
                 ("best column per horizon", [fidx(hi, best[hi]) for hi in range(H)]),
                 ("all 22 columns + f0, all horizons", list(range(H * C)))]
        out(f"  t{k}")
        for name, ci in specs:
            beta, r2 = fit(ci, k)
            out(f"    {name:<36} R^2 {r2[0]:.4f} ({r2[1]:.4f}), {len(ci)} features")
            if len(ci) == H:
                out("      weights per horizon (" + ("f0" if name.startswith("f0") else "column") + "): " + "; ".join(
                    f"h={h}{'' if name.startswith('f0') else ' ' + COLS[best[hi]]} {beta[hi]:+.4g}"
                    for hi, h in enumerate(HORIZONS)))
        # f0 increments between horizons: weights on non-overlapping segments of the future path
        ci = [fidx(hi, 22) for hi in range(H)]
        beta, _ = fit(ci, k)
        seg = np.cumsum(beta[:H][::-1])[::-1]  # weight on the move in (h_{i-1}, h_i]
        out("      implied weight of f0's move over each future segment (t+h_prev, t+h]: " + "; ".join(
            f"{([0] + HORIZONS)[hi]}-{h}: {seg[hi]:+.4g}" for hi, h in enumerate(HORIZONS)))

    # (e)
    out("\n(e) t0 versus t1")
    y0 = Ys[:, :, 0].ravel().astype(np.float64)
    y1 = Ys[:, :, 1].ravel().astype(np.float64)
    r = np.corrcoef(y0, y1)[0, 1]
    out(f"  corr(t0, t1) {r:.4f}; R^2 of t1 on t0 (linear) {r * r:.4f}")
    v0 = np.unique(Ys[:, :, 0], return_inverse=True)[1].ravel()
    cnt = np.bincount(v0)
    mb = np.bincount(v0, y1) / np.maximum(cnt, 1)
    out(f"  eta^2 of t1 given the exact value of t0 (any function of t0): "
        f"{(cnt * (mb - y1.mean()) ** 2).sum() / ((y1 - y1.mean()) ** 2).sum():.4f}")
    out(f"  {'series':<8} {'variance':>9} {'distinct':>9} {'top value share':>16} {'share of zeros':>15}")
    for name, s in [("t0", y0), ("t1", y1), ("t0 + t1", y0 + y1), ("t0 - t1", y0 - y1)]:
        s32 = s.astype(np.float32)
        v, c = np.unique(s32, return_counts=True)
        out(f"  {name:<8} {s.var():>9.4f} {len(v):>9,} {c.max() / c.sum():>16.3f} {np.mean(s32 == 0):>15.3f}")
    # (f)
    out("\n(f) Units: is every target an integer multiple of one per-sequence unit?")
    unit = np.array([np.abs(s[s != 0]).min() if (s != 0).any() else np.nan for s in Ys.reshape(N_SEQ, -1)])
    uv, uc = np.unique(np.round(unit, 4), return_counts=True)
    out("  per-sequence unit = smallest nonzero |t0| or |t1| in the sequence: " + ", ".join(
        f"{a:.4f} ({b} seqs)" for a, b in zip(uv, uc)))
    ticks = Ys.astype(np.float64) / unit[:, None, None]
    kk = np.round(ticks)
    out(f"  rows where t / unit is within 0.001 of an integer: t0 {np.mean(np.abs(ticks[:, :, 0] - kk[:, :, 0]) < 1e-3):.4f}, "
        f"t1 {np.mean(np.abs(ticks[:, :, 1] - kk[:, :, 1]) < 1e-3):.4f}")
    for name, s in [("t0", kk[:, :, 0]), ("t1", kk[:, :, 1]), ("t0 + t1", kk[:, :, 0] + kk[:, :, 1]),
                    ("t0 - t1", kk[:, :, 0] - kk[:, :, 1])]:
        v, c = np.unique(s, return_counts=True)
        o = np.argsort(-c)[:10]
        out(f"  {name:<8} in units: top 10 " + ", ".join(f"{int(v[q]):+d} ({c[q] / c.sum():.1%})" for q in o)
            + f"; share > 0 {np.mean(s > 0):.3f}, < 0 {np.mean(s < 0):.3f}")
    # t0 + t1 and t0 - t1 from the current row's 112 raw features (linear ridge, every 10th row, 70/30 sequences)
    Xs, S, G = [], [], []
    F = np.load(f"{CACHE}/train_features.npy", mmap_mode="r")
    for i, j in enumerate(idx[:100]):
        x = np.column_stack([np.asarray(P[j], dtype=np.float64), np.asarray(F[j], dtype=np.float64)])[::10]
        x[np.abs(x) > 5] = 0.0  # the 5.199 fill values
        Xs.append(np.column_stack([x, np.ones(len(x))]))
        S.append(Ys[i, ::10].astype(np.float64))
        G.append(np.full(len(x), i))
    Xs, S, G = np.concatenate(Xs), np.concatenate(S), np.concatenate(G)
    tr = G % 10 < 7
    out("  linear R^2 from the CURRENT row's 112 features (ridge, first 100 of the sequences, every 10th row, "
        "held-out 30% of them):")
    for name, s in [("t0", S[:, 0]), ("t1", S[:, 1]), ("t0 + t1", S[:, 0] + S[:, 1]), ("t0 - t1", S[:, 0] - S[:, 1])]:
        beta = np.linalg.solve(Xs[tr].T @ Xs[tr] + np.eye(Xs.shape[1]), Xs[tr].T @ s[tr])
        res = s[~tr] - Xs[~tr] @ beta
        out(f"    {name:<8} {1 - res.var() / s[~tr].var():.4f}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
