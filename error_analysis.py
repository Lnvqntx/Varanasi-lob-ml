"""Where does the v7+v9 ensemble lose weighted Pearson? Error analysis on the valid scored rows (CPU only).

Predictions: ~/work/calib_v7v9_pred_t0.npy and _t1.npy (calib_check.py: fp32 average of v7 and v9, saved as
float16, shape (1873, 19901) for steps 99-19,999). Rows: is_scored & step >= 99 of all 1,873 valid sequences.
Targets and predictions are clipped to [-2, 2], weights w = |clipped y|, as in the metric.

For each bin: rows, weighted Pearson per target and their mean, weight share (sum of w in the bin over the
total), mean |y|, and the share of the weighted squared error w * (y - b p)^2, with b per target fitted
by weighted least squares (weights w, no intercept) on all scored rows. A bin with SSE share well above
its weight share is where the loss concentrates. Correlation inside a bin is over a narrower range of
y than the full set, so it is lower than the global score by construction for |y| bins.
  (a) step range, bins of 2,000 steps (the first is 99-1,999)
  (b) |y| decile and sign of y, each target binned on its own y (y = 0 separate, tied decile edges merged)
  (d) for a proxy whose tercile cut points coincide, bins below / at / above its most common value instead
  (c) t0 versus t1
  (d) terciles of i0 volatility (vol0), i0 spread proxy (sp0) and i0 order-book imbalance proxy (imb0,
      signed and absolute); see probe_features.py for how these proxies are built from the anonymized columns
  (e) per-sequence: the worst 10% of sequences by per-sequence weighted Pearson (mean of t0 and t1) and the
      top 10% by weighted squared error, with their share of weight and of squared error

  OMP_NUM_THREADS=4 nice -n 10 nohup python error_analysis.py > ~/work/error_analysis.log 2>&1 &
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time

import numpy as np

import probe_features as pf

CACHE = "/home/jovyan/work/cache"
PRED = [f"/home/jovyan/work/calib_v7v9_pred_t{j}.npy" for j in (0, 1)]
CHECK6 = "/home/jovyan/work/ensemble_check6_stats.npz"
RESULTS_OUT = "/home/jovyan/work/error_analysis_results.txt"
WARMUP = 99
CLIP = 2.0


def collect():
    """Scored rows of all valid sequences: y, p (N,2) float64 clipped; step, seq; proxies."""
    fac = pf.fit_factors()
    P = np.load(f"{CACHE}/valid_prices.npy", mmap_mode="r")
    F = np.load(f"{CACHE}/valid_features.npy", mmap_mode="r")
    Y = np.load(f"{CACHE}/valid_targets.npy", mmap_mode="r")
    M = np.load(f"{CACHE}/valid_is_scored.npy", mmap_mode="r")
    pr = [np.load(p, mmap_mode="r") for p in PRED]
    n, T = M.shape
    assert pr[0].shape == (n, T - WARMUP)
    cols = {k: [] for k in ["y", "p", "step", "seq", "vol0", "sp0", "imb0"]}
    t = time.time()
    for j in range(n):
        m = np.asarray(M[j]).copy()
        m[:WARMUP] = False
        steps = np.flatnonzero(m)
        s = pf.seq_series(np.asarray(P[j]), np.asarray(F[j]), fac)
        cols["y"].append(np.asarray(Y[j])[steps])
        cols["p"].append(np.stack([np.asarray(pr[k][j])[steps - WARMUP] for k in (0, 1)], 1))
        cols["step"].append(steps)
        cols["seq"].append(np.full(len(steps), j))
        for k in ["vol0", "sp0", "imb0"]:
            cols[k].append(s[k][steps])
        if (j + 1) % 200 == 0:
            print(f"  {j + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["y"] = np.clip(d["y"].astype(np.float64), -CLIP, CLIP)
    d["p"] = np.clip(d["p"].astype(np.float64), -CLIP, CLIP)
    return d, fac


def wp(y, p):
    w = np.abs(y)
    sw = w.sum()
    my, mp = (w * y).sum() / sw, (w * p).sum() / sw
    cov = (w * (y - my) * (p - mp)).sum()
    return cov / np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum())


def main():
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    d, fac = collect()
    y, p = d["y"], d["p"]
    N = len(y)
    w = np.abs(y)
    b = (w * y * p).sum(0) / (w * p * p).sum(0)
    se = w * (y - b * p) ** 2
    W, SE = w.sum(0), se.sum(0)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    g = [wp(y[:, k], p[:, k]) for k in (0, 1)]
    ref = None
    if os.path.exists(CHECK6):
        from spread_v2 import wp_from_stats
        c6 = np.load(CHECK6)
        ref = wp_from_stats(c6["stats"][list(c6["subsets"]).index("v7+v9")].sum(0))[0, 2]
    out("Error analysis of the v7+v9 ensemble (fp32 average saved as float16) on valid is_scored rows, steps >= 99")
    out(f"rows {N:,}, sequences {len(np.unique(d['seq']))}; global weighted Pearson t0 {g[0]:.4f}, t1 {g[1]:.4f}, "
        f"mean {np.mean(g):.6f}" + (f" (ensemble_check6 fp32: {ref:.6f}, diff {np.mean(g) - ref:+.1e})" if ref else ""))
    out(f"WLS scale b (y ~ b p, weights |y|): t0 {b[0]:.4f}, t1 {b[1]:.4f}")
    out("Columns: rows; weighted Pearson t0, t1, mean; weight share (sum |y| in bin / total, mean of targets); "
        "mean |y|; share of weighted squared error w (y - b p)^2; SSE/weight = SSE share / weight share")

    hdr = f"  {'bin':<22} {'rows':>9} {'wp t0':>7} {'wp t1':>7} {'wp':>7} {'w share':>8} {'mean|y|':>8} {'SSE share':>9} {'SSE/w':>6}"

    def row(name, sel):
        """sel: boolean (N,) for both targets, or (N,2) per target."""
        sel = np.broadcast_to(sel.reshape(N, -1), (N, 2))
        r = [wp(y[sel[:, k], k], p[sel[:, k], k]) for k in (0, 1)]
        ws = np.mean([w[sel[:, k], k].sum() / W[k] for k in (0, 1)])
        ss = np.mean([se[sel[:, k], k].sum() / SE[k] for k in (0, 1)])
        my = np.mean([w[sel[:, k], k].mean() for k in (0, 1)])
        rows = int(sel[:, 0].sum()) if (sel[:, 0] == sel[:, 1]).all() else int(sel.sum() / 2)
        out(f"  {name:<22} {rows:>9,} {r[0]:>7.4f} {r[1]:>7.4f} {np.mean(r):>7.4f} {ws:>8.3f} {my:>8.3f} "
            f"{ss:>9.3f} {ss / ws:>6.2f}")

    hdr1 = f"  {'bin':<22} {'rows':>9} {'row share':>9} {'wp':>7} {'w share':>8} {'mean|y|':>8} {'SSE share':>9} {'SSE/w':>6}"

    def row1(k, name, sel):
        ws, ss = w[sel, k].sum() / W[k], se[sel, k].sum() / SE[k]
        r = wp(y[sel, k], p[sel, k]) if ws > 0 else float("nan")
        out(f"  {name:<22} {int(sel.sum()):>9,} {sel.mean():>9.3f} {r:>7.4f} {ws:>8.3f} {w[sel, k].mean():>8.3f} "
            f"{ss:>9.3f} {(ss / ws if ws > 0 else float('nan')):>6.2f}")

    out("\n(a) by step range")
    out(hdr)
    edges = [WARMUP] + list(range(2000, 20001, 2000))
    for lo, hi in zip(edges[:-1], edges[1:]):
        row(f"steps {lo}-{hi - 1}", (d["step"] >= lo) & (d["step"] < hi))

    out("\n(b) by |y| decile, per target. |y| is discrete (exact zeros, a point mass near 0.66, and many rows at")
    out("    the clip value 2), so y = 0 is its own bin and decile edges that coincide are merged")
    for k in (0, 1):
        out(f"  t{k}")
        out(hdr1)
        z = w[:, k] == 0
        row1(k, "y = 0 (weight 0)", z)
        e = np.unique(np.quantile(w[~z, k], np.linspace(0, 1, 11)))
        bi = np.clip(np.searchsorted(e[1:-1], w[:, k], side="right"), 0, len(e) - 2)
        for i in range(len(e) - 1):
            row1(k, f"|y| {e[i]:.3f}-{e[i + 1]:.3f}", ~z & (bi == i))
    out("  by sign of y")
    out(hdr)
    row("y > 0", y > 0)
    row("y < 0", y < 0)

    out("\n(c) t0 versus t1")
    out(f"  {'target':<8} {'wp':>7} {'sum |y|':>12} {'mean |y|':>9} {'share of total SSE (both targets)':>34}")
    for k in (0, 1):
        out(f"  t{k:<7} {g[k]:>7.4f} {W[k]:>12,.0f} {w[:, k].mean():>9.4f} {SE[k] / SE.sum():>34.3f}")
    out(f"  correlation of t0 and t1 (|y|-weighted by t0): {wp(y[:, 0], y[:, 1]):.4f}; of the two predictions: "
        f"{np.corrcoef(p[:, 0], p[:, 1])[0, 1]:.4f}")

    out("\n(d) by terciles of causal proxies (probe_features.py; cut points over the scored rows)")
    out(f"  proxy columns: vol0 = rolling 100-step std of f0 (i0 mid move = first principal component of the 22")
    out(f"  i0_p* diffs); sp0 = ask-minus-bid mean of the {int((np.abs(fac['slope0']) >= pf.MIN_LOAD).sum())} i0_p* "
        f"columns with |loading| >= {pf.MIN_LOAD}, each divided by its loading; imb0 = ridge combination of")
    out(f"  i0_v0-v21 predicting the next {pf.FUT}-step f0 move (corr {fac['imb_r0']:.3f} on the {pf.FIT_SEQS} "
        f"fit train sequences). Anonymized columns, so these are proxies, not exact quantities.")
    for name, x in [("vol0", d["vol0"]), ("sp0", d["sp0"]), ("imb0", d["imb0"]), ("|imb0|", np.abs(d["imb0"]))]:
        cut = np.quantile(x, [1 / 3, 2 / 3])
        if cut[0] == cut[1]:
            vals, cnt = np.unique(x, return_counts=True)
            mode = vals[cnt.argmax()]
            out(f"  {name}: terciles collapse ({cnt.max() / len(x):.1%} of rows at {mode:.4g}, likely the tightest "
                f"common spread); bins below / at / above that value")
            out(hdr)
            row(f"{name} below mode", x < mode)
            row(f"{name} at mode", x == mode)
            row(f"{name} above mode", x > mode)
            continue
        out(f"  {name}: cut points {cut[0]:.4g}, {cut[1]:.4g}")
        out(hdr)
        for i, lab in enumerate(["low", "mid", "high"]):
            lo = -np.inf if i == 0 else cut[i - 1]
            hi = np.inf if i == 2 else cut[i]
            row(f"{name} {lab}", (x > lo) & (x <= hi) if i else (x <= hi))

    out("\n(e) per sequence")
    seq = d["seq"]
    ns = seq.max() + 1
    sw = np.stack([np.bincount(seq, w[:, k], ns) for k in (0, 1)], 1)
    sse = np.stack([np.bincount(seq, se[:, k], ns) for k in (0, 1)], 1)
    order = np.argsort(seq, kind="stable")
    bounds = np.searchsorted(seq[order], np.arange(ns + 1))
    swp = np.array([np.mean([wp(y[order[bounds[s]:bounds[s + 1]], k], p[order[bounds[s]:bounds[s + 1]], k])
                             for k in (0, 1)]) for s in range(ns)])
    k10 = int(round(0.1 * ns))
    out(f"  per-sequence weighted Pearson: median {np.median(swp):.4f}, 10th pct {np.percentile(swp, 10):.4f}, "
        f"90th pct {np.percentile(swp, 90):.4f}")
    for lab, idx in [("worst 10% by per-sequence wp", np.argsort(swp)[:k10]),
                     ("top 10% by squared error", np.argsort(-sse.sum(1))[:k10]),
                     ("top 10% by weight (sum |y|)", np.argsort(-sw.sum(1))[:k10])]:
        sel = np.isin(seq, idx)
        rest = [wp(y[~sel, k], p[~sel, k]) for k in (0, 1)]
        out(f"  {lab} ({k10} seqs): weight share {sw[idx].sum() / sw.sum():.3f}, SSE share "
            f"{sse[idx].sum() / sse.sum():.3f}, their wp {np.mean([wp(y[sel, k], p[sel, k]) for k in (0, 1)]):.4f}, "
            f"wp of the other 90% {np.mean(rest):.4f}, mean |y| {w[sel].mean():.3f}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
