"""Causal post-filters on the saved v7+v9 valid predictions: do they raise the score?

Read-only, CPU only (2 threads), no feature caches: the saved float16 predictions ~/work/calib_v7v9_pred_t{0,1}.npy
(calib_check.py, shape (1873, 19901) = steps 99..19999) and the small ~/work/cache/valid_targets.npy and
valid_is_scored.npy only. Writes output_filter_check_results.txt.

Metric as official: targets and predictions clipped to [-2, 2], weights |y|, pooled over the scored rows, mean of
t0 and t1 (per-sequence sufficient statistics, so any set of sequences is scored by summing them). The raw score
must reproduce 0.6850 on all 1,873 and 0.6915 on the 469 held-out sequences (index % 4 == 0).

Filters, per target, causal (step t uses steps <= t of the same sequence only; lags before step 99 repeat the
step-99 prediction):
  (a) exponential smoothing p'_t = a p_t + (1 - a) p'_{t-1}, p'_99 = p_99, a in ALPHAS;
  (b) learned causal linear filter: y_t (clipped) on [1, p_t, p_{t-L} for L in LAGS] of the same target and the
      same 12 lags of the other target, weighted least squares with weights |y| + 0.05 on all rows with step >= 99,
      2-fold CV by sequence (even / odd index), scored out of fold;
  (c) the same with 4-fold CV (index % 4; fold 0 is the 469).
Also: step-to-step noise (sd of p_t - p_{t-1} vs y_t - y_{t-1}) and lag-1 autocorrelation of predictions vs targets.

  nice -n 10 python output_filter_check.py
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "2"

import time

import numpy as np
from scipy.signal import lfilter

PRED = [f"/home/jovyan/work/calib_v7v9_pred_t{j}.npy" for j in (0, 1)]
TARGETS = "/home/jovyan/work/cache/valid_targets.npy"
SCORED = "/home/jovyan/work/cache/valid_is_scored.npy"
OUT = "/home/jovyan/work/varanasi-lob-ml/output_filter_check_results.txt"
WARMUP = 99
ALPHAS = [1, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1]
LAGS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
BATCH = 25
REF_ALL, REF_HELD = 0.6850, 0.6915
GAIN_BAR = 0.002

lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def seq_stats(y, p, m):
    """(n_seq, 6) float64 sums [w, wy, wp, wyy, wpp, wyp] on mask m, after clipping; w = |y|."""
    y = np.clip(y.astype(np.float64), -2, 2)
    p = np.clip(p.astype(np.float64), -2, 2)
    w = np.abs(y) * m
    return np.stack([w.sum(1), (w * y).sum(1), (w * p).sum(1), (w * y * y).sum(1), (w * p * p).sum(1),
                     (w * y * p).sum(1)], 1)


def wp(s):
    """Weighted Pearson from summed stats (..., 6)."""
    w, wy, wp_, wyy, wpp, wyp = np.moveaxis(s, -1, 0)
    my, mp = wy / w, wp_ / w
    return (wyp / w - my * mp) / np.sqrt(np.maximum(wyy / w - my * my, 0) * np.maximum(wpp / w - mp * mp, 0))


def score(S, sel):
    """S: list of two (n_seq, 6) arrays (t0, t1) -> [t0, t1, mean] on the sequences sel."""
    r = [wp(S[t][sel].sum(0)) for t in range(2)]
    return np.array([r[0], r[1], (r[0] + r[1]) / 2])


def lagged(P, lags):
    """P (n, T) -> (n, T, len(lags)) causal lags, the first value repeated before the start."""
    out_ = np.empty(P.shape + (len(lags),), np.float64)
    for i, L in enumerate(lags):
        if L == 0:
            out_[:, :, i] = P
        else:
            out_[:, L:, i] = P[:, :-L]
            out_[:, :L, i] = P[:, :1]
    return out_


def design(P0, P1, t):
    """[1, own lags, other lags] for target t: (n, T, 1 + 2 * len(LAGS))."""
    own, oth = (P0, P1) if t == 0 else (P1, P0)
    n, T = own.shape
    return np.concatenate([np.ones((n, T, 1)), lagged(own, LAGS), lagged(oth, LAGS)], 2)


def main():
    t_start = time.time()
    P = [np.load(f, mmap_mode="r") for f in PRED]
    n, T = P[0].shape
    Yall = np.load(TARGETS, mmap_mode="r")
    Mall = np.load(SCORED, mmap_mode="r")
    assert Yall.shape[:2] == (n, WARMUP + T) and Mall.shape == (n, WARMUP + T)
    P = [np.asarray(p, np.float32) for p in P]
    Y = [np.asarray(Yall[:, WARMUP:, t], np.float32) for t in range(2)]
    M = np.asarray(Mall[:, WARMUP:])
    held = np.arange(n) % 4 == 0
    every = np.ones(n, bool)
    out(f"Causal output filters on the v7+v9 valid predictions ({PRED[0]}, _t1; float16, {n} x {T} = steps "
        f"{WARMUP}..{WARMUP + T - 1}); targets / is_scored from {TARGETS}, {SCORED}")
    out(f"Metric: clip to [-2, 2], weights |y|, pooled over scored rows ({int(M.sum()):,}), mean of t0 and t1. "
        f"469 = valid index % 4 == 0 ({held.sum()} sequences).")

    S_raw = [seq_stats(Y[t], P[t], M) for t in range(2)]
    r_all, r_held = score(S_raw, every), score(S_raw, held)
    out(f"\nRaw: all 1,873 {r_all[2]:.4f} (t0 {r_all[0]:.4f}, t1 {r_all[1]:.4f}); 469 {r_held[2]:.4f} "
        f"(t0 {r_held[0]:.4f}, t1 {r_held[1]:.4f})")
    ok = abs(r_all[2] - REF_ALL) < 5e-5 and abs(r_held[2] - REF_HELD) < 5e-5
    out(f"  reproduces {REF_ALL} / {REF_HELD}: {'YES' if ok else 'NO'}")
    assert ok, "raw score does not reproduce the reference; stopping"

    # step-to-step noise and autocorrelation (clipped values, within sequences, steps >= 100)
    out("\nStep-to-step noise (clipped values, consecutive steps inside each sequence, steps 100..19999)")
    for t in range(2):
        p, y = np.clip(P[t].astype(np.float64), -2, 2), np.clip(Y[t].astype(np.float64), -2, 2)
        dp, dy = np.diff(p, axis=1), np.diff(y, axis=1)
        ms = M[:, 1:]  # rows t scored (difference ending on a scored row)
        def ac1(x, m=None):
            a, b = x[:, 1:], x[:, :-1]
            if m is not None:
                a, b = a[m], b[m]
            a, b = a - a.mean(), b - b.mean()
            return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        out(f"  t{t}: sd(p_t - p_t-1) {dp.std():.4f} vs sd(y_t - y_t-1) {dy.std():.4f} (ratio {dp.std() / dy.std():.3f}); "
            f"on scored rows {dp[ms].std():.4f} vs {dy[ms].std():.4f}; sd of levels p {p.std():.4f}, y {y.std():.4f}")
        out(f"      lag-1 autocorrelation: predictions {ac1(p):.4f}, targets {ac1(y):.4f}; on scored rows "
            f"{ac1(p, ms):.4f} vs {ac1(y, ms):.4f}; share of steps where p changes by > 0.01: "
            f"{np.mean(np.abs(dp) > 0.01):.3f}, y changes at all: {np.mean(dy != 0):.3f}")

    # (a) exponential smoothing
    out("\n(a) Exponential smoothing p'_t = a p_t + (1 - a) p'_t-1, restarted at step 99 (both targets the same a)")
    out(f"  {'a':>5} {'all 1,873':>10} {'gain':>8} {'t0':>7} {'t1':>7}   {'469':>7} {'gain':>8}   "
        f"{'gain even / odd':>17}   {'gain by index % 4':>32}")
    res_a = {}
    for a in ALPHAS:
        S = []
        for t in range(2):
            x = P[t].astype(np.float64)
            f = lfilter([a], [1, -(1 - a)], x, axis=1, zi=(1 - a) * x[:, :1])[0]
            S.append(seq_stats(Y[t], f, M))
        ra, rh = score(S, every), score(S, held)
        par = [score(S, np.arange(n) % 2 == k)[2] - score(S_raw, np.arange(n) % 2 == k)[2] for k in range(2)]
        q4 = [score(S, np.arange(n) % 4 == k)[2] - score(S_raw, np.arange(n) % 4 == k)[2] for k in range(4)]
        res_a[a] = (ra[2] - r_all[2], rh[2] - r_held[2], par, q4)
        out(f"  {a:>5} {ra[2]:>10.4f} {ra[2] - r_all[2]:>+8.4f} {ra[0]:>7.4f} {ra[1]:>7.4f}   {rh[2]:>7.4f} "
            f"{rh[2] - r_held[2]:>+8.4f}   {par[0]:+.4f} / {par[1]:+.4f}   " + " ".join(f"{g:+.4f}" for g in q4))

    # (b), (c) learned causal linear filter: per-sequence normal equations, then CV
    k = 1 + 2 * len(LAGS)
    G = np.zeros((2, n, k, k))
    b = np.zeros((2, n, k))
    for i in range(0, n, BATCH):
        sl = slice(i, min(i + BATCH, n))
        for t in range(2):
            X = design(P[0][sl].astype(np.float64), P[1][sl].astype(np.float64), t)
            yc = np.clip(Y[t][sl].astype(np.float64), -2, 2)
            w = np.abs(yc) + 0.05
            G[t, sl] = np.einsum("stk,st,stl->skl", X, w, X)
            b[t, sl] = np.einsum("stk,st->sk", X, w * yc)
    out(f"\nNormal equations accumulated ({time.time() - t_start:.0f}s); {k} regressors per target: constant, "
        f"lags {LAGS} of the same target and of the other target")

    for name, folds in [("(b) learned filter, 2-fold CV (even / odd index)", np.arange(n) % 2),
                        ("(c) learned filter, 4-fold CV (index % 4)", np.arange(n) % 4)]:
        nf = folds.max() + 1
        S = [np.zeros((n, 6)) for _ in range(2)]
        betas = {}
        for f in range(nf):
            tr, te = folds != f, folds == f
            for t in range(2):
                beta = np.linalg.solve(G[t, tr].sum(0), b[t, tr].sum(0))
                betas[(f, t)] = beta
                idx = np.flatnonzero(te)
                for i in range(0, len(idx), BATCH):
                    sl = idx[i:i + BATCH]
                    X = design(P[0][sl].astype(np.float64), P[1][sl].astype(np.float64), t)
                    S[t][sl] = seq_stats(Y[t][sl], X @ beta, M[sl])
        ra, rh = score(S, every), score(S, held)
        out(f"\n{name}: out-of-fold scores")
        out(f"  all 1,873 {ra[2]:.4f} (gain {ra[2] - r_all[2]:+.4f}; t0 {ra[0]:.4f} {ra[0] - r_all[0]:+.4f}, t1 {ra[1]:.4f} "
            f"{ra[1] - r_all[1]:+.4f}); 469 {rh[2]:.4f} (gain {rh[2] - r_held[2]:+.4f})")
        gains = []
        for f in range(nf):
            te = folds == f
            g = score(S, te)[2] - score(S_raw, te)[2]
            gains.append(g)
            out(f"  fold {f} ({te.sum()} test sequences): raw {score(S_raw, te)[2]:.4f} -> {score(S, te)[2]:.4f}, "
                f"gain {g:+.4f}")
        for t in range(2):
            bt = np.mean([betas[(f, t)] for f in range(nf)], 0)
            sd = np.std([betas[(f, t)] for f in range(nf)], 0)
            out(f"  t{t} mean coefficients (fold sd): const {bt[0]:+.4f}; own lags " + " ".join(
                f"{L}:{bt[1 + j]:+.3f}({sd[1 + j]:.3f})" for j, L in enumerate(LAGS)))
            out(f"     other target lags " + " ".join(
                f"{L}:{bt[1 + len(LAGS) + j]:+.3f}({sd[1 + len(LAGS) + j]:.3f})" for j, L in enumerate(LAGS)))
        res_a[name] = (ra[2] - r_all[2], rh[2] - r_held[2], gains, None)

    out(f"\nVerdict (pass = gain >= +{GAIN_BAR} on all 1,873, >= 0 on the 469, and > 0 in every fold; for (a) the "
        "folds are even / odd and index % 4, for (b) / (c) the CV folds):")
    passed = []
    for key, (ga, gh, par, q4) in res_a.items():
        fold_gains = par if q4 is None else par + q4
        if key == 1:
            continue
        if ga >= GAIN_BAR and min(fold_gains) > 0 and gh >= 0:
            passed.append(key)
        label = f"a = {key}" if not isinstance(key, str) else key.split(":")[0]
        out(f"  {label:<50} all {ga:+.4f}, 469 {gh:+.4f}, folds min {min(fold_gains):+.4f} / max {max(fold_gains):+.4f}")
    out(f"  -> {'PASS: ' + ', '.join(map(str, passed)) if passed else 'no filter gains +0.002 stably'}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {OUT} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
