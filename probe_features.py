"""Causal per-row features for error_analysis.py and residual_probe.py (numpy, CPU only).

The 44 price-like and 44 volume-like columns are anonymized: the docs say an index is not a depth level,
and the columns come with arbitrary sign and scale (on one sequence the i0 price diffs correlate +0.99
or -0.94 with i0_p0). Mid, spread and imbalance are therefore proxies, fitted once without targets
of valid (FIT_SEQS train sequences, saved to FACTORS for reproducibility):
  f0, f1    mid-price move of i0 / i1: projection of the 22 price diffs on their first principal component
            (sign chosen so the loadings sum to >= 0)
  sp0       i0 spread proxy: each price column with |loading on f0| >= MIN_LOAD divided by its loading (so
            it moves 1:1 with the mid), then mean of those ask columns (p11-p21) minus mean of those bid
            columns (p0-p10); up to a constant this is the average ask-minus-bid distance. Columns that
            barely follow the mid (loading near 0) are left out, as dividing by them only adds noise
  imb0, imb1  order-book imbalance proxy of i0 / i1: the linear combination of the instrument's 22
            volume columns that best predicts its next FUT-step mid move (ridge, train sequences only)
  vol0      i0 volatility: causal rolling std of f0 over the last 100 steps
All series are causal: the value at step t uses rows <= t only.
"""
import os

import numpy as np

CACHE = "/home/jovyan/work/cache"
FACTORS = "/home/jovyan/work/probe_factors.npz"
FIT_SEQS = 40
FUT = 20
WINDOWS = [5, 20, 100, 500]
MIN_LOAD = 0.05
SERIES = ["f0", "f1", "sp0", "imb0", "imb1"]
FILL = 5.19921875  # train_v1.FILL
FLAG_IDX = [60, 65, 66, 67, 26, 27, 28, 29, 56, 57, 58, 59]  # train_v1.FLAG_IDX (a0 a5 a6 a7, i0/i1 dv0-3)
VOL_IDX = {0: list(range(0, 22)), 1: list(range(30, 52))}  # i0_v0-21, i1_v0-21 in the 68 other columns
PRICE_IDX = {0: list(range(0, 22)), 1: list(range(22, 44))}


def fit_factors():
    """PCA loadings, per-column slopes and imbalance weights from FIT_SEQS evenly spaced train sequences."""
    if os.path.exists(FACTORS):
        d = np.load(FACTORS)
        return {k: d[k] for k in d.files}
    P = np.load(f"{CACHE}/train_prices.npy", mmap_mode="r")
    F = np.load(f"{CACHE}/train_features.npy", mmap_mode="r")
    idx = np.linspace(0, len(P) - 1, FIT_SEQS).astype(int)
    fac = {"fit_idx": idx}
    for i in (0, 1):
        dps, vols, futs = [], [], []
        for j in idx:
            p = np.asarray(P[j][:, PRICE_IDX[i]], dtype=np.float64)
            dps.append(np.diff(p, axis=0))
        dp = np.concatenate(dps)
        w, v = np.linalg.eigh(np.cov(dp.T))
        u = v[:, -1] * (1 if v[:, -1].sum() >= 0 else -1)
        f = dp @ u
        fac[f"u{i}"] = u
        fac[f"slope{i}"] = (dp * f[:, None]).sum(0) / (f * f).sum()  # Δp_k ≈ slope_k · f
        for j in idx:
            p = np.asarray(P[j][:, PRICE_IDX[i]], dtype=np.float64)
            m = np.concatenate([[0.0], np.cumsum(np.diff(p, axis=0) @ u)])
            vols.append(np.asarray(F[j][:-FUT, VOL_IDX[i]], dtype=np.float64))
            futs.append(m[FUT:] - m[:-FUT])
        X, y = np.concatenate(vols), np.concatenate(futs)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        beta = np.linalg.solve(Z.T @ Z + 1e3 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        fac[f"imb_mu{i}"], fac[f"imb_sd{i}"], fac[f"imb_w{i}"] = mu, sd, beta
        fac[f"imb_r{i}"] = np.corrcoef(Z @ beta, y)[0, 1]
    np.savez(FACTORS, **fac)
    return fac


def rolling(s, w):
    """Causal rolling mean and std over the last w values (fewer at the start)."""
    c1 = np.concatenate([[0.0], np.cumsum(s)])
    c2 = np.concatenate([[0.0], np.cumsum(s * s)])
    t = np.arange(1, len(s) + 1)
    lo = np.maximum(t - w, 0)
    n = t - lo
    m = (c1[t] - c1[lo]) / n
    v = np.maximum((c2[t] - c2[lo]) / n - m * m, 0.0)
    return m, np.sqrt(v)


def seq_series(prices, other, fac):
    """prices (T,44) f32, other (T,68) f16 -> dict of (T,) float64 causal series."""
    out = {}
    for i in (0, 1):
        p = prices[:, PRICE_IDX[i]].astype(np.float64)
        dp = np.zeros_like(p)
        dp[1:] = p[1:] - p[:-1]
        out[f"f{i}"] = dp @ fac[f"u{i}"]
        Z = (other[:, VOL_IDX[i]].astype(np.float64) - fac[f"imb_mu{i}"]) / fac[f"imb_sd{i}"]
        out[f"imb{i}"] = Z @ fac[f"imb_w{i}"]
    x = prices[:, :22].astype(np.float64) / fac["slope0"]
    keep = np.abs(fac["slope0"]) >= MIN_LOAD
    bid, ask = keep & (np.arange(22) < 11), keep & (np.arange(22) >= 11)
    out["sp0"] = x[:, ask].mean(1) - x[:, bid].mean(1)
    out["vol0"] = rolling(out["f0"], 100)[1]
    return out


def input_matrix(prices, other):
    """The 168 model inputs (train_v1.build_inputs without the standardization): prices, diffs, other, flags."""
    p = prices.astype(np.float32)
    dp = np.zeros_like(p)
    dp[1:] = p[1:] - p[:-1]
    o = other.astype(np.float32)
    flags = (np.abs(o[:, FLAG_IDX]) == FILL).astype(np.float32)
    return np.concatenate([p, dp, o, flags], axis=1)


def input_names(price_cols, other_cols):
    flag_cols = [other_cols[k] for k in FLAG_IDX]
    return price_cols + [f"d_{c}" for c in price_cols] + other_cols + [f"flag_{c}" for c in flag_cols]
