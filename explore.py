"""Read-only exploration of train.parquet, one row group (= one sequence) at a time.

Reads at most N_SEQ row groups, spread evenly across the file. Never writes data.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PATH = "/home/jovyan/work/wnn/wnn_connectome_starterpack/datasets/train.parquet"
N_SEQ = 20
LAGS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
TARGETS = ["t0", "t1"]

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def acf(x, lags):
    x = x - x.mean()
    denom = (x * x).sum()
    return [np.nan if lag >= len(x) or denom == 0 else (x[:-lag] * x[lag:]).sum() / denom for lag in lags]


def main():
    pf = pq.ParquetFile(PATH)
    md = pf.metadata
    print(f"file: {md.num_rows:,} rows, {md.num_row_groups:,} row groups, {md.num_columns} columns")
    rg_ids = np.linspace(0, md.num_row_groups - 1, N_SEQ).astype(int)
    print(f"reading row groups: {rg_ids.tolist()}\n")

    feat_cols = [c for c in pf.schema_arrow.names
                 if c not in ("seq_ix", "step_in_seq", "need_prediction")]

    # Running per-column stats (streamed, so memory stays at one row group)
    n = pd.Series(0, index=feat_cols, dtype="int64")
    n_nan = pd.Series(0, index=feat_cols, dtype="int64")
    n_inf = pd.Series(0, index=feat_cols, dtype="int64")
    s1 = pd.Series(0.0, index=feat_cols)
    s2 = pd.Series(0.0, index=feat_cols)
    cmin = pd.Series(np.inf, index=feat_cols)
    cmax = pd.Series(-np.inf, index=feat_cols)
    per_seq_q = []  # per-sequence quantiles, combined at the end (approximate)

    targets_all = {t: [] for t in TARGETS}        # all rows
    targets_pred = {t: [] for t in TARGETS}       # rows with need_prediction
    per_seq = []
    acf_rows = {t: [] for t in TARGETS}
    acf_diff_rows = {t: [] for t in TARGETS}

    for rg in rg_ids:
        df = pf.read_row_group(int(rg)).to_pandas()
        seqs = df["seq_ix"].unique()
        assert len(seqs) == 1, f"row group {rg} has {len(seqs)} sequences"
        assert (np.diff(df["step_in_seq"].values) == 1).all(), f"row group {rg} steps not contiguous"

        X = df[feat_cols].astype("float64")
        vals = X.values
        finite = np.isfinite(vals)
        n += len(df)
        n_nan += np.isnan(vals).sum(0)
        n_inf += np.isinf(vals).sum(0)
        v0 = np.where(finite, vals, 0.0)
        s1 += v0.sum(0)
        s2 += (v0 * v0).sum(0)
        cmin = np.minimum(cmin, np.where(finite, vals, np.inf).min(0))
        cmax = np.maximum(cmax, np.where(finite, vals, -np.inf).max(0))
        per_seq_q.append(X.quantile([0.001, 0.5, 0.999]))

        mask = df["need_prediction"].values
        rec = {"rg": int(rg), "seq_ix": int(seqs[0]), "len": len(df),
               "n_need_pred": int(mask.sum()),
               "first_pred_step": int(df["step_in_seq"].values[mask][0]) if mask.any() else -1}
        for t in TARGETS:
            y = df[t].values.astype("float64")
            targets_all[t].append(y)
            targets_pred[t].append(y[mask])
            rec[f"{t}_std"] = y.std()
            rec[f"{t}_frac_unchanged"] = (np.diff(y) == 0).mean()
            yf = y[np.isfinite(y)]
            acf_rows[t].append(acf(yf, LAGS))
            acf_diff_rows[t].append(acf(np.diff(yf), LAGS))
        yy = df[TARGETS].dropna()
        rec["corr_t0_t1"] = yy["t0"].corr(yy["t1"])
        per_seq.append(rec)
        del df, X, vals

    # ---------- Features ----------
    mean = s1 / (n - n_nan - n_inf)
    std = np.sqrt(s2 / (n - n_nan - n_inf) - mean ** 2)
    q = pd.concat(per_seq_q).groupby(level=0).median()  # median of per-seq quantiles
    feat = pd.DataFrame({"mean": mean, "std": std, "min": cmin, "q0.1%": q.loc[0.001],
                         "median": q.loc[0.5], "q99.9%": q.loc[0.999], "max": cmax,
                         "nan": n_nan, "inf": n_inf})
    print("=" * 100)
    print("FEATURE SCALES (over sampled sequences; quantiles = median of per-sequence quantiles)")
    print("=" * 100)
    print(feat.round(3).to_string())

    groups = {"i0_p": "i0_p", "i0_v": "i0_v", "i0_dp": "i0_dp", "i0_dv": "i0_dv",
              "i1_p": "i1_p", "i1_v": "i1_v", "i1_dp": "i1_dp", "i1_dv": "i1_dv", "a": "a", "t": "t"}
    print("\nPer-group summary:")
    rows = []
    for g, pre in groups.items():
        cols = [c for c in feat_cols if c.startswith(pre) and c[len(pre):].isdigit()]
        sub = feat.loc[cols]
        rows.append({"group": g, "ncols": len(cols), "mean_range": f"[{sub['mean'].min():.2f}, {sub['mean'].max():.2f}]",
                     "std_range": f"[{sub['std'].min():.2f}, {sub['std'].max():.2f}]",
                     "min": sub["min"].min(), "max": sub["max"].max(), "nan": int(sub["nan"].sum()),
                     "inf": int(sub["inf"].sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal NaN: {int(n_nan.sum())}, total inf: {int(n_inf.sum())}, "
          f"cols with any NaN: {list(feat.index[feat['nan'] > 0])}")

    # ---------- Sequence structure ----------
    ps = pd.DataFrame(per_seq)
    print("\n" + "=" * 100)
    print("PER-SEQUENCE")
    print("=" * 100)
    print(ps.round(3).to_string(index=False))

    # ---------- Target distributions ----------
    print("\n" + "=" * 100)
    print("TARGET DISTRIBUTIONS")
    print("=" * 100)
    for label, src in [("all rows", targets_all), ("need_prediction rows", targets_pred)]:
        print(f"\n-- {label} --")
        rows = []
        for t in TARGETS:
            y = np.concatenate(src[t])
            y = y[np.isfinite(y)]
            z = (y - y.mean()) / y.std()
            r = {"target": t, "n": len(y), "mean": y.mean(), "std": y.std(),
                 "skew": (z ** 3).mean(), "excess_kurt": (z ** 4).mean() - 3,
                 "min": y.min(), "max": y.max()}
            for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]:
                r[f"p{p}"] = np.percentile(y, p)
            for k in [1, 2, 3, 5, 10]:
                r[f"P(|t|>{k})"] = (np.abs(y) > k).mean()
            r["P(|z|>3) [normal=0.0027]"] = (np.abs(z) > 3).mean()
            rows.append(r)
        print(pd.DataFrame(rows).set_index("target").T.to_string(float_format=lambda v: f"{v:.4g}"))

    # ---------- Correlation ----------
    print("\n" + "=" * 100)
    print("t0 vs t1 CORRELATION")
    print("=" * 100)
    for label, src in [("all rows", targets_all), ("need_prediction rows", targets_pred)]:
        a, b = np.concatenate(src["t0"]), np.concatenate(src["t1"])
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        pear = np.corrcoef(a, b)[0, 1]
        spear = pd.Series(a).rank().corr(pd.Series(b).rank())
        big = (np.abs(a) > 2) | (np.abs(b) > 2)
        pear_tail = np.corrcoef(a[big], b[big])[0, 1]
        print(f"{label:>22}: pearson={pear:.4f} spearman={spear:.4f} "
              f"pearson on rows with |t0|>2 or |t1|>2 (n={big.sum()})={pear_tail:.4f}")
        print(f"{'':>22}  P(|t0|>2 & |t1|>2)={((np.abs(a) > 2) & (np.abs(b) > 2)).mean():.4f}, "
              f"P(|t0|>2)P(|t1|>2)={(np.abs(a) > 2).mean() * (np.abs(b) > 2).mean():.4f}")
    print(f"per-sequence pearson: mean={ps['corr_t0_t1'].mean():.4f} "
          f"min={ps['corr_t0_t1'].min():.4f} max={ps['corr_t0_t1'].max():.4f}")

    # ---------- Autocorrelation ----------
    print("\n" + "=" * 100)
    print("AUTOCORRELATION (within-sequence, averaged over sequences)")
    print("=" * 100)
    for t in TARGETS:
        A = pd.DataFrame(acf_rows[t], columns=LAGS)
        D = pd.DataFrame(acf_diff_rows[t], columns=LAGS)
        out = pd.DataFrame({"acf_mean": A.mean(), "acf_min": A.min(), "acf_max": A.max(),
                            "diff_acf_mean": D.mean()})
        out.index.name = "lag"
        print(f"\n{t}:")
        print(out.round(4).T.to_string())
        print(f"  fraction of steps where {t} is unchanged from previous step: "
              f"{ps[f'{t}_frac_unchanged'].mean():.4f}")


if __name__ == "__main__":
    main()
