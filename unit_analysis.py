"""Per-unit analysis of the v7+v9 ensemble on valid (numpy, CPU, 2 threads; no model, no features).

Targets are integer multiples of a per-sequence unit (target_structure.py: 0.1254 / 0.2413 / 0.4685 / 0.6569).
For every valid sequence the unit is the smallest nonzero |target| over all steps and both targets, matched to
the nearest known unit (relative tolerance 1%; anything else is reported). Per unit group, on is_scored & step
>= 99 (the metric rows): sequences, share of the metric weight (sum |clipped y|), mean |y|, weighted Pearson of
the v7+v9 ensemble (t0, t1, mean), share of the total weighted squared error (weights |clipped y|, both clipped),
and the weighted least-squares slope (with intercept) of clipped y on the raw prediction, weights |clipped y| +
0.05. Also mean mask probability (valid_maskp.npy) and scored rate per group.
Checks: overall score from the merged group sums must equal 0.6850 (calib_check: float16 preds give 0.6850).
Upper bound: each group's prediction multiplied by its own slope (then clipped), overall score recomputed;
in-sample, and 2-fold across sequences (slopes fitted on even-index sequences applied to odd and vice versa).

Inputs (small files only): ~/work/cache/valid_targets.npy, valid_is_scored.npy, valid_maskp.npy and
~/work/calib_v7v9_pred_t0.npy / _t1.npy (float16, (1873, 19901) = steps 99..19999).

  nice -n 10 python unit_analysis.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "2"

import sys

import numpy as np

CACHE = "/home/jovyan/work/cache"
PRED = [f"/home/jovyan/work/calib_v7v9_pred_t{j}.npy" for j in (0, 1)]
OUT = "/home/jovyan/work/varanasi-lob-ml/unit_analysis_results.txt"
UNITS = [0.1254, 0.2413, 0.4685, 0.6569]
WARMUP = 99
CLIP = 2.0
REL_TOL = 0.01
EXPECT = 0.6850


def sums(y, p):
    """(2, 6) float64 [w, wy, wp, wyy, wpp, wyp] on clipped y and p, weights |clipped y| (spread_v2.seq_stats)."""
    y = np.clip(y, -CLIP, CLIP).astype(np.float64)
    p = np.clip(p, -CLIP, CLIP).astype(np.float64)
    w = np.abs(y)
    return np.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0), (w * y * y).sum(0), (w * p * p).sum(0),
                     (w * y * p).sum(0)], axis=-1)


def wp(s):
    """(..., 2, 6) -> (..., 3) [t0, t1, mean], same formula as spread_v2.wp_from_stats."""
    w, wy, wp_, wyy, wpp, wyp = np.moveaxis(s, -1, 0)
    my, mp = wy / w, wp_ / w
    vy, vp = np.maximum(wyy / w - my * my, 0), np.maximum(wpp / w - mp * mp, 0)
    r = np.clip((wyp / w - my * mp) / np.sqrt(vy * vp), -1, 1)
    return np.concatenate([r, r.mean(-1, keepdims=True)], axis=-1)


def slope(y, p):
    """weighted LS slope (with intercept) of clipped y on raw p, weights |clipped y| + 0.05; per target."""
    y = np.clip(y, -CLIP, CLIP).astype(np.float64)
    p = p.astype(np.float64)
    w = np.abs(y) + 0.05
    my, mp = (w * y).sum(0) / w.sum(0), (w * p).sum(0) / w.sum(0)
    return (w * (y - my) * (p - mp)).sum(0) / (w * (p - mp) ** 2).sum(0)


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; not overwriting")
    tg = np.load(f"{CACHE}/valid_targets.npy", mmap_mode="r")
    sc = np.load(f"{CACHE}/valid_is_scored.npy", mmap_mode="r")
    mp = np.load(f"{CACHE}/valid_maskp.npy", mmap_mode="r")
    pr = [np.load(f, mmap_mode="r") for f in PRED]
    n, T = sc.shape
    assert pr[0].shape == (n, T - WARMUP) and tg.shape == (n, T, 2)

    unit, unit_raw, max_frac = np.full(n, -1), np.zeros(n), np.zeros(n)
    ys, ps, gs, seq_of = [], [], [], []
    mean_p, n_rows, n_scored = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        y_all = np.asarray(tg[i])
        a = np.abs(y_all[y_all != 0])
        u = a.min()
        unit_raw[i] = u
        k = int(np.argmin([abs(u / x - 1) for x in UNITS]))
        if abs(u / UNITS[k] - 1) <= REL_TOL:
            unit[i] = k
        max_frac[i] = np.abs(y_all / u - np.round(y_all / u)).max()  # integer-multiple check
        m = np.asarray(sc[i, WARMUP:])
        y = y_all[WARMUP:][m]
        p = np.stack([np.asarray(pr[0][i]), np.asarray(pr[1][i])], -1)[m].astype(np.float32)
        ys.append(y)
        ps.append(p)
        gs.append(np.full(len(y), unit[i]))
        seq_of.append(np.full(len(y), i))
        q = np.asarray(mp[i, WARMUP:]).astype(np.float64)
        mean_p[i], n_rows[i], n_scored[i] = q.mean(), len(q), m.sum()
    y, p, g, sq = np.concatenate(ys), np.concatenate(ps), np.concatenate(gs), np.concatenate(seq_of)
    del ys, ps, gs, seq_of

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("Unit analysis: v7+v9 ensemble (fp32 average saved as float16 by calib_check.py) on all 1,873 valid sequences")
    out("Metric rows: is_scored & step >= 99; targets and predictions clipped to [-2, 2]; weights |clipped y|.")
    out(f"Unit = smallest nonzero |target| over all steps and both targets; matched to {UNITS} within {REL_TOL:.0%}.")
    bad = np.flatnonzero(unit < 0)
    out(f"\nUnmatched sequences: {len(bad)}" + ("" if not len(bad) else
        " -> " + ", ".join(f"valid index {i} (smallest |y| {unit_raw[i]:.5f})" for i in bad)))
    for k, x in enumerate(UNITS):
        r = unit_raw[unit == k]
        if len(r):
            out(f"  unit {x}: measured smallest |y| {r.min():.6f} .. {r.max():.6f}; max |y/unit - round| "
                f"{max_frac[unit == k].max():.2e}")

    S_tot = sums(y, p)
    w_all = np.abs(np.clip(y, -CLIP, CLIP))
    e_all = w_all * (np.clip(y, -CLIP, CLIP) - np.clip(p, -CLIP, CLIP)) ** 2
    groups = [k for k in range(len(UNITS)) if (unit == k).any()] + ([-1] if len(bad) else [])
    S_g, slopes = {}, {}
    out(f"\n  {'unit':>8} {'seqs':>5} {'weight':>7} {'mean|y|':>8} {'mean|yc|':>8} {'wp t0':>7} {'wp t1':>7} "
        f"{'wp mean':>7} {'sq err':>7} {'slope t0':>8} {'slope t1':>8} {'mean p':>7} {'scored':>7}")
    for k in groups:
        r = g == k
        S_g[k] = sums(y[r], p[r])
        slopes[k] = slope(y[r], p[r])
        R = wp(S_g[k])
        name = f"{UNITS[k]}" if k >= 0 else "other"
        s_seq = unit == k
        out(f"  {name:>8} {s_seq.sum():>5} {w_all[r].sum() / w_all.sum():>7.1%} {np.abs(y[r]).mean():>8.4f} "
            f"{w_all[r].mean():>8.4f} {R[0]:>7.4f} {R[1]:>7.4f} {R[2]:>7.4f} {e_all[r].sum() / e_all.sum():>7.1%} "
            f"{slopes[k][0]:>8.3f} {slopes[k][1]:>8.3f} {(mean_p[s_seq] * n_rows[s_seq]).sum() / n_rows[s_seq].sum():>7.4f}"
            f" {n_scored[s_seq].sum() / n_rows[s_seq].sum():>7.4f}")
    R_all = wp(S_tot)
    R_merge = wp(sum(S_g.values()))
    sl_all = slope(y, p)
    out(f"  {'all':>8} {n:>5} {1:>7.1%} {np.abs(y).mean():>8.4f} {w_all.mean():>8.4f} {R_all[0]:>7.4f} "
        f"{R_all[1]:>7.4f} {R_all[2]:>7.4f} {1:>7.1%} {sl_all[0]:>8.3f} {sl_all[1]:>8.3f}")
    out("  weight = share of sum |clipped y|; sq err = share of sum |yc| (yc - pc)^2; slope = weighted LS slope of "
        "clipped y on raw p (weights |yc| + 0.05, with intercept); mean p = mask probability over steps >= 99; "
        "scored = is_scored rate")
    out(f"\nCheck: overall from the merged group sums {R_merge[2]:.6f} (t0 {R_merge[0]:.4f}, t1 {R_merge[1]:.4f}); "
        f"direct {R_all[2]:.6f}; expected {EXPECT:.4f} -> {'OK' if round(R_merge[2], 4) == EXPECT else 'MISMATCH'}")

    def rescaled(slope_of):
        q = p.copy()
        for k in groups:
            r = g == k
            q[r] = p[r] * slope_of(k, r).astype(np.float32)
        return wp(sums(y, q))

    R_in = rescaled(lambda k, r: slopes[k])
    even = sq % 2 == 0
    fold = {(k, f): slope(y[(g == k) & (even == f)], p[(g == k) & (even == f)]) for k in groups for f in (True, False)}
    q = p.copy()
    for k in groups:
        for f in (True, False):
            r = (g == k) & (even == f)
            q[r] = p[r] * fold[(k, not f)].astype(np.float32)  # slope from the other half
    R_cv = wp(sums(y, q))
    R_one = rescaled(lambda k, r: sl_all)  # one global slope: should change only through the clip
    out(f"\nPer-unit rescaling (prediction x its group's slope, then clipped):")
    out(f"  in-sample slopes:           {R_in[2]:.4f} (t0 {R_in[0]:.4f}, t1 {R_in[1]:.4f}), {R_in[2] - R_all[2]:+.4f}")
    out(f"  2-fold (even/odd sequences): {R_cv[2]:.4f} (t0 {R_cv[0]:.4f}, t1 {R_cv[1]:.4f}), {R_cv[2] - R_all[2]:+.4f}")
    out(f"  one global slope (control): {R_one[2]:.4f}, {R_one[2] - R_all[2]:+.4f}")
    out("  This is an upper bound: it uses the true unit of each sequence, which is NOT known at test time "
        "(the targets are not given there); a model would have to infer it from the inputs.")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
