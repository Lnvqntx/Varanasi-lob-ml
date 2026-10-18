"""How much does the global weighted Pearson move across random subsets of valid sequences?

1. Run train_v2_best.pt on the valid cache (GPU 0 only, memory capped at 6 GB).
2. Per sequence and per target, store weighted sufficient statistics on clipped values over
   is_scored & step >= 99 rows: W = sum w, sum w*y, sum w*p, sum w*y^2, sum w*p^2, sum w*y*p, w = |y|.
3. Check: merging all sequences reproduces the full-valid score (and GlobalAccumulator on the same preds).
4. 1,000 random subsets (without replacement) at 5/10/25/50% of valid: mean, std, 5th-95th percentile.

Usage:
  python spread_v2.py --n 50                    # quick test on the first 50 valid sequences, no file saved
  python spread_v2.py                           # all 1,873 sequences -> ~/work/spread_v2_stats.npz
  python spread_v2.py --from-stats              # step 4 only, from the saved stats, no GPU
  python spread_v2.py --analyze                 # full analysis from the saved stats -> ~/work/spread_v2_results.txt
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import sys
import time

import numpy as np

STARTER = "/home/jovyan/work/wnn/wnn_connectome_starterpack"
sys.path.insert(0, STARTER)
from utils import METRIC_CLIP, WARMUP, GlobalAccumulator  # noqa: E402

CKPT = "/home/jovyan/work/checkpoints/train_v2_best.pt"
STATS_OUT = "/home/jovyan/work/spread_v2_stats.npz"
FULL_VALID_WP = 0.6797034822967548  # official scorer, submission_v2_int8_std.zip
OFFICIAL = (0.6752908623181588, 0.6841161022753508, FULL_VALID_WP)  # t0, t1, wp of that run
RESULTS_OUT = "/home/jovyan/work/spread_v2_results.txt"
DROP = 0.028  # local full valid 0.6797 vs public leaderboard 0.6520
GPU_MEM_CAP_GB = 6.0
FRACS = [0.05, 0.10, 0.25, 0.50]
STAT_NAMES = ["w", "wy", "wp", "wyy", "wpp", "wyp"]


def seq_stats(targets, preds, mask):
    """(2, 6) float64 sufficient statistics for one sequence."""
    y = np.clip(targets[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    p = np.clip(preds[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    w = np.abs(y)
    return np.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0),
                     (w * y * y).sum(0), (w * p * p).sum(0), (w * y * p).sum(0)], axis=-1)


def wp_from_stats(s):
    """s (..., 2, 6) merged sums -> (..., 3) [t0, t1, mean], same formula and clipping as GlobalAccumulator."""
    w, wy, wp, wyy, wpp, wyp = np.moveaxis(s, -1, 0)
    my, mp = wy / w, wp / w
    vy = np.maximum(wyy / w - my * my, 0.0)
    vp = np.maximum(wpp / w - mp * mp, 0.0)
    cov = wyp / w - my * mp
    sy, sp = np.sqrt(vy), np.sqrt(vp)
    ok = (w > 1e-8) & (sy > 1e-8) & (sp > 1e-8)
    r = np.where(ok, np.clip(cov / np.where(ok, sy * sp, 1.0), -1.0, 1.0), 0.0)
    return np.concatenate([r, r.mean(-1, keepdims=True)], axis=-1)


def compute_stats(n, batch):
    import torch
    import train_v1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = train_v1.GRUModel(hidden=ck["args"]["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    print(f"{CKPT}: step {ck['step']}, hidden {ck['args']['hidden']}, logged valid(300) {ck['valid']['weighted_pearson']:.4f}")

    valid = train_v1.load_split("valid")
    n = n or valid["prices"].shape[0]
    need = np.arange(train_v1.SEQ_LEN) >= WARMUP
    out = np.zeros((n, 2, 6), dtype=np.float64)
    rows = np.zeros(n, dtype=np.int64)
    acc = GlobalAccumulator()
    batches = [np.arange(i, min(i + batch, n)) for i in range(0, n, batch)]  # contiguous, already sorted
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            x = train_v1.build_inputs(torch.from_numpy(b["prices"]).to(device),
                                      torch.from_numpy(b["features"]).to(device), stats)
            h, preds = None, []
            for s in range(0, train_v1.SEQ_LEN, 1000):
                p, h = model(x[:, s:s + 1000], h)
                preds.append(p)
            preds = torch.cat(preds, 1).float().cpu().numpy()
            for k, i in enumerate(batches[bi]):
                mask = b["is_scored"][k] & need
                out[i] = seq_stats(b["targets"][k], preds[k], mask)
                rows[i] = mask.sum()
                acc.add(b["targets"][k], preds[k], mask)
            if (bi + 1) % 10 == 0:
                print(f"{batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"model pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
    return out, rows, acc.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="first N valid sequences only (test; nothing saved)")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--from-stats", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="full analysis from saved stats -> RESULTS_OUT")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.from_stats or args.analyze:
        d = np.load(STATS_OUT)
        S, rows = d["stats"], d["rows"]
    else:
        S, rows, ref = compute_stats(args.n, args.batch)
        merged = wp_from_stats(S.sum(0))
        print(f"merged stats over {len(S)} seqs, {rows.sum():,} scored rows: wp={merged[2]:.6f} "
              f"(t0 {merged[0]:.6f}, t1 {merged[1]:.6f})")
        print(f"GlobalAccumulator on same preds:  wp={ref['weighted_pearson']:.6f} "
              f"(t0 {ref['t0']:.6f}, t1 {ref['t1']:.6f}) | |diff| {abs(merged[2] - ref['weighted_pearson']):.2e}")
        if args.n is None:
            print(f"full-valid official run (int8 ONNX): {FULL_VALID_WP:.6f} | diff {merged[2] - FULL_VALID_WP:+.6f}")
            if os.path.exists(STATS_OUT):
                sys.exit(f"{STATS_OUT} exists; not overwriting")
            np.savez(STATS_OUT, stats=S, rows=rows, stat_names=STAT_NAMES, ckpt=CKPT)
            print(f"saved {STATS_OUT}")

    if args.analyze:
        analyze(S, rows, args.n_draws, args.seed)
    else:
        subset_table(S, args.n_draws, args.seed, print)


def subset_table(S, n_draws, seed, out):
    """Random subsets without replacement at each fraction; returns [(k, std of wp)]."""
    n = len(S)
    full = wp_from_stats(S.sum(0))[2]
    rng = np.random.default_rng(seed)
    out(f"{n_draws} random subsets of {n} sequences (without replacement, seed {seed}); full-set wp {full:.4f}")
    out(f"{'frac':>5} {'seqs':>5} {'mean':>8} {'std':>8} {'p5':>8} {'p95':>8} {'p95-p5':>8} {'std t0':>8} {'std t1':>8}")
    res = []
    for f in FRACS:
        k = max(1, round(f * n))
        idx = np.argsort(rng.random((n_draws, n)), axis=1)[:, :k]
        r = wp_from_stats(S[idx].sum(1))  # (draws, 3)
        wp = r[:, 2]
        p5, p95 = np.percentile(wp, [5, 95])
        out(f"{f:>5.0%} {k:>5} {wp.mean():>8.4f} {wp.std(ddof=1):>8.4f} {p5:>8.4f} {p95:>8.4f} {p95 - p5:>8.4f} "
            f"{r[:, 0].std(ddof=1):>8.4f} {r[:, 1].std(ddof=1):>8.4f}")
        res.append((k, wp.std(ddof=1)))
    return res


def analyze(S, rows, n_draws, seed):
    """Steps 2-4 of the spread analysis from the saved stats; printed and written to RESULTS_OUT."""
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    n = len(S)
    out(f"Score-spread analysis, {CKPT} (fp32 PyTorch), stats {STATS_OUT}")
    out(f"{n} valid sequences, {rows.sum():,} scored rows (is_scored & step >= {WARMUP})")

    out("\n== 1. Merged all-sequence score vs official full-valid run (int8 ONNX, ScorerStepByStep) ==")
    r = wp_from_stats(S.sum(0))
    for name, mine, off in zip(["t0", "t1", "wp"], r, OFFICIAL):
        out(f"  {name}: merged fp32 {mine:.6f} | official int8 {off:.6f} | diff {mine - off:+.6f}")
    out(f"  |diff wp| {abs(r[2] - OFFICIAL[2]):.6f} -> {'within' if abs(r[2] - OFFICIAL[2]) <= 0.001 else 'OUTSIDE'} "
        f"the expected 0.001")

    out("\n== 2. Subset spread ==")
    res = subset_table(S, n_draws, seed, lambda s: out("  " + s))

    out("\n== 3a. Per-sequence weighted Pearson ==")
    ps = wp_from_stats(S)  # (n, 3)
    degen = int(((S[:, :, 0] <= 1e-8)).any(1).sum())
    out(f"  sequences with a target of zero total weight (score set to 0 for it): {degen}")
    out(f"  {'':>4} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8} {'share<0':>8}")
    for j, name in enumerate(["t0", "t1", "wp"]):
        v = ps[:, j]
        out(f"  {name:>4} {v.mean():>8.4f} {np.median(v):>8.4f} {np.percentile(v, 10):>8.4f} "
            f"{np.percentile(v, 90):>8.4f} {(v < 0).mean():>8.2%}")
    W = S[:, :, 0]
    out(f"  weight-weighted mean of per-seq wp {np.average(ps[:, 2], weights=W.sum(1)):.4f} "
        f"(vs global {r[2]:.4f}: the global score also rewards getting between-sequence levels right)")

    out("\n== 3b. Weight concentration (weight = sum |clipped y| over scored rows) ==")
    out(f"  {'':>8} {'top 1%':>8} {'top 5%':>8} {'top 10%':>8}")
    for name, w in [("t0", W[:, 0]), ("t1", W[:, 1]), ("t0+t1", W.sum(1))]:
        ws = np.sort(w)[::-1] / w.sum()
        out(f"  {name:>8} " + " ".join(f"{ws[:max(1, int(np.ceil(q * n)))].sum():>8.2%}" for q in [0.01, 0.05, 0.10]))
    out(f"  (uniform weights would give 1%, 5%, 10%; scored rows per sequence: min {rows.min():,}, "
        f"median {int(np.median(rows)):,}, max {rows.max():,})")

    out(f"\n== 3c. Subset SD vs 1/sqrt(k), and sizes at which a drop of {DROP} is a 1/2/3-SD event ==")
    k = np.array([a for a, _ in res], dtype=float)
    sd = np.array([b for _, b in res])
    x = 1 / np.sqrt(k)
    c = (sd * x).sum() / (x * x).sum()
    xf = np.sqrt(1 / k - 1 / n)  # with the finite-population correction (subsets are drawn from these n)
    cf = (sd * xf).sum() / (xf * xf).sum()
    out(f"  plain fit      SD = {c:.4f} / sqrt(k)")
    out(f"  with FPC fit   SD = {cf:.4f} * sqrt(1/k - 1/{n})   (c = SD of a single-sequence-sized sample from the "
        f"population; use for an independent set such as the test set)")
    out(f"  {'k':>5} {'observed':>9} {'plain fit':>9} {'FPC fit':>9}")
    for ki, si, xi, xfi in zip(k, sd, x, xf):
        out(f"  {int(ki):>5} {si:>9.4f} {c * xi:>9.4f} {cf * xfi:>9.4f}")
    out(f"  sequences k needed so that {DROP} = z SD:")
    out(f"  {'z':>3} {'plain k':>9} {'indep k':>9} {'indep k, valid noise too':>25}")
    for z in [1, 2, 3]:
        k_plain = (z * c / DROP) ** 2
        k_ind = (z * cf / DROP) ** 2
        # gap between full valid (n seqs) and an independent k-seq set: var = cf^2 (1/k + 1/n)
        inv = (DROP / (z * cf)) ** 2 - 1 / n
        k_both = f"{1 / inv:,.0f}" if inv > 0 else "impossible"
        out(f"  {z:>3} {k_plain:>9,.0f} {k_ind:>9,.0f} {k_both:>25}")
    out("  plain k: from the plain fit. indep k: an independent sample of k sequences (FPC fit, infinite population).")
    out(f"  last column: the full-valid score itself also has noise (SD {cf / np.sqrt(n):.4f} for {n} seqs) when "
        f"compared with an independent set.")

    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}")


if __name__ == "__main__":
    main()
