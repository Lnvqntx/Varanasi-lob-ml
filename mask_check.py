"""Mask robustness on valid: is the model tuned to the moments the is_scored mask selects?

For each checkpoint, one pass over all 1,873 valid sequences (GPU 0, 6 GB cap, batches of 25,
1000-step chunks, as spread_v2.py), then global weighted Pearson (t0, t1, mean) on:
  scored    is_scored & step >= 99          (= the official metric; v2 must match 0.6797 within 0.0005)
  all       step >= 99, ignoring is_scored
  unscored  step >= 99 & ~is_scored
each over all steps >= 99 and split by step range. Per-sequence sufficient statistics in float64 on
clipped values with w = |clipped y| (spread_v2.seq_stats), merged by summing.

Usage:
  nohup python mask_check.py > ~/work/mask_check.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time

import numpy as np

from spread_v2 import GPU_MEM_CAP_GB, METRIC_CLIP, OFFICIAL, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

CKPTS = {"v2": "/home/jovyan/work/checkpoints/train_v2_best.pt",
         "v4": "/home/jovyan/work/checkpoints/train_v4_h128_e5_fullval_best.pt"}
STATS_OUT = "/home/jovyan/work/mask_check_stats.npz"
RESULTS_OUT = "/home/jovyan/work/mask_check_results.txt"
RANGES = [(WARMUP, 20000), (99, 1000), (1000, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
RANGE_NAMES = ["all >=99", "99-999", "1000-4999", "5000-9999", "10000-14999", "15000-19999"]
SETS = ["scored", "all", "unscored"]
TOL = 0.0005
BATCH, CHUNK = 25, 1000


def masks_for(is_scored, steps):
    """(n_ranges, n_sets, T) bool row selections for one sequence."""
    out = []
    for lo, hi in RANGES:
        r = (steps >= lo) & (steps < hi)
        out.append([r & is_scored, r, r & ~is_scored])
    return np.array(out)


def run_ckpt(path, valid, device):
    """Per-sequence stats (n, n_ranges, n_sets, 2, 6) for one checkpoint."""
    import torch
    import train_v1
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = train_v1.GRUModel(hidden=ck["args"]["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    print(f"{path}: step {ck['step']}, hidden {ck['args']['hidden']}, n_val used for selection {ck['args']['n_val']}",
          flush=True)
    n, T = valid["prices"].shape[:2]
    steps = np.arange(T)
    S = np.zeros((n, len(RANGES), len(SETS), 2, 6), dtype=np.float64)
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            x = train_v1.build_inputs(torch.from_numpy(b["prices"]).to(device),
                                      torch.from_numpy(b["features"]).to(device), stats)
            h, preds = None, []
            for s in range(0, T, CHUNK):
                p, h = model(x[:, s:s + CHUNK], h)
                preds.append(p)
            preds = torch.cat(preds, 1).float().cpu().numpy()
            for k, i in enumerate(batches[bi]):
                M = masks_for(b["is_scored"][k], steps)
                for ri in range(len(RANGES)):
                    for si in range(len(SETS)):
                        if M[ri, si].any():
                            S[i, ri, si] = seq_stats(b["targets"][k], preds[k], M[ri, si])
            if (bi + 1) % 15 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"  pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True)
    return S


def target_stats(valid):
    """Model-independent: rows, sum |y| raw and clipped, per range x set (summed over sequences)."""
    n, T = valid["targets"].shape[:2]
    steps = np.arange(T)
    rows = np.zeros((len(RANGES), len(SETS)), dtype=np.int64)
    abs_raw = np.zeros((len(RANGES), len(SETS), 2))
    abs_clip = np.zeros((len(RANGES), len(SETS), 2))
    for i0 in range(0, n, BATCH):
        y = np.asarray(valid["targets"][i0:i0 + BATCH], dtype=np.float64)
        sc = np.asarray(valid["is_scored"][i0:i0 + BATCH])
        for k in range(len(y)):
            M = masks_for(sc[k], steps)
            for ri in range(len(RANGES)):
                for si in range(len(SETS)):
                    m = M[ri, si]
                    rows[ri, si] += m.sum()
                    abs_raw[ri, si] += np.abs(y[k][m]).sum(0)
                    abs_clip[ri, si] += np.abs(np.clip(y[k][m], -METRIC_CLIP, METRIC_CLIP)).sum(0)
    return rows, abs_raw, abs_clip


def report(all_S, rows, abs_raw, abs_clip):
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("Mask robustness check on valid (1,873 sequences), fp32 PyTorch, GPU 0")
    out("Global weighted Pearson, clipped to [-2, 2], weights |clipped target|; sets restricted to step >= 99.")
    out("  scored = is_scored rows (official metric), all = every row, unscored = rows with is_scored False")
    for name, path in CKPTS.items():
        out(f"  {name}: {path}")
    out("  note: v4's checkpoint was selected on all 1,873 valid sequences' scored rows, v2's on 300 of them.")

    out("\n== A. Whole valid (step >= 99) ==")
    out(f"  {'model':<5} {'set':<9} {'rows':>11} {'t0':>8} {'t1':>8} {'mean':>8} {'vs scored':>10}")
    for name, S in all_S.items():
        r = wp_from_stats(S.sum(0))  # (n_ranges, n_sets, 3)
        for si, sname in enumerate(SETS):
            out(f"  {name:<5} {sname:<9} {rows[0, si]:>11,} {r[0, si, 0]:>8.4f} {r[0, si, 1]:>8.4f} {r[0, si, 2]:>8.4f} "
                f"{r[0, si, 2] - r[0, 0, 2]:>+10.4f}")
    r2 = wp_from_stats(all_S["v2"].sum(0))[0, 0]
    out(f"  v2 scored vs official 0.679703 (int8 ONNX): {r2[2] - OFFICIAL[2]:+.6f} (t0 {r2[0] - OFFICIAL[0]:+.6f}, "
        f"t1 {r2[1] - OFFICIAL[1]:+.6f}); tolerance {TOL}")

    out("\n== B. By step range: rows and |target| (model-independent) ==")
    out(f"  {'steps':<12} {'rows':>11} {'scored%':>8} {'|y| sc t0':>10} {'|y| un t0':>10} {'|y| sc t1':>10} "
        f"{'|y| un t1':>10} {'clip|y| sc':>10} {'clip|y| un':>10}")
    for ri, rn in enumerate(RANGES):
        ns, na, nu = rows[ri]
        m_sc = abs_raw[ri, 0] / max(ns, 1)
        m_un = abs_raw[ri, 2] / max(nu, 1)
        c_sc = abs_clip[ri, 0].mean() / max(ns, 1)
        c_un = abs_clip[ri, 2].mean() / max(nu, 1)
        out(f"  {RANGE_NAMES[ri]:<12} {na:>11,} {ns / na:>8.2%} {m_sc[0]:>10.4f} {m_un[0]:>10.4f} {m_sc[1]:>10.4f} "
            f"{m_un[1]:>10.4f} {c_sc:>10.4f} {c_un:>10.4f}")
    out("  |y| = mean raw |target|; clip|y| = mean |clipped target| averaged over t0, t1 (the metric weight per row)")

    out("\n== C. Score by step range (mean of t0, t1; t0/t1 in brackets for 'all') ==")
    out(f"  {'model':<5} {'steps':<12} {'all':>8} {'(t0':>8} {'t1)':>8} {'scored':>8} {'unscored':>9} {'un-sc':>8}")
    for name, S in all_S.items():
        r = wp_from_stats(S.sum(0))
        for ri in range(len(RANGES)):
            out(f"  {name:<5} {RANGE_NAMES[ri]:<12} {r[ri, 1, 2]:>8.4f} {r[ri, 1, 0]:>8.4f} {r[ri, 1, 1]:>8.4f} "
                f"{r[ri, 0, 2]:>8.4f} {r[ri, 2, 2]:>9.4f} {r[ri, 2, 2] - r[ri, 0, 2]:>+8.4f}")

    out("\n== D. Sequence-level noise on the scored-vs-unscored gap (1,000 bootstrap resamples of sequences) ==")
    rng = np.random.default_rng(0)
    n = len(next(iter(all_S.values())))
    idx = rng.integers(0, n, size=(1000, n))
    for name, S in all_S.items():
        S0, g = S[:, 0], []  # whole range only: (n, n_sets, 2, 6)
        for b in range(0, 1000, 100):
            rb = wp_from_stats(S0[idx[b:b + 100]].sum(1))  # (100, n_sets, 3)
            g.append(rb[:, 2, 2] - rb[:, 0, 2])
        g = np.concatenate(g)
        lo, hi = np.percentile(g, [2.5, 97.5])
        out(f"  {name}: unscored - scored = {g.mean():+.4f}, bootstrap SD {g.std(ddof=1):.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    d = wp_from_stats(all_S["v4"].sum(0))[0, :, 2] - wp_from_stats(all_S["v2"].sum(0))[0, :, 2]
    out(f"  v4 - v2: scored {d[0]:+.4f}, all {d[1]:+.4f}, unscored {d[2]:+.4f}")

    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


def main():
    for p in [STATS_OUT, RESULTS_OUT]:
        if os.path.exists(p):
            sys.exit(f"{p} exists; not overwriting")
    import torch
    import train_v1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)}, memory cap {GPU_MEM_CAP_GB} GB", flush=True)
    valid = train_v1.load_split("valid")

    all_S = {}
    for name, path in CKPTS.items():
        all_S[name] = run_ckpt(path, valid, device)
        if name == "v2":
            r = wp_from_stats(all_S[name].sum(0))[0, 0]
            print(f"v2 scored: wp {r[2]:.6f} (t0 {r[0]:.6f}, t1 {r[1]:.6f}) vs official {OFFICIAL[2]:.6f}", flush=True)
            if abs(r[2] - OFFICIAL[2]) > TOL:
                np.savez(STATS_OUT.replace(".npz", "_v2_FAILED.npz"), v2=all_S[name])
                sys.exit(f"STOP: v2 scored differs from official by {r[2] - OFFICIAL[2]:+.6f} > {TOL}")
    rows, abs_raw, abs_clip = target_stats(valid)
    np.savez(STATS_OUT, **{f"stats_{k}": v for k, v in all_S.items()}, rows=rows, abs_raw=abs_raw, abs_clip=abs_clip,
             ranges=np.array(RANGES), range_names=RANGE_NAMES, sets=SETS, stat_names=STAT_NAMES,
             ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)
    report(all_S, rows, abs_raw, abs_clip)


if __name__ == "__main__":
    main()
