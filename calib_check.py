"""Does scaling the v7+v9 ensemble prediction by the mask predictor's p(scored) help the metric?

1. One pass over all 1,873 valid sequences with v7 and v9 (best checkpoints, fp32, GPU 0, batches of 25,
   1000-step chunks, as ensemble_check3.py). The average of the raw outputs on steps >= 99 is saved as
   float16 ~/work/calib_v7v9_pred_t0.npy and _t1.npy, shape (1873, 19901). If both exist they are
   reused, never overwritten.
2. numpy, on the 469 held-out valid sequences (index % 4 == 0, never seen by the mask GRU):
     power   pred * (p / mean_p) ** gamma          gamma in 0, 0.1, 0.25, 0.5, 1.0  (gamma 0 = unscaled)
     linear  pred * (a + (1 - a) * p / mean_p)     a in 0.25, 0.5, 0.75
   p = ~/work/cache/valid_maskp.npy (probabilities); mean_p = mean p over the held-out rows with step >= 99.
   Clipped to [-2, 2] after scaling (spread_v2.seq_stats); official weighted Pearson on is_scored &
   step >= 99 and on all rows step >= 99, mean of t0 and t1.
3. Robustness: the held-out sequences split by even / odd position; the setting with the best scored-row
   score on one half is reported on the other, in both directions.

  nohup python calib_check.py > ~/work/calib_check.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time

import numpy as np

from ensemble_check2 import BATCH, load_model, predict
from spread_v2 import GPU_MEM_CAP_GB, WARMUP, seq_stats, wp_from_stats

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = [f"{CKPT_DIR}/train_v7_maskw_a05_h128_e5_fullval_best.pt",
         f"{CKPT_DIR}/train_v9_maskw_a05_h128_e5_seed1_best.pt"]
PRED_OUT = [f"/home/jovyan/work/calib_v7v9_pred_t{j}.npy" for j in (0, 1)]
MASKP = "/home/jovyan/work/cache/valid_maskp.npy"
RESULTS_OUT = "/home/jovyan/work/calib_check_results.txt"
REF_V7V9 = {"all": 0.6850, "held": 0.6915}  # ensemble_check4 (fp32), masked, 4 decimals
SETTINGS = [("power", g) for g in (0.0, 0.1, 0.25, 0.5, 1.0)] + [("linear", a) for a in (0.25, 0.5, 0.75)]


def label(kind, v):
    if kind == "power":
        return "none (gamma 0)" if v == 0 else f"(p/mean_p)^{v:g}"
    return f"{v:g} + {1 - v:g} * p/mean_p"


def make_preds():
    import torch
    import train_v1
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    models = [load_model(p, device) for p in CKPTS]
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    tmp = [p + ".tmp" for p in PRED_OUT]
    out = [np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(n, T - WARMUP)) for p in tmp]
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            avg = np.mean([predict(m, st, b, device) for m, st in models], axis=0, dtype=np.float32)
            for j in (0, 1):
                out[j][batches[bi]] = avg[:, WARMUP:, j].astype(np.float16)
            if (bi + 1) % 15 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    for o, tp, p in zip(out, tmp, PRED_OUT):
        o.flush()
        del o
        os.rename(tp, p)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; "
          f"saved {PRED_OUT}", flush=True)


def main():
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    if not all(os.path.exists(p) for p in PRED_OUT):
        assert not any(os.path.exists(p) for p in PRED_OUT), "only one prediction file exists"
        make_preds()
    else:
        print(f"reusing {PRED_OUT}", flush=True)
    import train_v1
    valid = train_v1.load_split("valid")
    n = valid["targets"].shape[0]
    pred = np.stack([np.load(p, mmap_mode="r") for p in PRED_OUT], axis=-1)  # (n, T-99, 2) f16
    maskp = np.load(MASKP, mmap_mode="r")
    held = np.arange(n)[np.arange(n) % 4 == 0]

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    # 1. is valid_maskp probabilities?
    mp_all = np.asarray(maskp[:, WARMUP:], dtype=np.float32)
    sc_all = np.asarray(valid["is_scored"][:, WARMUP:])
    out("valid_maskp.npy check (steps >= 99, all 1,873 sequences):")
    out(f"  dtype {maskp.dtype}, min {mp_all.min():.5f}, max {mp_all.max():.5f}, mean {mp_all.mean():.5f}; "
        f"true is_scored rate {sc_all.mean():.5f}")
    out(f"  mean p on scored rows {mp_all[sc_all].mean():.4f}, on unscored rows {mp_all[~sc_all].mean():.4f}")
    out("  -> values lie in [0, 1] and their mean matches the scored rate: sigmoid probabilities from mask_gru.py,")
    out("     not training weights (a weight m = a + (1-a) p/mean_p would have mean about 1 and exceed 1).")
    del mp_all, sc_all

    # 2. per-sequence stats for every setting on the held-out sequences
    mean_p = float(np.mean([maskp[i, WARMUP:].astype(np.float64).mean() for i in held]))
    out(f"\nmean_p over the {len(held)} held-out sequences, steps >= 99: {mean_p:.5f}")
    S = np.zeros((len(SETTINGS), len(held), 2, 2, 6))  # setting, seq, [scored, all], target, stat
    S_full0 = np.zeros((n, 2, 6))
    for k, i in enumerate(held):
        y = np.asarray(valid["targets"][i, WARMUP:])
        sc = np.asarray(valid["is_scored"][i, WARMUP:])
        allr = np.ones_like(sc)
        pr = np.asarray(pred[i], dtype=np.float32)
        r = np.asarray(maskp[i, WARMUP:], dtype=np.float32) / mean_p
        for si, (kind, v) in enumerate(SETTINGS):
            f = r ** v if kind == "power" else v + (1 - v) * r
            ps = pr * f[:, None]
            S[si, k, 0] = seq_stats(y, ps, sc)
            S[si, k, 1] = seq_stats(y, ps, allr)
    for i in range(n):  # sanity: unscaled float16 ensemble on all 1,873 vs the fp32 ensemble_check4 result
        S_full0[i] = seq_stats(np.asarray(valid["targets"][i, WARMUP:]), np.asarray(pred[i], dtype=np.float32),
                               np.asarray(valid["is_scored"][i, WARMUP:]))
    full0 = wp_from_stats(S_full0.sum(0))[2]
    R = wp_from_stats(S.sum(1))  # (settings, 2 sets, 3)
    out(f"sanity, unscaled float16 v7+v9: all 1,873 masked {full0:.4f} (fp32 ensemble_check4 {REF_V7V9['all']:.4f}), "
        f"held out {R[0, 0, 2]:.4f} (fp32 {REF_V7V9['held']:.4f})")

    out(f"\nHeld-out {len(held)} sequences; gain vs unscaled")
    out(f"  {'scaling':<24} {'scored t0':>9} {'scored t1':>9} {'scored':>8} {'gain':>8} {'all>=99':>8} {'gain':>8}")
    for si, (kind, v) in enumerate(SETTINGS):
        out(f"  {label(kind, v):<24} {R[si, 0, 0]:>9.4f} {R[si, 0, 1]:>9.4f} {R[si, 0, 2]:>8.4f} "
            f"{R[si, 0, 2] - R[0, 0, 2]:>+8.4f} {R[si, 1, 2]:>8.4f} {R[si, 1, 2] - R[0, 1, 2]:>+8.4f}")

    # 3. even / odd halves of the held-out list
    halves = {"even": np.arange(len(held)) % 2 == 0, "odd": np.arange(len(held)) % 2 == 1}
    H = {h: wp_from_stats(S[:, m].sum(1)) for h, m in halves.items()}
    out("\nRobustness: pick the best scored-row setting on one half, report it on the other")
    out(f"  {'pick on':<8} {'setting':<24} {'gain there':>10} | {'report on':<9} {'scored':>8} {'gain':>8} "
        f"{'all>=99':>8} {'gain':>8}")
    gains = []
    for a, b in [("even", "odd"), ("odd", "even")]:
        si = int(np.argmax(H[a][:, 0, 2]))
        g = H[b][si, 0, 2] - H[b][0, 0, 2]
        gains.append(g)
        out(f"  {a:<8} {label(*SETTINGS[si]):<24} {H[a][si, 0, 2] - H[a][0, 0, 2]:>+10.4f} | {b:<9} "
            f"{H[b][si, 0, 2]:>8.4f} {g:>+8.4f} {H[b][si, 1, 2]:>8.4f} {H[b][si, 1, 2] - H[b][0, 1, 2]:>+8.4f}")
    verdict = "PROMISING" if min(gains) >= 0.002 else "NO GAIN"
    out(f"\nVerdict (promising only if the out-of-half gain is >= 0.002 in both directions): {verdict}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
