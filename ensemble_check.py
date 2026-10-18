"""Ensemble check on valid: does averaging several GRU checkpoints beat a single one?

One pass over all 1,873 valid sequences (GPU 0, 6 GB cap, batches of 25, 1000-step chunks, as
spread_v2.py / mask_check.py). Each model runs once per batch; for every non-empty subset of models the
raw outputs are averaged, then clipped to [-2, 2] by the metric (spread_v2.seq_stats). Per-sequence
float64 sufficient statistics (w = |clipped y|) on two row sets:
  masked   is_scored & step >= 99   (official metric)
  all      step >= 99
Single-model masked scores must match earlier results within 0.0005, otherwise no results are written.

Usage:
  nohup python ensemble_check.py > ~/work/ensemble_check.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import itertools
import sys
import time

import numpy as np

from spread_v2 import GPU_MEM_CAP_GB, OFFICIAL, STAT_NAMES, WARMUP, seq_stats, wp_from_stats

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v2": f"{CKPT_DIR}/train_v2_best.pt",
         "v4": f"{CKPT_DIR}/train_v4_h128_e5_fullval_best.pt",
         "v5": f"{CKPT_DIR}/train_v5_h128_e5_seed1_best.pt"}  # trained with --n-val 1873 despite the name
# masked full-valid references: v2 official int8 scorer run; v4/v5 the scores logged at their saved step
REFS = {"v2": OFFICIAL[2], "v4": 0.6787872333938094, "v5": 0.6801017775278928}
STATS_OUT = "/home/jovyan/work/ensemble_check_stats.npz"
RESULTS_OUT = "/home/jovyan/work/ensemble_check_results.txt"
SETS = ["masked", "all"]
TOL = 0.0005
BATCH, CHUNK = 25, 1000


def load_model(path, device):
    import torch
    import train_v1
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = train_v1.GRUModel(hidden=ck["args"]["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    print(f"{path}: step {ck['step']}, hidden {ck['args']['hidden']}, seed {ck['args']['seed']}, "
          f"n_val {ck['args']['n_val']}", flush=True)
    return model, stats


def predict(model, stats, b, device):
    import torch
    import train_v1
    x = train_v1.build_inputs(torch.from_numpy(b["prices"]).to(device),
                              torch.from_numpy(b["features"]).to(device), stats)
    h, preds = None, []
    for s in range(0, x.shape[1], CHUNK):
        p, h = model(x[:, s:s + CHUNK], h)
        preds.append(p)
    return torch.cat(preds, 1).float().cpu().numpy()


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

    names = list(CKPTS)
    models = {k: load_model(v, device) for k, v in CKPTS.items()}
    subsets = [c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    need = np.arange(T) >= WARMUP
    S = np.zeros((len(subsets), n, len(SETS), 2, 6), dtype=np.float64)
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            preds = {k: predict(m, st, b, device) for k, (m, st) in models.items()}
            for si, sub in enumerate(subsets):
                avg = np.mean([preds[k] for k in sub], axis=0, dtype=np.float32)
                for k, i in enumerate(batches[bi]):
                    S[si, i, 0] = seq_stats(b["targets"][k], avg[k], b["is_scored"][k] & need)
                    S[si, i, 1] = seq_stats(b["targets"][k], avg[k], need)
            if (bi + 1) % 15 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True)
    sub_names = ["+".join(s) for s in subsets]
    np.savez(STATS_OUT, stats=S, subsets=sub_names, sets=SETS, stat_names=STAT_NAMES,
             ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)

    R = wp_from_stats(S.sum(1))  # (n_subsets, n_sets, 3)
    bad = []
    for k in names:
        r = R[sub_names.index(k), 0, 2]
        print(f"sanity {k}: masked {r:.6f} vs reference {REFS[k]:.6f} | diff {r - REFS[k]:+.6f}", flush=True)
        if abs(r - REFS[k]) > TOL:
            bad.append(k)
    if bad:
        sys.exit(f"STOP: single-model masked score off by more than {TOL} for {bad}; no results written")
    report(subsets, sub_names, R, names)


def report(subsets, sub_names, R, names):
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    single = {k: R[sub_names.index(k), 0, 2] for k in names}
    out("Ensemble check on valid (1,873 sequences, steps >= 99), fp32 PyTorch, GPU 0")
    out("Simple average of raw model outputs, then clipped to [-2, 2]; global weighted Pearson, weights |clipped target|.")
    for k, v in CKPTS.items():
        out(f"  {k}: {v}")
    out("  v2 picked its checkpoint on 300 valid sequences; v4, v5 on all 1,873 (masked rows), so their masked")
    out("  scores are slightly optimistic. v6 not included (no checkpoint, no finished run).")
    out("\nSingle-model sanity vs earlier results (tolerance 0.0005):")
    for k in names:
        out(f"  {k}: {single[k]:.6f} vs {REFS[k]:.6f} ({single[k] - REFS[k]:+.6f})")
    out("\nSorted by masked mean; gain = masked mean - best single member's masked mean")
    out(f"  {'subset':<10} {'n':>2} {'masked t0':>10} {'masked t1':>10} {'masked':>8} {'all rows':>9} "
        f"{'gain':>8} {'gain all':>9}")
    order = np.argsort(-R[:, 0, 2])
    for si in order:
        sub = subsets[si]
        best = max(sub, key=lambda k: single[k])
        best_all = max(R[sub_names.index(k), 1, 2] for k in sub)
        gain = R[si, 0, 2] - single[best] if len(sub) > 1 else float("nan")
        gain_all = R[si, 1, 2] - best_all if len(sub) > 1 else float("nan")
        g = f"{gain:>+8.4f}" if len(sub) > 1 else f"{'-':>8}"
        ga = f"{gain_all:>+9.4f}" if len(sub) > 1 else f"{'-':>9}"
        out(f"  {sub_names[si]:<10} {len(sub):>2} {R[si, 0, 0]:>10.4f} {R[si, 0, 1]:>10.4f} {R[si, 0, 2]:>8.4f} "
            f"{R[si, 1, 2]:>9.4f} {g} {ga}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
