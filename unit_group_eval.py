"""Reference in-group scores of the generalists on the coarse-unit valid sequences (unit 0.656868, 697 of 1,873).

The question for v21 (coarse-unit specialists, train_multi.py --unit-filter 0.6569) is whether a model trained only
on that unit beats the generalists on the same 697 sequences. This script gives the generalists' in-group scores.
Models: final checkpoints of v18c (wd 0.05, data seed 0), v19a (wd 0.1, data seed 1), v20a (wd 0.05 + EMA 0.9995,
data seed 0), and the v18c+v19a pair (current best submission; raw outputs averaged, then clipped by the metric).
Unit labels: unit_labels.py (unit_probe.py's labelling, cached in ~/work/cache/unit_labels_valid.npy).
Method as ensemble_check9.py: GPU 0, 6 GB cap, batches of 25, 1000-step chunks, ONE pass over all 1,873 valid
sequences (so the full-valid scores can be checked against the checkpoints), per-sequence float64 sufficient
statistics on is_scored & step >= 99 ("masked") and on all steps >= 99 ("all"); the global weighted Pearson of a
group is computed from the summed statistics of its sequences, exactly as the trainer's evaluation on that group.
Reported per model: in-group t0, t1, mean on the masked and the all-rows sets; the same for each of the four units;
the 469 held-out sequences (index % 4 == 0) of the coarse unit. Per-sequence statistics are saved to
~/work/unit_group_eval_stats.npz (new file) so a specialist can later be combined with a generalist offline.
Sanity (tolerance 0.00001): full-valid masked and all-rows scores of each model against its checkpoint, and v18c,
v19a, v18c+v19a against ensemble_check9_stats.npz (full and held-out, both sets).

  nohup nice -n 10 python unit_group_eval.py > ~/work/unit_group_eval.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import sys
import time

import numpy as np

from ensemble_check2 import BATCH, predict
from ensemble_check7 import load_gru
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, WARMUP, seq_stats, wp_from_stats
from unit_labels import unit_index, unit_labels
from unit_probe import UNITS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ensemble_check7 sets "1" on import

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v18c": f"{CKPT_DIR}/train_v18c_wd05_nodrop_final.pt",
         "v19a": f"{CKPT_DIR}/train_v19a_wd10_ds1_final.pt",
         "v20a": f"{CKPT_DIR}/train_v20a_wd05_ema_final.pt"}
NAMES = list(CKPTS)
SUBSETS = [(k,) for k in NAMES] + [("v18c", "v19a")]
GROUP = unit_index(0.6569)
CHECK9 = "/home/jovyan/work/ensemble_check9_stats.npz"
STATS_OUT = "/home/jovyan/work/unit_group_eval_stats.npz"
RESULTS_OUT = "/home/jovyan/work/varanasi-lob-ml/unit_group_eval_results.txt"
SETS = ["masked", "all"]
TOL = 0.00001


def main():
    if os.path.exists(RESULTS_OUT) or os.path.exists(STATS_OUT):
        sys.exit(f"{RESULTS_OUT} or {STATS_OUT} exists; not overwriting")
    missing = [v for v in CKPTS.values() if not os.path.exists(v)]
    if missing:
        sys.exit(f"missing checkpoints: {missing}")
    import torch
    import train_v1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0" and not torch.cuda.is_initialized()
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)} (GPU 0), memory cap {GPU_MEM_CAP_GB} GB", flush=True)

    unit = unit_labels("valid")
    models = {k: load_gru(v, device) for k, v in CKPTS.items()}
    cks = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in CKPTS.items()}
    valid = train_v1.load_split("valid")
    n, T = valid["prices"].shape[:2]
    assert len(unit) == n
    is_held = np.arange(n) % 4 == 0
    need = np.arange(T) >= WARMUP
    S = np.zeros((len(SUBSETS), n, len(SETS), 2, 6), dtype=np.float64)
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(train_v1.Prefetcher(valid, batches)):
            preds = {k: predict(mod, st, b, device) for k, (mod, st) in models.items()}
            for si, sub in enumerate(SUBSETS):
                avg = np.mean([preds[k] for k in sub], axis=0, dtype=np.float32)
                for k, j in enumerate(batches[bi]):
                    S[si, j, 0] = seq_stats(b["targets"][k], avg[k], b["is_scored"][k] & need)
                    S[si, j, 1] = seq_stats(b["targets"][k], avg[k], need)
            if (bi + 1) % 10 == 0:
                print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    sub_names = ["+".join(s) for s in SUBSETS]
    np.savez(STATS_OUT, stats=S, subsets=sub_names, sets=SETS, stat_names=STAT_NAMES, unit=unit, units=UNITS,
             held=np.flatnonzero(is_held), ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)

    def score(sel):
        return wp_from_stats(S[:, sel].sum(1))  # (n_subsets, n_sets, 3) [t0, t1, mean]

    R_full = score(np.ones(n, bool))
    in_group = unit == GROUP
    R_grp, R_grp_held = score(in_group), score(in_group & is_held)
    idx = {s: i for i, s in enumerate(sub_names)}
    checks = []
    for k, ck in cks.items():
        checks.append((f"{k} full masked vs checkpoint", R_full[idx[k], 0, 2], ck["valid"]["weighted_pearson"]))
        checks.append((f"{k} full all rows vs checkpoint", R_full[idx[k], 1, 2], ck["valid_all"]["weighted_pearson"]))
    c = np.load(CHECK9)
    assert np.array_equal(c["held"], np.flatnonzero(is_held)) and c["stats"].shape[1] == n
    R9 = {"full": wp_from_stats(c["stats"].sum(1)), "held": wp_from_stats(c["stats"][:, c["held"]].sum(1))}
    R_held = score(is_held)
    for ci, name in enumerate(c["subsets"]):
        if name in idx:
            for part, R in [("full", R_full), ("held", R_held)]:
                for s, set_name in enumerate(SETS):
                    checks.append((f"{name} {part} {set_name} vs ensemble_check9", R[idx[name], s, 2], R9[part][ci, s, 2]))
    bad = []
    for name, r, ref in checks:
        print(f"sanity {name}: {r:.7f} vs {ref:.7f} | diff {r - ref:+.2e}", flush=True)
        if abs(r - ref) > TOL:
            bad.append(name)
    if bad:
        sys.exit(f"STOP: sanity off by more than {TOL} for {bad}; no results written")

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"Generalists on the coarse-unit valid sequences (unit {UNITS[GROUP]}), fp32 PyTorch, GPU 0, steps >= 99")
    out("Global weighted Pearson (weights |clipped target|) over the group's rows only; pair = average of raw outputs.")
    for k, v in CKPTS.items():
        a = cks[k]["args"]
        out(f"  {k:<5} {v} (step {cks[k]['step']}; data seed {a.get('data_seed', a['seed'])}, init seed {a['seed']}, "
            f"wd {a.get('wd')}, ema {a.get('ema', 0)})")
    out(f"  units (unit_labels.py): " + ", ".join(f"{u} {np.sum(unit == k)}" for k, u in enumerate(UNITS))
        + f"; coarse group {in_group.sum()} sequences, {np.sum(in_group & is_held)} of them in the 469 held out")
    out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
    for name, r, ref in checks:
        out(f"  {name}: {r:.7f} vs {ref:.7f} ({r - ref:+.1e})")

    out(f"\nIn-group score on the {in_group.sum()} coarse-unit sequences (unit {UNITS[GROUP]})")
    out(f"  {'model':<10} {'masked t0':>9} {'t1':>7} {'mean':>7}   {'all>=99 t0':>10} {'t1':>7} {'mean':>7}   "
        f"{'469 masked':>10} {'full masked':>11}")
    for name in sub_names:
        i = idx[name]
        out(f"  {name:<10} {R_grp[i, 0, 0]:>9.4f} {R_grp[i, 0, 1]:>7.4f} {R_grp[i, 0, 2]:>7.4f}   "
            f"{R_grp[i, 1, 0]:>10.4f} {R_grp[i, 1, 1]:>7.4f} {R_grp[i, 1, 2]:>7.4f}   "
            f"{R_grp_held[i, 0, 2]:>10.4f} {R_full[i, 0, 2]:>11.4f}")

    out("\nPer unit, masked mean / all>=99 mean (sequences)")
    out(f"  {'model':<10} " + " ".join(f"{u:>17}" for u in UNITS))
    out(f"  {'':<10} " + " ".join(f"{'(' + str(int(np.sum(unit == k))) + ')':>17}" for k in range(len(UNITS))))
    for name in sub_names:
        i = idx[name]
        cells = []
        for k in range(len(UNITS)):
            R = score(unit == k)
            cells.append(f"{R[i, 0, 2]:.4f} / {R[i, 1, 2]:.4f}")
        out(f"  {name:<10} " + " ".join(f"{c_:>17}" for c_ in cells))
    w = S[idx["v18c+v19a"], :, 0, :, 0].sum(1)  # metric weight per sequence (masked, both targets)
    out(f"\nShare of the masked metric weight in the coarse group: {w[in_group].sum() / w.sum():.1%}")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
