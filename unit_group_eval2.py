"""In-group scores of the v31 models (train_multi.py ynorm= / unitemb=) within each of the four unit groups on valid.

For every v31 final checkpoint (and v18c, the control) this reports the global weighted Pearson on the scored rows
(is_scored & step >= 99) and on all rows >= 99 within each unit group (0.125433 / 0.241330 / 0.468536 / 0.656868),
plus the overall score on all 1,873 valid sequences. The coarse group's reference is v18c 0.4334
(unit_group_eval.py). Groups are the true units (unit_labels.py, the same grouping as unit_group_eval.py); the
models themselves only see the causally predicted class (unit_pred.py, 100% correct on valid).
Predictions as train_multi.evaluate_multi: train_multi.load_model, each checkpoint's input stats, 1000-step chunks
with the state carried, the unit embedding (unitemb=) of the predicted class ("unknown" before step 20) appended
last, and for ynorm= models the prediction clip(o * u, -2, 2) with u the predicted class's unit (1 before step 20).
ONE pass over all 1,873 valid sequences, GPU 0 (6 GB cap), batches of 25; per-sequence float64 sufficient
statistics as unit_group_eval.py, so a group's score is computed from the summed statistics of its sequences.
Sanity (tolerance 0.00001): each model's full-valid masked and all-rows scores against its checkpoint; v18c's
per-sequence statistics against ~/work/unit_group_eval_stats.npz (every unit group, both row sets). Any failure:
stop, no results written.
Outputs (new files only; written as .tmp, then renamed): unit_group_eval2_results.txt (repo) and
~/work/unit_group_eval2_stats.npz. --smoke: the vmulti11 smoke checkpoints on the first --n-seq valid sequences,
outputs with a _smoke suffix, no checkpoint sanity (the smoke runs evaluated a different subset).

  nohup nice -n 10 python unit_group_eval2.py > ~/work/unit_group_eval2.log 2>&1 &
"""
import os

GPU = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import argparse
import sys
import time

import numpy as np
import torch

import train_multi as tm
from spread_v2 import GPU_MEM_CAP_GB, STAT_NAMES, seq_stats, wp_from_stats
from train_v5m import SEQ_LEN, WARMUP, Prefetcher, build_inputs, load_split
from unit_labels import unit_labels

os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # unit_labels / unit_probe hide the GPU on import

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v18c": "train_v18c_wd05_nodrop_final.pt",
         "v31a": "train_v31a_ynorm_a_s0_final.pt", "v31b": "train_v31b_ynorm_a_s1_final.pt",
         "v31c": "train_v31c_ynorm_b_s0_final.pt", "v31d": "train_v31d_ynorm_b_s1_final.pt",
         "v31e": "train_v31e_ynorm_a_unit8_final.pt", "v31f": "train_v31f_ynorm_b_unit8_final.pt"}
SMOKE = {"v18c": "train_v18c_wd05_nodrop_final.pt",
         **{f"v31{k}": f"vmulti11_v31{k}_smoke_final.pt" for k in "abcdef"}}
REF = "/home/jovyan/work/unit_group_eval_stats.npz"
PRED_VALID = "/home/jovyan/work/cache/unit_pred_valid.npy"  # unit_pred.pred_path("valid"); unit_pred.py hides the GPU on import
SETS = ["masked", "all"]
BATCH = 25
CHUNK = 1000
TOL = 0.00001


@torch.no_grad()
def predict(model, ck, b, cls, device):
    """(B, T, 2) float32 predictions of one batch, as train_multi.evaluate_multi."""
    x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), ck["stats"])
    emb, yn = getattr(model, "unit_emb", None), model.ynorm
    c = torch.from_numpy(cls.astype(np.int64)).to(device)
    h, preds = None, []
    for s in range(0, SEQ_LEN, CHUNK):
        xs = x[:, s:s + CHUNK]
        if emb is not None:
            xs = torch.cat([xs, tm.unit_input(emb, c, s, s + xs.shape[1], ck["unit"]["pred"])], 2)
        p, h = model(xs, h)
        p = p[:, :, :2]
        if yn is not None:
            p = (p * tm.unit_scale(c, s, s + xs.shape[1], yn["fix"])).clamp(-2, 2)
        preds.append(p)
    return torch.cat(preds, 1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-seq", type=int, default=50, help="--smoke: first N valid sequences")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    results_out = f"/home/jovyan/work/varanasi-lob-ml/unit_group_eval2{sfx}_results.txt"
    stats_out = f"/home/jovyan/work/unit_group_eval2{sfx}_stats.npz"
    if os.path.exists(results_out) or os.path.exists(stats_out):
        sys.exit(f"{results_out} or {stats_out} exists; not overwriting")
    paths = {k: f"{CKPT_DIR}/{v}" for k, v in (SMOKE if args.smoke else CKPTS).items()}
    missing = [v for v in paths.values() if not os.path.exists(v)]
    if missing:
        sys.exit(f"missing checkpoints: {missing}")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}), "
          f"memory cap {GPU_MEM_CAP_GB} GB", flush=True)

    unit, upred = unit_labels("valid"), np.load(PRED_VALID)
    names = list(paths)
    loaded = {k: tm.load_model(v, device) for k, v in paths.items()}
    valid = load_split("valid")
    n_all, T = valid["prices"].shape[:2]
    assert len(unit) == n_all == len(upred)
    n = min(args.n_seq, n_all) if args.smoke else n_all
    need = np.arange(T) >= WARMUP
    S = np.zeros((len(names), n, len(SETS), 2, 6), dtype=np.float64)
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    for bi, b in enumerate(Prefetcher(valid, batches)):
        for mi, k in enumerate(names):
            model, ck = loaded[k]
            pr = predict(model, ck, b, upred[batches[bi]], device)
            for r, j in enumerate(batches[bi]):
                S[mi, j, 0] = seq_stats(b["targets"][r], pr[r], b["is_scored"][r] & need)
                S[mi, j, 1] = seq_stats(b["targets"][r], pr[r], need)
        if (bi + 1) % 10 == 0:
            print(f"  {batches[bi][-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    unit = unit[:n]

    def score(sel):
        return wp_from_stats(S[:, sel].sum(1))  # (n_models, n_sets, 3) [t0, t1, mean]

    R_full = score(np.ones(n, bool))
    R_grp = [score(unit == g) for g in range(len(tm.UNIT_VALUES))]
    checks = []
    if not args.smoke:
        for mi, k in enumerate(names):
            ck = loaded[k][1]
            checks.append((f"{k} full masked vs checkpoint", R_full[mi, 0, 2], ck["valid"]["weighted_pearson"]))
            checks.append((f"{k} full all rows vs checkpoint", R_full[mi, 1, 2], ck["valid_all"]["weighted_pearson"]))
    ref = np.load(REF)
    ri = list(ref["subsets"]).index("v18c")
    rs = ref["stats"][ri, :n]
    for g, u in enumerate(tm.UNIT_VALUES):
        Rr = wp_from_stats(rs[unit == g].sum(0))
        for s, set_name in enumerate(SETS):
            checks.append((f"v18c unit {u} {set_name} vs unit_group_eval", R_grp[g][0, s, 2], Rr[s, 2]))
    bad = []
    for name, r, rf in checks:
        print(f"sanity {name}: {r:.7f} vs {rf:.7f} | diff {r - rf:+.2e}", flush=True)
        if not abs(r - rf) <= TOL:
            bad.append(name)
    if bad:
        sys.exit(f"STOP: sanity off by more than {TOL} for {bad}; no results written")

    np.savez(stats_out + ".tmp.npz", stats=S, models=names, ckpts=[paths[k] for k in names], sets=SETS,
             stat_names=STAT_NAMES, unit=unit, unit_pred=upred[:n], units=tm.UNIT_VALUES)
    os.rename(stats_out + ".tmp.npz", stats_out)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"v31 models within each unit group on valid ({n} sequences{', SMOKE' if args.smoke else ''}), steps >= 99")
    out("Global weighted Pearson (weights |clipped target|) over the group's rows only; groups = true units "
        "(unit_labels.py); models use the predicted class (unit_pred.py)")
    for k in names:
        a = loaded[k][1]["args"]
        out(f"  {k:<5} {paths[k]} (step {loaded[k][1]['step']}; init seed {a['seed']}, ynorm {a.get('ynorm') or '-'}, "
            f"unitemb {a.get('unitemb', 0)})")
    out(f"  sequences per unit: " + ", ".join(f"{u} {np.sum(unit == g)}" for g, u in enumerate(tm.UNIT_VALUES))
        + f"; predicted class = true unit on {np.mean(upred[:n] == unit):.5f}")
    out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
    for name, r, rf in checks:
        out(f"  {name}: {r:.7f} vs {rf:.7f} ({r - rf:+.1e})")
    for s, set_name in enumerate(SETS):
        out(f"\n{'Scored rows (is_scored & step >= 99)' if s == 0 else 'All rows >= 99'}: mean of t0 and t1 "
            f"(difference to v18c in brackets)")
        out(f"  {'model':<6} " + " ".join(f"{('unit ' + str(u)):>17}" for u in tm.UNIT_VALUES) + f" {'overall':>17}")
        v = names.index("v18c")
        for mi, k in enumerate(names):
            cells = [(R_grp[g][mi, s, 2], R_grp[g][v, s, 2]) for g in range(len(tm.UNIT_VALUES))]
            cells.append((R_full[mi, s, 2], R_full[v, s, 2]))
            out(f"  {k:<6} " + " ".join(f"{x:.4f} ({x - x0:+.4f})" if k != "v18c" else f"{x:>17.4f}"
                                        for x, x0 in cells))
    w = S[names.index("v18c"), :, 0, :, 0].sum(1)  # masked metric weight per sequence (both targets)
    out("\nShare of the masked metric weight per unit group (v18c rows): "
        + ", ".join(f"{u} {w[unit == g].sum() / w.sum():.1%}" for g, u in enumerate(tm.UNIT_VALUES)))
    with open(results_out + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.rename(results_out + ".tmp", results_out)
    print(f"\nsaved {results_out} and {stats_out}", flush=True)


if __name__ == "__main__":
    main()
