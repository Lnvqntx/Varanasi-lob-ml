"""v32 (train_multi.py loss=pearson / mix): scores before and after one global output scale, on all 1,873 valid sequences.

A Pearson-trained model's outputs have an arbitrary scale (the loss is scale- and shift-free), which matters for the pair
ensemble (an average of two models) and the int8 export. Per model, one scale is fitted on valid:
    a* = sum(w * y * o) / sum(w * o^2),  w = |y|, y clipped to [-2, 2], o = the model's raw output (not clipped),
over the scored rows (is_scored & step >= 99), the metric's weights and rows. Scores, as the v18c control's final eval
(global weighted Pearson, clipped predictions): scored rows and all rows >= 99, before (clip(o)) and after (clip(a* o))
scaling. Pearson is scale-free, so scaling changes a score only through the clip at +-2; the clip-active row share is
reported before and after. a* is fitted on the same valid rows it is scored on (one scalar per model).
v18c (wMSE) is included as the reference: its a* shows how far an MSE model is from the least-squares scale.
Predictions as train_multi.evaluate_multi (train_multi.load_model, the checkpoint's input stats, 1000-step chunks with
the state carried). ONE pass over valid, GPU 0 (6 GB cap), batches of 25; raw outputs of rows >= 99 kept in RAM
(float32, about 0.3 GB per model). Sanity (tolerance 0.00001): every model's unscaled scores against its checkpoint's
final eval (valid / valid_all); any failure: stop, no results written.
Outputs (new files only; .tmp, then renamed): pearson_scale_eval_results.txt (repo), ~/work/pearson_scale_eval_stats.npz.
--smoke: the vmulti12 smoke checkpoints on the first --n-seq valid sequences, outputs with a _smoke suffix, no checkpoint
sanity (the smoke runs evaluated a different subset).

  nohup nice -n 10 python pearson_scale_eval.py > ~/work/pearson_scale_eval.log 2>&1 &
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
from spread_v2 import GPU_MEM_CAP_GB, seq_stats, wp_from_stats
from train_v5m import SEQ_LEN, WARMUP, Prefetcher, build_inputs, load_split

os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # spread_v2 sets "0" on import

CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v18c": "train_v18c_wd05_nodrop_final.pt", "v32a": "train_v32a_pearson_final.pt",
         "v32b": "train_v32b_mix_final.pt", "v32c": "train_v32c_pearson_alpha0_final.pt"}
SMOKE = {"v18c": "train_v18c_wd05_nodrop_final.pt",
         **{f"v32{k}": f"vmulti12_v32{k}_smoke_final.pt" for k in "abc"}}
BATCH = 25
CHUNK = 1000
TOL = 0.00001


@torch.no_grad()
def predict(model, ck, b, device):
    """(B, T, 2) float32 raw outputs (not clipped) of one batch, as train_multi.evaluate_multi."""
    x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), ck["stats"])
    h, preds = None, []
    for s in range(0, SEQ_LEN, CHUNK):
        p, h = model(x[:, s:s + CHUNK], h)
        preds.append(p[:, :, :2])
    return torch.cat(preds, 1).float().cpu().numpy()


def scores(Y, P, M, scale):
    """[scored, all >= 99] x [t0, t1, mean] global weighted Pearson of clip(scale * P), merged per-sequence stats."""
    S = np.zeros((2, 2, 6))
    full = np.ones(Y.shape[1], bool)
    for j in range(len(Y)):
        p = P[j] * np.float32(scale)
        S[0] += seq_stats(Y[j], p, M[j])
        S[1] += seq_stats(Y[j], p, full)
    return wp_from_stats(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-seq", type=int, default=50, help="--smoke: first N valid sequences")
    args = ap.parse_args()
    sfx = f"_smoke{args.n_seq}" if args.smoke else ""
    results_out = f"/home/jovyan/work/varanasi-lob-ml/pearson_scale_eval{sfx}_results.txt"
    stats_out = f"/home/jovyan/work/pearson_scale_eval{sfx}_stats.npz"
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

    names = list(paths)
    loaded = {k: tm.load_model(v, device) for k, v in paths.items()}
    valid = load_split("valid")
    n_all = valid["prices"].shape[0]
    n = min(args.n_seq, n_all) if args.smoke else n_all
    R = SEQ_LEN - WARMUP  # rows >= 99
    Y = np.zeros((n, R, 2), np.float32)
    M = np.zeros((n, R), bool)
    P = {k: np.zeros((n, R, 2), np.float32) for k in names}
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    for bi, b in enumerate(Prefetcher(valid, batches)):
        j = batches[bi]
        Y[j] = b["targets"][:, WARMUP:]
        M[j] = b["is_scored"][:, WARMUP:]
        for k in names:
            P[k][j] = predict(*loaded[k], b, device)[:, WARMUP:]
        if (bi + 1) % 10 == 0:
            print(f"  {j[-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)

    yc = np.clip(Y[M], -2, 2).astype(np.float64)  # (rows, 2) scored rows
    w = np.abs(yc)
    res = {}
    for k in names:
        o = P[k][M].astype(np.float64)
        a = float((w * yc * o).sum() / (w * o * o).sum())  # both targets together: one global scale
        a_t = (w * yc * o).sum(0) / (w * o * o).sum(0)  # per target, reported only
        before, after = scores(Y, P[k], M, 1.0), scores(Y, P[k], M, a)
        clip_b, clip_a = np.mean(np.abs(o) > 2), np.mean(np.abs(a * o) > 2)
        wsd = np.sqrt((w * (o - (w * o).sum(0) / w.sum(0)) ** 2).sum(0) / w.sum(0))  # weighted sd of o, per target
        res[k] = dict(a=a, a_t=a_t, before=before, after=after, clip_b=clip_b, clip_a=clip_a, wsd=wsd)
        print(f"{k}: a* {a:.4f}, masked {before[0, 2]:.6f} -> {after[0, 2]:.6f}, all {before[1, 2]:.6f} -> "
              f"{after[1, 2]:.6f}", flush=True)
    ysd = np.sqrt((w * (yc - (w * yc).sum(0) / w.sum(0)) ** 2).sum(0) / w.sum(0))

    checks = []
    if not args.smoke:
        for k in names:
            ck = loaded[k][1]
            checks.append((f"{k} masked vs checkpoint", res[k]["before"][0, 2], ck["valid"]["weighted_pearson"]))
            checks.append((f"{k} all rows vs checkpoint", res[k]["before"][1, 2], ck["valid_all"]["weighted_pearson"]))
    bad = []
    for name, r, rf in checks:
        print(f"sanity {name}: {r:.7f} vs {rf:.7f} | diff {r - rf:+.2e}", flush=True)
        if not abs(r - rf) <= TOL:
            bad.append(name)
    if bad:
        sys.exit(f"STOP: sanity off by more than {TOL} for {bad}; no results written")

    np.savez(stats_out + ".tmp.npz", models=names, ckpts=[paths[k] for k in names],
             a=[res[k]["a"] for k in names], a_t=np.array([res[k]["a_t"] for k in names]),
             before=np.array([res[k]["before"] for k in names]), after=np.array([res[k]["after"] for k in names]),
             clip_before=[res[k]["clip_b"] for k in names], clip_after=[res[k]["clip_a"] for k in names], n_seq=n)
    os.rename(stats_out + ".tmp.npz", stats_out)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"v32 loss=pearson / mix: one global output scale, valid ({n} sequences{', SMOKE' if args.smoke else ''})")
    out("a* = sum(w y o) / sum(w o^2) on the scored rows (is_scored & step >= 99), w = |y|, y clipped, o raw; "
        "scores = global weighted Pearson of the clipped (scaled) prediction")
    for k in names:
        a = loaded[k][1]["args"]
        out(f"  {k:<5} {paths[k]} (step {loaded[k][1]['step']}; loss {a.get('loss', 'mse')}, alpha {a['alpha']}, "
            f"init seed {a['seed']}, data seed {a.get('data_seed')})")
    if checks:
        out(f"\nSanity (tolerance {TOL}): all {len(checks)} checks pass")
        for name, r, rf in checks:
            out(f"  {name}: {r:.7f} vs {rf:.7f} ({r - rf:+.1e})")
    v = res["v18c"]
    out(f"\n{'model':<6} {'a*':>7} {'a* t0/t1':>15} {'scored before':>14} {'after':>16} {'all>=99 before':>15} "
        f"{'after':>16} {'|o|>2 before/after':>19}   weighted sd of o (t0/t1; y {ysd[0]:.3f}/{ysd[1]:.3f})")
    for k in names:
        r = res[k]
        d = "" if k == "v18c" else f" | vs v18c: scored {r['after'][0, 2] - v['before'][0, 2]:+.4f}, all " \
                                   f"{r['after'][1, 2] - v['before'][1, 2]:+.4f} (after scaling)"
        out(f"{k:<6} {r['a']:7.4f} {r['a_t'][0]:7.4f}/{r['a_t'][1]:<7.4f} {r['before'][0, 2]:14.4f} "
            f"{r['after'][0, 2]:.4f} ({r['after'][0, 2] - r['before'][0, 2]:+.4f}) {r['before'][1, 2]:15.4f} "
            f"{r['after'][1, 2]:.4f} ({r['after'][1, 2] - r['before'][1, 2]:+.4f}) "
            f"{r['clip_b']:9.2%}/{r['clip_a']:<9.2%}   {r['wsd'][0]:.3f}/{r['wsd'][1]:.3f}{d}")
    out("\nt0 / t1 (scored rows, before -> after):")
    for k in names:
        r = res[k]
        out(f"  {k:<6} t0 {r['before'][0, 0]:.4f} -> {r['after'][0, 0]:.4f}, t1 {r['before'][0, 1]:.4f} -> "
            f"{r['after'][0, 1]:.4f}")
    with open(results_out + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.rename(results_out + ".tmp", results_out)
    print(f"\nsaved {results_out} and {stats_out}", flush=True)


if __name__ == "__main__":
    main()
