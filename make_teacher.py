"""Teacher predictions for distillation (train_multi.py kd=): the submitted pair v18c + v19a on every train row.

Both FINAL checkpoints run in fp32 PyTorch on GPU 0 over ALL 10,607 train sequences, with train_multi.py's input
preparation (train_v5m.build_inputs with each checkpoint's own input stats) and the GRU state carried across the
1,000-step chunks of each sequence, as in training. The two raw outputs are averaged, clipped to [-2, 2] and saved
for every row (all 20,000 steps, steps < 99 included) as float16 in the train caches' row order:
  ~/work/cache/train_teacher_t0.npy, ~/work/cache/train_teacher_t1.npy   (written to .tmp, then renamed)
Per-sequence float64 sufficient statistics on all steps >= 99 (spread_v2.seq_stats; the trainer's train-holdout
metric) of v18c, v19a and the saved teacher are kept in ~/work/teacher_stats.npz.

Sanity (tolerance 1e-5): v18c on its data-seed-0 train holdout and v19a on its data-seed-1 holdout equal the
"holdout" score stored in their checkpoints.

--verify (after the main run): a fresh forward pass of the pair, one sequence at a time, on 20 train sequences
(seeded draw) must equal the saved values to within 1e-3; then the teacher's weighted Pearson against the train
targets (all rows >= 99), recomputed from the saved files, on: the data-seed-0 holdout (unseen by v18c, the
holdout of a data-seed-0 student), the data-seed-1 holdout (unseen by v19a), their intersection (unseen by both),
and 300 other train sequences (seeded draw outside both holdouts, trained on by both). Results go to
teacher_results.txt.

  nohup nice -n 10 python make_teacher.py > ~/work/make_teacher.log 2>&1 &
  nice -n 10 python make_teacher.py --verify >> ~/work/make_teacher.log 2>&1
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import argparse
import sys
import time

import numpy as np

from spread_v2 import GPU_MEM_CAP_GB, WARMUP, seq_stats, wp_from_stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE = "/home/jovyan/work/cache"
CKPT_DIR = "/home/jovyan/work/checkpoints"
CKPTS = {"v18c": f"{CKPT_DIR}/train_v18c_wd05_nodrop_final.pt",
         "v19a": f"{CKPT_DIR}/train_v19a_wd10_ds1_final.pt"}
OUT = {t: f"{CACHE}/train_teacher_t{t}.npy" for t in range(2)}
STATS_OUT = "/home/jovyan/work/teacher_stats.npz"
RESULTS_OUT = "/home/jovyan/work/varanasi-lob-ml/teacher_results.txt"
BATCH = 25
CHUNK = 1000
N_HOLDOUT = 300
N_VERIFY = 20
N_OTHER = 300
VERIFY_SEED = 20261023
TOL_CK = 1e-5
TOL_VERIFY = 1e-3


def load(device):
    import torch
    from train_v5m import GRUModel
    models = {}
    for k, path in CKPTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        assert a["arch"] == "gru" and not a["aux"] and a["dropout"] == 0 and a["n_holdout"] == N_HOLDOUT
        model = GRUModel(hidden=a["hidden"], layers=a["layers"], n_out=2).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        stats = {s: v.to(device) for s, v in ck["stats"].items()}
        print(f"{k}: {path} step {ck['step']}, hidden {a['hidden']}, layers {a['layers']}, wd {a['wd']}, "
              f"data seed {a['data_seed']}, init seed {a['seed']}", flush=True)
        models[k] = (model, stats, ck)
    return models


def forward(model, stats, b, device):
    """(B, T, 2) float32 raw outputs, the GRU state carried across the 1,000-step chunks."""
    import torch
    from train_v5m import build_inputs
    x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
    h, preds = None, []
    for s in range(0, x.shape[1], CHUNK):
        p, h = model(x[:, s:s + CHUNK], h)
        preds.append(p)
    return torch.cat(preds, 1).float().cpu().numpy()


def teacher(preds):
    return np.clip((preds["v18c"] + preds["v19a"]) / 2, -2, 2).astype(np.float16)


def holdout(data_seed, n):
    return np.sort(np.random.default_rng(data_seed).permutation(n)[:N_HOLDOUT])  # train_multi.py's draw


def setup():
    import torch
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0" and not torch.cuda.is_initialized()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    print(f"device {torch.cuda.get_device_name(0)} (GPU 0), memory cap {GPU_MEM_CAP_GB} GB", flush=True)
    return device


def make():
    import torch
    from train_v5m import Prefetcher, load_split
    for p in list(OUT.values()) + [v + ".tmp" for v in OUT.values()] + [STATS_OUT]:
        if os.path.exists(p):
            sys.exit(f"{p} exists; not overwriting")
    device = setup()
    models = load(device)
    train = load_split("train")
    n, T = train["prices"].shape[:2]
    need = np.arange(T) >= WARMUP
    out = {t: np.lib.format.open_memmap(OUT[t] + ".tmp", mode="w+", dtype=np.float16, shape=(n, T)) for t in OUT}
    S = np.zeros((3, n, 2, 6), dtype=np.float64)  # v18c, v19a, teacher (saved float16 values)
    batches = [np.arange(i, min(i + BATCH, n)) for i in range(0, n, BATCH)]
    t = time.time()
    with torch.no_grad():
        for bi, b in enumerate(Prefetcher(train, batches)):
            idx = batches[bi]
            preds = {k: forward(m, st, b, device) for k, (m, st, _) in models.items()}
            tch = teacher(preds)
            for tt in OUT:
                out[tt][idx] = tch[:, :, tt]
            for k, j in enumerate(idx):
                S[0, j] = seq_stats(b["targets"][k], preds["v18c"][k], need)
                S[1, j] = seq_stats(b["targets"][k], preds["v19a"][k], need)
                S[2, j] = seq_stats(b["targets"][k], tch[k].astype(np.float32), need)
            if (bi + 1) % 20 == 0:
                el = time.time() - t
                print(f"  {idx[-1] + 1}/{n} seqs, {el:.0f}s, eta {el / (idx[-1] + 1) * (n - idx[-1] - 1) / 60:.1f} min",
                      flush=True)
    print(f"pass {time.time() - t:.0f}s; gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
          flush=True)
    bad = []
    for k, ds_key in [("v18c", 0), ("v19a", 1)]:
        ck = models[k][2]
        ho = holdout(ck["args"]["data_seed"], n)
        r = wp_from_stats(S[ds_key, ho].sum(0))[2]
        ref = ck["holdout"]["weighted_pearson"]
        print(f"sanity {k} on its data-seed-{ck['args']['data_seed']} holdout: {r:.7f} vs checkpoint {ref:.7f} "
              f"| diff {r - ref:+.2e}", flush=True)
        if abs(r - ref) > TOL_CK:
            bad.append(k)
    for tt in OUT:
        out[tt].flush()
    del out
    if bad:
        sys.exit(f"STOP: sanity off by more than {TOL_CK} for {bad}; .tmp files left, not renamed")
    for tt in OUT:
        os.rename(OUT[tt] + ".tmp", OUT[tt])
        print(f"saved {OUT[tt]}", flush=True)
    np.savez(STATS_OUT, stats=S, models=["v18c", "v19a", "teacher"], ckpts=[f"{k}={v}" for k, v in CKPTS.items()])
    print(f"saved {STATS_OUT}", flush=True)


def verify():
    import torch
    from train_v5m import load_split, read_batch
    if os.path.exists(RESULTS_OUT):
        sys.exit(f"{RESULTS_OUT} exists; not overwriting")
    device = setup()
    models = load(device)
    train = load_split("train")
    n, T = train["prices"].shape[:2]
    saved = {t: np.load(OUT[t], mmap_mode="r") for t in OUT}
    assert all(saved[t].shape == (n, T) and saved[t].dtype == np.float16 for t in OUT)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    rng = np.random.default_rng(VERIFY_SEED)
    vidx = np.sort(rng.choice(n, N_VERIFY, replace=False))
    worst = 0.0
    with torch.no_grad():
        for j in vidx:
            b = read_batch(train, [j])
            tch = teacher({k: forward(m, st, b, device) for k, (m, st, _) in models.items()})[0]
            for t in OUT:
                worst = max(worst, np.abs(tch[:, t].astype(np.float32) - np.asarray(saved[t][j], np.float32)).max())
    out("Teacher = v18c + v19a final checkpoints (average of raw outputs, clipped to [-2, 2], float16), all train rows")
    for k, v in CKPTS.items():
        out(f"  {k}: {v}")
    out(f"\nVerify: fresh forward pass (one sequence at a time) on {N_VERIFY} train sequences (seed {VERIFY_SEED}: "
        f"{vidx.tolist()}): max |fresh - saved| = {worst:.2e} over both targets and all 20,000 steps "
        f"({'PASS' if worst <= TOL_VERIFY else 'FAIL'}, tolerance {TOL_VERIFY})")
    if worst > TOL_VERIFY:
        with open(RESULTS_OUT, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sys.exit("STOP: verify failed")

    h0, h1 = holdout(0, n), holdout(1, n)
    both = np.intersect1d(h0, h1)
    rest = np.setdiff1d(np.arange(n), np.union1d(h0, h1))
    other = np.sort(rng.choice(rest, N_OTHER, replace=False))
    sets = [("data-seed-0 holdout (unseen by v18c; v25's holdout)", h0),
            ("data-seed-1 holdout (unseen by v19a)", h1),
            ("both holdouts (unseen by both)", both),
            (f"{N_OTHER} other train seqs (trained on by both)", other)]
    need = np.arange(T) >= WARMUP
    S = np.load(STATS_OUT)["stats"]  # main-pass statistics: v18c, v19a, teacher
    out(f"\nWeighted Pearson against the train targets (all rows >= 99, weights |clipped y|; t0 / t1 / mean)")
    out(f"  teacher recomputed from the saved files; v18c and v19a from the main pass ({STATS_OUT})")
    out(f"  {'set':<55} {'seqs':>5}   {'teacher':>22}   {'v18c':>7} {'v19a':>7}")
    res = {}
    for name, sel in sets:
        st = np.zeros((2, 6))
        for j in sel:
            y = np.asarray(train["targets"][j])
            p = np.stack([np.asarray(saved[t][j], np.float32) for t in OUT], axis=1)
            st += seq_stats(y, p, need)
        r = wp_from_stats(st)
        rm = wp_from_stats(S[:, sel].sum(1))
        assert abs(r[2] - rm[2, 2]) < 1e-9, (r, rm[2])  # same values as the main pass
        res[name] = r[2]
        out(f"  {name:<55} {len(sel):>5}   {r[0]:.4f} / {r[1]:.4f} / {r[2]:.4f}   {rm[0, 2]:>7.4f} {rm[1, 2]:>7.4f}")
    gap = res[sets[3][0]] - res[sets[2][0]]
    out(f"\nOverfit gap of the teacher (trained-on minus unseen-by-both): {gap:+.4f}; "
        f"trained-on minus data-seed-0 holdout: {res[sets[3][0]] - res[sets[0][0]]:+.4f}")
    out("Note: a data-seed-0 student never trains on the data-seed-0 holdout; on the rows it trains on, the teacher "
        "has seen every row through v18c and all but the data-seed-1 holdout through v19a.")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    if ap.parse_args().verify:
        verify()
    else:
        make()
