"""Small causal GRU that predicts the valid is_scored mask per row from the 168 model inputs.

  python mask_gru.py train   # fit on valid seqs with index % 4 != 0, early stopping on held-out (% 4 == 0) AUC
  python mask_gru.py infer   # p(scored) for every train and valid row -> ~/work/cache/{train,valid}_maskp.npy
  python mask_gru.py score CKPT [CKPT ...]   # main-model scores on held-out vs fitting valid sequences

Inputs: train_v1.build_inputs (copied here, train_v1 is not imported) with the standardization stats
of train_v4_h128_e5_fullval_best.pt; those are compute_stats on the seed-0 train permutation, i.e.
exactly what train_v1 / train_v2m compute with --seed 0. Model: 1-layer GRU, hidden 64, linear head,
BCE on steps >= 99, whole sequences in 1000-step chunks with the hidden state carried (as train_v1).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 only
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import math
import queue
import random
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

sys.path.insert(0, "/home/jovyan/work/wnn/wnn_connectome_starterpack")
from utils import GlobalAccumulator  # noqa: E402

CACHE = "/home/jovyan/work/cache"
CKPT_DIR = "/home/jovyan/work/checkpoints"
STATS_CKPT = f"{CKPT_DIR}/train_v4_h128_e5_fullval_best.pt"
MASK_CKPT = f"{CKPT_DIR}/mask_gru_h64_best.pt"
RESULTS_OUT = "/home/jovyan/work/mask_gru_results.txt"
SCORE_OUT = "/home/jovyan/work/mask_gru_heldout_scores.txt"
SEQ_LEN, WARMUP, CHUNK = 20000, 99, 1000
FILL = 5.19921875  # as train_v1.FILL
GPU_MEM_CAP_GB = 6.0
HIDDEN, BATCH, LR, WD = 64, 16, 1e-3, 1e-2
MAX_EPOCHS, PATIENCE, SEED = 15, 3, 0

LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def set_seed(seed):  # as train_v1.set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load(split, keys):
    return {k: np.load(f"{CACHE}/{split}_{k}.npy", mmap_mode="r") for k in keys}


def read_batch(data, idx):  # as train_v1.read_batch
    idx = np.sort(np.asarray(idx))
    return {k: np.ascontiguousarray(v[idx]) for k, v in data.items()}


class Prefetcher:  # as train_v1.Prefetcher; also yields the batch indices
    def __init__(self, data, batches, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, args=(data, batches), daemon=True)
        self.t.start()

    def _run(self, data, batches):
        for idx in batches:
            self.q.put((np.sort(idx), read_batch(data, idx)))
        self.q.put(None)

    def __iter__(self):
        while (b := self.q.get()) is not None:
            yield b


class Inputs:
    """train_v1.build_inputs with fixed stats and flag columns from a checkpoint."""

    def __init__(self, device):
        ck = torch.load(STATS_CKPT, map_location="cpu", weights_only=False)
        self.mean = ck["stats"]["mean"].to(device)
        self.std = ck["stats"]["std"].to(device)
        self.flag_idx = [ck["other_cols"].index(c) for c in ck["flag_cols"]]
        self.device = device

    def __call__(self, b):
        prices = torch.from_numpy(b["prices"]).to(self.device).float()
        other = torch.from_numpy(b["features"]).to(self.device).float()
        dprices = torch.zeros_like(prices)
        dprices[:, 1:] = prices[:, 1:] - prices[:, :-1]
        flags = (other[:, :, self.flag_idx].abs() == FILL).float()
        cont = torch.cat([prices, dprices, other], dim=2)
        cont = (cont - self.mean) / self.std
        return torch.cat([cont, flags], dim=2)


class MaskGRU(nn.Module):
    def __init__(self, n_in=168, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        o, h = self.gru(x, h)
        return self.head(o)[..., 0], h


class MainGRU(nn.Module):  # same layout and state_dict keys as train_v1.GRUModel
    def __init__(self, n_in=168, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h=None):
        o, h = self.gru(x, h)
        return self.head(o), h


def device_setup():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    return device


@torch.no_grad()
def predict_p(model, inputs, data, idx, batch=50):
    """p(scored) for whole sequences idx (sorted) -> (len(idx), SEQ_LEN) float32."""
    model.eval()
    idx = np.sort(np.asarray(idx))
    res = np.zeros((len(idx), SEQ_LEN), dtype=np.float32)
    pos = {v: k for k, v in enumerate(idx)}
    batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    for bidx, b in Prefetcher({k: data[k] for k in ["prices", "features"]}, batches):
        x = inputs(b)
        h, ps = None, []
        for s in range(0, SEQ_LEN, CHUNK):
            lg, h = model(x[:, s:s + CHUNK], h)
            ps.append(torch.sigmoid(lg))
        res[[pos[i] for i in bidx]] = torch.cat(ps, 1).cpu().numpy()
    model.train()
    return res


def auc(score, y):
    """ROC AUC, ties counted half (rank formulation)."""
    y = np.asarray(y, dtype=bool)
    npos, nneg = y.sum(), (~y).sum()
    r = rankdata(score)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def top_decile_capture(score, y):
    k = int(np.ceil(0.1 * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].sum() / y.sum()


def split_idx(n):
    i = np.arange(n)
    return i[i % 4 != 0], i[i % 4 == 0]


def report_mask(p, y):
    """p, y: (n_seq, SEQ_LEN - WARMUP) held-out predictions and labels."""
    out(f"  overall AUC                      {auc(p.ravel(), y.ravel()):.4f}")
    per = [auc(p[i], y[i]) for i in range(len(p)) if y[i].sum() >= 20 and (~y[i]).sum() >= 20]
    out(f"  mean within-sequence AUC         {np.mean(per):.4f}  (median {np.median(per):.4f}, "
        f"{len(per)} of {len(p)} sequences with >= 20 scored and >= 20 unscored rows)")
    pc = p - p.mean(1, keepdims=True)
    out(f"  AUC after subtracting seq mean p {auc(pc.ravel(), y.ravel()):.4f}")
    out(f"  top-decile capture               {top_decile_capture(p.ravel(), y.ravel()):.4f}  (random 0.10)")
    out(f"  scored rate {y.mean():.4%}, mean p {p.mean():.4%}")
    f, yr = p.mean(1), y.mean(1)
    out(f"  per-sequence scored rate vs mean p: Pearson {np.corrcoef(f, yr)[0, 1]:.4f}")
    out("  calibration (10 equal-width bins of p): bin, rows, mean p, true rate")
    pv, yv = p.ravel(), y.ravel()
    b = np.minimum((pv * 10).astype(int), 9)
    for k in range(10):
        s = b == k
        if s.any():
            out(f"    [{k / 10:.1f}, {(k + 1) / 10:.1f}{']' if k == 9 else ')'} {s.sum():>10,} {pv[s].mean():8.4f} {yv[s].mean():8.4f}")


def train():
    if os.path.exists(MASK_CKPT) or os.path.exists(RESULTS_OUT):
        sys.exit("mask_gru outputs exist; not overwriting")
    set_seed(SEED)
    device = device_setup()
    inputs = Inputs(device)
    valid = load("valid", ["prices", "features"])
    valid["is_scored"] = np.load(f"{CACHE}/valid_is_scored.npy", mmap_mode="r")
    sc = np.asarray(valid["is_scored"])
    fit_idx, ho_idx = split_idx(len(sc))
    out(f"mask GRU: 1 layer, hidden {HIDDEN}; fit on {len(fit_idx)} valid seqs (index % 4 != 0), "
        f"held out {len(ho_idx)} (index % 4 == 0); seed {SEED}; batch {BATCH}, chunk {CHUNK}, AdamW lr {LR}, "
        f"wd {WD}, cosine over {MAX_EPOCHS} epochs, early stopping on held-out AUC (patience {PATIENCE})")
    model = MaskGRU().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = math.ceil(len(fit_idx) / BATCH) * (SEQ_LEN // CHUNK)
    total = MAX_EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    rng = np.random.default_rng(SEED)
    keep = (torch.arange(SEQ_LEN, device=device) >= WARMUP)
    y_ho = sc[ho_idx, WARMUP:]
    best, bad, step = -1.0, 0, 0
    for epoch in range(MAX_EPOCHS):
        t = time.time()
        order = rng.permutation(fit_idx)
        batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        tot, n = 0.0, 0
        for _, b in Prefetcher(valid, batches):
            x = inputs(b)
            y = torch.from_numpy(b["is_scored"]).to(device).float()
            h = None
            for s in range(0, SEQ_LEN, CHUNK):
                lg, h = model(x[:, s:s + CHUNK], h)
                k = keep[s:s + CHUNK]
                loss = nn.functional.binary_cross_entropy_with_logits(lg[:, k], y[:, s:s + CHUNK][:, k])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                h = h.detach()
                step += 1
                tot += loss.item()
                n += 1
        p = predict_p(model, inputs, valid, ho_idx)[:, WARMUP:]
        a = auc(p.ravel(), y_ho.ravel())
        print(f"epoch {epoch} step {step}: train BCE {tot / n:.4f} | held-out AUC {a:.4f} | {time.time() - t:.0f}s | "
              f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        if a > best:
            best, bad = a, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "step": step, "heldout_auc": a,
                        "hidden": HIDDEN, "stats_ckpt": STATS_CKPT, "seed": SEED}, MASK_CKPT)
            print(f"  saved {MASK_CKPT}", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop after epoch {epoch}", flush=True)
                break
    ck = torch.load(MASK_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    out(f"best epoch {ck['epoch']} (step {ck['step']}), held-out AUC {ck['heldout_auc']:.4f}; checkpoint {MASK_CKPT}")
    out(f"\nHeld-out {len(ho_idx)} valid sequences, steps >= {WARMUP}:")
    p = predict_p(model, inputs, valid, ho_idx)[:, WARMUP:]
    report_mask(p, y_ho)
    out(f"  LightGBM (mask_predict.py) within-sequence AUC: not available (its predictions were not saved)")
    out(f"  gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"saved {RESULTS_OUT}", flush=True)


def infer():
    targets = {s: f"{CACHE}/{s}_maskp.npy" for s in ["train", "valid"]}
    for p in targets.values():
        if os.path.exists(p) or os.path.exists(p + ".tmp"):
            sys.exit(f"{p} (or .tmp) exists; not overwriting")
    set_seed(SEED)
    device = device_setup()
    inputs = Inputs(device)
    ck = torch.load(MASK_CKPT, map_location="cpu", weights_only=False)
    model = MaskGRU(hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"{MASK_CKPT}: epoch {ck['epoch']}, held-out AUC {ck['heldout_auc']:.4f}", flush=True)
    summary = {}
    for split, path in targets.items():
        data = load(split, ["prices", "features"])
        n = data["prices"].shape[0]
        t = time.time()
        arr = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float16, shape=(n, SEQ_LEN))
        tot = 0.0
        for i0 in range(0, n, 500):
            idx = np.arange(i0, min(i0 + 500, n))
            p16 = predict_p(model, inputs, data, idx).astype(np.float16)
            arr[idx] = p16
            tot += p16[:, WARMUP:].sum(dtype=np.float64)
            print(f"  {split}: {idx[-1] + 1}/{n} seqs, {time.time() - t:.0f}s", flush=True)
        mean_p = tot / (n * (SEQ_LEN - WARMUP))  # of the stored float16 values, steps >= 99
        arr[:, :WARMUP] = np.float16(mean_p)  # neutral weight for warm-up rows (the loss ignores them)
        arr.flush()
        nan = sum(int(np.isnan(arr[i:i + 500]).sum()) for i in range(0, n, 500))
        assert nan == 0, f"{split}: {nan} NaN values"
        del arr
        os.rename(path + ".tmp", path)
        chk = np.load(path, mmap_mode="r")
        sample = np.asarray(chk[:, WARMUP::7], dtype=np.float32)  # every 7th step for quantiles
        q = np.percentile(sample, [1, 5, 25, 50, 75, 95, 99])
        summary[split] = dict(shape=list(chk.shape), dtype=str(chk.dtype), mean_p=mean_p,
                              quantiles=dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], map(float, q))),
                              max=float(sample.max()), nan=nan)
        print(f"{split}: {path} shape {chk.shape} {chk.dtype}, NaN {nan}, mean p (steps >= 99) {mean_p:.5f}, "
              + ", ".join(f"{k} {v:.5f}" for k, v in summary[split]["quantiles"].items())
              + f", max {summary[split]['max']:.4f}  ({time.time() - t:.0f}s)", flush=True)
    print(json.dumps(summary, indent=1))


@torch.no_grad()
def score(ckpts):
    if os.path.exists(SCORE_OUT):
        sys.exit(f"{SCORE_OUT} exists; not overwriting")
    set_seed(SEED)
    device = device_setup()
    valid = load("valid", ["prices", "features", "targets"])
    sc = np.load(f"{CACHE}/valid_is_scored.npy")
    fit_idx, ho_idx = split_idx(len(sc))
    need = np.arange(SEQ_LEN) >= WARMUP
    out(f"Main-model weighted Pearson on valid, by mask-predictor split (held out = index % 4 == 0, "
        f"{len(ho_idx)} seqs; fitting = {len(fit_idx)} seqs; all = 1873)")
    out(f"  {'checkpoint':<44} {'split':<9} {'masked':>8} {'t0':>8} {'t1':>8} {'all>=99':>8}")
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = MainGRU(hidden=ck["args"]["hidden"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        mean, std = ck["stats"]["mean"].to(device), ck["stats"]["std"].to(device)
        flag_idx = [ck["other_cols"].index(c) for c in ck["flag_cols"]]
        acc = {s: (GlobalAccumulator(), GlobalAccumulator()) for s in ["held out", "fitting", "all"]}
        batches = [np.arange(i, min(i + 25, len(sc))) for i in range(0, len(sc), 25)]
        for bidx, b in Prefetcher(valid, batches):
            prices = torch.from_numpy(b["prices"]).to(device).float()
            other = torch.from_numpy(b["features"]).to(device).float()
            dp = torch.zeros_like(prices)
            dp[:, 1:] = prices[:, 1:] - prices[:, :-1]
            flags = (other[:, :, flag_idx].abs() == FILL).float()
            x = torch.cat([(torch.cat([prices, dp, other], 2) - mean) / std, flags], 2)
            h, ps = None, []
            for s in range(0, SEQ_LEN, CHUNK):
                p, h = model(x[:, s:s + CHUNK], h)
                ps.append(p)
            pred = torch.cat(ps, 1).float().cpu().numpy()
            for k, i in enumerate(bidx):
                for split in ["all", "held out" if i % 4 == 0 else "fitting"]:
                    acc[split][0].add(b["targets"][k], pred[k], sc[i] & need)
                    acc[split][1].add(b["targets"][k], pred[k], need)
        for split, (am, aa) in acc.items():
            rm, ra = am.result(), aa.result()
            out(f"  {os.path.basename(path):<44} {split:<9} {rm['weighted_pearson']:>8.4f} {rm['t0']:>8.4f} "
                f"{rm['t1']:>8.4f} {ra['weighted_pearson']:>8.4f}")
    with open(SCORE_OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"saved {SCORE_OUT}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train":
        train()
    elif mode == "infer":
        infer()
    elif mode == "score":
        score(sys.argv[2:])
    else:
        sys.exit(__doc__)
