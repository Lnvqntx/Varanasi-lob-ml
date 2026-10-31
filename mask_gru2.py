"""Mask GRU v2: mask_gru.py with options (hidden size, layers, epochs, patience); same inputs and data split.

  python mask_gru2.py train [--hidden 128 --layers 2 --epochs 40 --patience 5] [--smoke]
  python mask_gru2.py infer [--hidden 128 --layers 2]   # -> ~/work/cache/{train,valid}_maskp2.npy (never overwrites)

Copy of mask_gru.py (unchanged there). Inputs are the same 168 columns as mask_gru.py and the main models: 44 prices,
their one-step differences (0 at step 0), the 68 other features, standardised with the stats of
train_v4_h128_e5_fullval_best.pt, plus the 12 fill-value flags (mask_gru.py already used the flags and the differences,
so nothing is added). Fit on the valid sequences with index % 4 != 0 (1,404), early stopping and checkpoint selection on
the AUC of the 469 held-out sequences (index % 4 == 0), BCE on steps >= 99, whole sequences in 1000-step chunks with
the hidden state carried, AdamW + cosine over --epochs, seed 0, as mask_gru.py. Outputs:
  checkpoint  ~/work/checkpoints/mask_gru2_h{H}_l{L}_best.pt      results  ~/work/mask_gru2_h{H}_l{L}_results.txt
--smoke: 32 fit / 16 held-out sequences, 2 epochs of at most 10 steps, outputs named ..._smoke (any GPU).
CUDA_VISIBLE_DEVICES defaults to 0 (set it to 1 only for --smoke).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
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

CACHE = "/home/jovyan/work/cache"
CKPT_DIR = "/home/jovyan/work/checkpoints"
STATS_CKPT = f"{CKPT_DIR}/train_v4_h128_e5_fullval_best.pt"
SEQ_LEN, WARMUP, CHUNK = 20000, 99, 1000
FILL = 5.19921875  # as train_v1.FILL
GPU_MEM_CAP_GB = 6.0
BATCH, LR, WD, SEED = 16, 1e-3, 1e-2, 0

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
    def __init__(self, n_in=168, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        o, h = self.gru(x, h)
        return self.head(o)[..., 0], h


def device_setup():
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
    out(f"  scored rate {y.mean():.4%}, mean p {p.mean():.4%} (held-out rows)")
    f, yr = p.mean(1), y.mean(1)
    out(f"  per-sequence scored rate vs mean p: Pearson {np.corrcoef(f, yr)[0, 1]:.4f}")
    out("  calibration (10 equal-width bins of p): bin, rows, mean p, true rate")
    pv, yv = p.ravel(), y.ravel()
    b = np.minimum((pv * 10).astype(int), 9)
    for k in range(10):
        s = b == k
        if s.any():
            out(f"    [{k / 10:.1f}, {(k + 1) / 10:.1f}{']' if k == 9 else ')'} {s.sum():>10,} {pv[s].mean():8.4f} {yv[s].mean():8.4f}")


def names(args):
    tag = f"mask_gru2_h{args.hidden}_l{args.layers}" + ("_smoke" if args.smoke else "")
    return f"{CKPT_DIR}/{tag}_best.pt", f"/home/jovyan/work/{tag}_results.txt"


def train(args):
    MASK_CKPT, RESULTS_OUT = names(args)
    HIDDEN, LAYERS, MAX_EPOCHS, PATIENCE = args.hidden, args.layers, args.epochs, args.patience
    if os.path.exists(MASK_CKPT) or os.path.exists(RESULTS_OUT):
        sys.exit("mask_gru2 outputs exist; not overwriting")
    set_seed(SEED)
    device = device_setup()
    inputs = Inputs(device)
    valid = load("valid", ["prices", "features"])
    valid["is_scored"] = np.load(f"{CACHE}/valid_is_scored.npy", mmap_mode="r")
    sc = np.asarray(valid["is_scored"])
    fit_idx, ho_idx = split_idx(len(sc))
    all_rate = sc[:, WARMUP:].mean()
    if args.smoke:
        fit_idx, ho_idx, MAX_EPOCHS = fit_idx[:32], ho_idx[:16], 2
    out(f"mask GRU v2: {LAYERS} layers, hidden {HIDDEN}; fit on {len(fit_idx)} valid seqs (index % 4 != 0), "
        f"held out {len(ho_idx)} (index % 4 == 0); seed {SEED}; batch {BATCH}, chunk {CHUNK}, AdamW lr {LR}, "
        f"wd {WD}, cosine over {MAX_EPOCHS} epochs, early stopping on held-out AUC (patience {PATIENCE})")
    model = MaskGRU(hidden=HIDDEN, layers=LAYERS).to(device)
    out(f"params: {sum(q.numel() for q in model.parameters()):,}; inputs 168 (prices, diffs, other, 12 flags)")
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
                if args.smoke and n >= 10:
                    break
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
                        "hidden": HIDDEN, "layers": LAYERS, "stats_ckpt": STATS_CKPT, "seed": SEED}, MASK_CKPT)
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
    out(f"  scored rate over all {len(sc)} valid sequences (steps >= {WARMUP}): {all_rate:.4%}")
    out(f"  gpu peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    with open(RESULTS_OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"saved {RESULTS_OUT}", flush=True)


def infer(args):
    MASK_CKPT, _ = names(args)
    assert not args.smoke, "infer has no smoke mode"
    targets = {s: f"{CACHE}/{s}_maskp2.npy" for s in ["train", "valid"]}
    for p in targets.values():
        if os.path.exists(p) or os.path.exists(p + ".tmp"):
            sys.exit(f"{p} (or .tmp) exists; not overwriting")
    set_seed(SEED)
    device = device_setup()
    inputs = Inputs(device)
    ck = torch.load(MASK_CKPT, map_location="cpu", weights_only=False)
    model = MaskGRU(hidden=ck["hidden"], layers=ck["layers"]).to(device)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40, help="cosine schedule length (mask_gru.py: 15)")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs (mask_gru.py: 3)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    train(a) if a.mode == "train" else infer(a)
