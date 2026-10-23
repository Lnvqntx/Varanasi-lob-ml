"""GRU baseline v1 on the .npy cache built by make_cache.py.

Inputs per step (168): 44 prices (float32), their first difference within the sequence (0 at step 0),
68 other features, 12 binary flags marking the 5.199 fill value in a0, a5, a6, a7 and the dv columns.
Continuous inputs are standardized with per-column stats from a sample of training sequences.

Training: whole 20k-step sequences, batch of 16, truncated BPTT in 1000-step chunks with the hidden
state carried (detached) between chunks; one optimizer step per chunk. State resets per batch.
Loss: weighted MSE on [-2, 2]-clipped preds/targets, weight |clipped target| + 0.05, steps >= 99 only.

Usage:
  python train_v1.py --smoke      # 20 train seqs, 50 optimizer steps, separate log/checkpoint
  python train_v1.py              # full run
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import logging
import math
import queue
import random
import sys
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from make_cache import DATA_DIR, price_and_other_cols

sys.path.insert(0, "/home/jovyan/work/wnn/wnn_connectome_starterpack")
from utils import WARMUP, GlobalAccumulator  # noqa: E402

CACHE = "/home/jovyan/work/cache"
CKPT_DIR = "/home/jovyan/work/checkpoints"
SEQ_LEN = 20000
FILL = 5.19921875  # the 5.199337 fill value after the float16 cast
FLAG_COLS = ["a0", "a5", "a6", "a7"] + [f"i{i}_dv{j}" for i in range(2) for j in range(4)]
GPU_MEM_CAP_GB = 6.0

PRICE_COLS, OTHER_COLS = price_and_other_cols(pq.ParquetFile(f"{DATA_DIR}/train.parquet").schema_arrow.names)
FLAG_IDX = [OTHER_COLS.index(c) for c in FLAG_COLS]
N_IN = 44 + 44 + 68 + len(FLAG_COLS)


def load_split(split):
    d = {k: np.load(f"{CACHE}/{split}_{k}.npy", mmap_mode="r") for k in ["prices", "features", "targets"]}
    if split == "valid":
        d["is_scored"] = np.load(f"{CACHE}/valid_is_scored.npy", mmap_mode="r")
    return d


def read_batch(data, idx):
    """Read whole sequences for the given indices (sorted for sequential I/O) into RAM."""
    idx = np.sort(np.asarray(idx))
    return {k: np.ascontiguousarray(v[idx]) for k, v in data.items()}


def build_inputs(prices, other, stats):
    """prices (B,T,44) f32, other (B,T,68) f16 -> (B,T,168) f32 on GPU, standardized except flags."""
    prices = prices.float()
    other = other.float()
    dprices = torch.zeros_like(prices)
    dprices[:, 1:] = prices[:, 1:] - prices[:, :-1]
    flags = (other[:, :, FLAG_IDX].abs() == FILL).float()
    cont = torch.cat([prices, dprices, other], dim=2)
    cont = (cont - stats["mean"]) / stats["std"]
    return torch.cat([cont, flags], dim=2)


def compute_stats(data, idx, device):
    s1 = torch.zeros(156, dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    n = 0
    ident = {"mean": torch.zeros(156, device=device), "std": torch.ones(156, device=device)}
    for i in idx:
        b = read_batch(data, [i])
        x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), ident)
        x = x[0, 1:, :156].double()  # skip step 0 where the diff is forced to 0
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        n += x.shape[0]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-6)
    return {"mean": mean.float(), "std": std.float()}


class GRUModel(nn.Module):
    def __init__(self, n_in=N_IN, hidden=256, layers=2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.head(out), h


def loss_fn(pred, y, valid):
    yc = y.clamp(-2, 2)
    pc = pred.clamp(-2, 2)
    w = (yc.abs() + 0.05) * valid
    return (w * (pc - yc) ** 2).sum() / w.sum().clamp_min(1e-8)


class Prefetcher:
    """Background thread that reads the next batches from the memmaps while the GPU trains."""

    def __init__(self, data, batches, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, args=(data, batches), daemon=True)
        self.t.start()

    def _run(self, data, batches):
        for idx in batches:
            self.q.put(read_batch(data, idx))
        self.q.put(None)

    def __iter__(self):
        while (b := self.q.get()) is not None:
            yield b


@torch.no_grad()
def evaluate(model, data, idx, stats, device, chunk, use_is_scored, batch=25):
    """Score with the official accumulator. With use_is_scored, returns (is_scored result, all-steps result)."""
    model.eval()
    acc = GlobalAccumulator()
    acc_all = GlobalAccumulator()
    need = np.arange(SEQ_LEN) >= WARMUP
    batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    for b in Prefetcher(data, batches):
        x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
        h = None
        preds = []
        for s in range(0, SEQ_LEN, chunk):
            p, h = model(x[:, s:s + chunk], h)
            preds.append(p)
        preds = torch.cat(preds, 1).float().cpu().numpy()
        for k in range(len(preds)):
            mask = (b["is_scored"][k] & need) if use_is_scored else need
            acc.add(b["targets"][k], preds[k], mask)
            if use_is_scored:
                acc_all.add(b["targets"][k], preds[k], need)
    model.train()
    return (acc.result(), acc_all.result()) if use_is_scored else acc.result()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--max-steps", type=int, default=None, help="cap on optimizer steps")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--n-stats", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=2000, help="optimizer steps between evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--name", default="train_v1", help="log and checkpoint name")
    args = ap.parse_args()

    name = f"{args.name}_smoke" if args.smoke else args.name
    if args.smoke and args.max_steps is None:
        args.max_steps = 50
    os.makedirs(CKPT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(f"/home/jovyan/work/{name}.log"), logging.StreamHandler()])
    log = logging.info
    log(f"args: {vars(args)}")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    log(f"device: {torch.cuda.get_device_name(0)}, memory cap {GPU_MEM_CAP_GB} GB")

    train, valid = load_split("train"), load_split("valid")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(train["prices"].shape[0])
    holdout_idx = np.sort(perm[:args.n_holdout])
    train_idx = perm[args.n_holdout:]
    if args.smoke:
        train_idx = train_idx[:20]
    val_idx = np.sort(rng.choice(valid["prices"].shape[0], args.n_val, replace=False))
    log(f"train seqs {len(train_idx)}, train holdout {len(holdout_idx)}, valid subset {len(val_idx)}")

    t0 = time.time()
    stats = compute_stats(train, np.sort(train_idx[:args.n_stats]), device)
    log(f"input stats from {min(args.n_stats, len(train_idx))} seqs in {time.time() - t0:.1f}s")

    model = GRUModel(hidden=args.hidden).to(device)
    log(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    chunks_per_seq = SEQ_LEN // args.chunk
    batches_per_epoch = math.ceil(len(train_idx) / args.batch)
    total_steps = int(args.epochs * batches_per_epoch * chunks_per_seq)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    log(f"{batches_per_epoch} batches/epoch x {chunks_per_seq} chunks = {batches_per_epoch * chunks_per_seq} "
        f"steps/epoch; total optimizer steps {total_steps}")

    steps_in_chunk = torch.arange(SEQ_LEN, device=device)
    step = 0
    best = -float("inf")
    run_loss, run_n, t_train, t_last = 0.0, 0, 0.0, time.time()

    def do_eval():
        nonlocal best
        torch.cuda.synchronize()
        te = time.time()
        rv, rva = evaluate(model, valid, val_idx, stats, device, args.chunk, use_is_scored=True)
        tv = time.time() - te
        rh = evaluate(model, train, holdout_idx, stats, device, args.chunk, use_is_scored=False)
        log(f"[eval step {step}] valid is_scored wp={rv['weighted_pearson']:.4f} (t0 {rv['t0']:.4f}, "
            f"t1 {rv['t1']:.4f}, rows {rv['selected_rows']:,}) | valid all>=99 wp={rva['weighted_pearson']:.4f} "
            f"(t0 {rva['t0']:.4f}, t1 {rva['t1']:.4f}) | train-holdout wp={rh['weighted_pearson']:.4f} "
            f"(t0 {rh['t0']:.4f}, t1 {rh['t1']:.4f}) | valid eval {tv:.1f}s, holdout eval {time.time() - te - tv:.1f}s")
        if rv["weighted_pearson"] > best:
            best = rv["weighted_pearson"]
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args), "step": step,
                        "valid": rv, "valid_all": rva, "holdout": rh, "price_cols": PRICE_COLS, "other_cols": OTHER_COLS,
                        "flag_cols": FLAG_COLS}, f"{CKPT_DIR}/{name}_best.pt")
            log(f"  saved new best ({best:.4f}) to {CKPT_DIR}/{name}_best.pt")

    epoch = 0
    model.train()
    while step < total_steps:
        order = rng.permutation(train_idx)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        for b in Prefetcher(train, batches):
            x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
            y = torch.from_numpy(b["targets"]).to(device)
            h = None
            for s in range(0, SEQ_LEN, args.chunk):
                if step >= total_steps:
                    break
                ts = time.time()
                valid_mask = (steps_in_chunk[s:s + args.chunk] >= WARMUP).float()[None, :, None]
                pred, h = model(x[:, s:s + args.chunk], h)
                loss = loss_fn(pred, y[:, s:s + args.chunk], valid_mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                h = h.detach()
                step += 1
                run_loss += loss.item()
                run_n += 1
                t_train += time.time() - ts
                if step % 100 == 0 or (args.smoke and step % 10 == 0):
                    el = time.time() - t_last
                    log(f"epoch {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} | {run_n / el:.1f} steps/s wall, "
                        f"{run_n * args.batch * args.chunk / el:,.0f} seq-steps/s | "
                        f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
                    run_loss, run_n, t_last = 0.0, 0, time.time()
                if step % args.eval_every == 0:
                    do_eval()
                    t_last = time.time()
            if step >= total_steps:
                break
        epoch += 1

    do_eval()
    log(f"done. best valid wp {best:.4f}; pure train compute {t_train:.1f}s for {step} steps "
        f"({step / max(t_train, 1e-9):.2f} steps/s); gpu peak allocated "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
