"""Train K models from one shared stream of batches (train_v5m.py's recipe, one process).

The cache sits on one HDD (about 82 MB/s, about 16 steps/s for one model), while the GPU has spare
capacity. Here every batch is read from disk once and used by all K models: same data loader, batch
order, chunking, train/holdout/valid split and mask weights, and per model its own initialisation seed,
architecture, auxiliary heads, AdamW optimizer, cosine schedule, loss, log file and checkpoints.

--models "name:seed:arch:aux:aux_weight,name:seed:arch:aux:aux_weight,..."
    e.g. "train_v16_a:0:gru:50,200,-50,-200:0.25,train_v16_b:1:lstm::0"
    aux is train_v5m.py's --aux (commas inside it are fine; "" = none), aux_weight its --aux-weight.
    Optional per-model settings follow as key=value fields after aux_weight, each after a ':':
      alpha=A     mask-weight alpha (default: the shared --alpha)
      layers=L    recurrent layers (default 2)
      dropout=D   dropout between recurrent layers, training only (default 0; needs layers >= 2)
      wd=W        AdamW weight decay (default: the shared --wd, 0.01)
    e.g. "v16a:0:gru::0:dropout=0.1:wd=0.05,v16b:0:gru::0:alpha=0.75,v16c:0:gru::0:layers=3"
    With all defaults a model is built exactly as in train_v5m.py (same class, same RNG use).
--data-seed (default: the first model's seed) seeds the data stream exactly as --seed does in
train_v5m.py (holdout/train split, valid subset, batch order). Model i's weights are initialised after
torch.manual_seed(seed_i), as train_v5m.py does after set_seed(seed). So a K = 1 run with seed = data
seed reproduces train_v5m.py with that --seed (and train_v2m.py with aux "").

Per model, as train_v5m.py: loss lines in ~/work/{name}.log (same format), evaluation on the same valid
subset (first two outputs only), best checkpoint {name}_best.pt on valid is_scored wp, plus a final
checkpoint {name}_final.pt. Evaluation reads each valid / holdout batch once and runs all K models on it.
The combined log ~/work/{run}.log (--run, default "multi_" + first name) carries every line prefixed
with [name], and an [all] line with the wall rate and each model's own compute rate.

Usage (one process, one GPU):
  CUDA_VISIBLE_DEVICES=0 nohup python train_multi.py --models "..." --hidden 128 --epochs 5 --n-val 1873 \\
      --n-holdout 300 --eval-every 12800 --mask-p ~/work/cache/train_maskp.npy --alpha 0.5 > ~/work/x.out 2>&1 &
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import logging
import math
import random
import time

import numpy as np
import torch

from train_v5m import (ARCHS, CKPT_DIR, FLAG_COLS, GPU_MEM_CAP_GB, OTHER_COLS, PRICE_COLS, SEQ_LEN, WARMUP,
                       GlobalAccumulator, Prefetcher, aux_loss, aux_targets, build_inputs, compute_stats,
                       detach_state, load_split, loss_fn, parse_offsets, set_seed)


OPTIONS = {"alpha": float, "layers": int, "dropout": float, "wd": float}


def parse_models(spec):
    """'n:s:a:aux:w[:k=v...],n:s:a:aux:w[:k=v...]' -> list of dicts (options None = default). aux may hold
    commas, so split on ':'; after aux, every field is aux_weight or k=v, and a field holding a ',' ends the
    model, with the next model's name after the comma (aux_weight and option values have no commas)."""
    tok = spec.split(":")
    out, i = [], 0
    while i < len(tok):
        assert i + 4 < len(tok), f"bad --models {spec!r}"
        name, seed, arch, aux = tok[i:i + 4]
        assert arch in ARCHS, f"arch {arch!r}"
        fields, i, nxt = [], i + 4, None
        while i < len(tok):
            f = tok[i]
            i += 1
            if "," in f:
                f, nxt = f.split(",", 1)
            fields.append(f)
            if nxt is not None:
                break
        m = {"name": name, "seed": int(seed), "arch": arch, "aux": aux, "aux_weight": float(fields[0]),
             "offsets": parse_offsets(aux), **{k: None for k in OPTIONS}}
        for f in fields[1:]:
            k, v = f.split("=", 1)
            assert k in OPTIONS, f"unknown option {k!r} (known: {sorted(OPTIONS)})"
            m[k] = OPTIONS[k](v)
        assert m["dropout"] is None or m["dropout"] == 0 or (m["layers"] or 2) >= 2
        out.append(m)
        if nxt is not None:
            tok[i - 1] = nxt  # the next model starts at the name peeled off this field
            i -= 1
    names = [m["name"] for m in out]
    assert len(set(names)) == len(names), f"duplicate names {names}"
    return out


class RNNModel(torch.nn.Module):
    """train_v5m.GRUModel / LSTMModel with dropout between layers (same attribute names and init order)."""

    def __init__(self, arch, n_in, hidden, layers, n_out, dropout):
        super().__init__()
        rnn = torch.nn.GRU if arch == "gru" else torch.nn.LSTM
        setattr(self, arch, rnn(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout))
        self.rnn_name = arch
        self.head = torch.nn.Linear(hidden, n_out)

    def forward(self, x, h=None):
        out, h = getattr(self, self.rnn_name)(x, h)
        return self.head(out), h


def build_model(sp, hidden):
    n_out, layers, dropout = 2 + 2 * len(sp["offsets"]), sp["layers"] or 2, sp["dropout"] or 0.0
    if dropout == 0:
        return ARCHS[sp["arch"]](hidden=hidden, layers=layers, n_out=n_out)
    from train_v5m import N_IN
    return RNNModel(sp["arch"], N_IN, hidden, layers, n_out, dropout)


@torch.no_grad()
def evaluate_multi(models, data, idx, stats, device, chunk, use_is_scored, batch=25):
    """train_v5m.evaluate for K models; each batch is read once. Returns a list of per-model results."""
    for m in models:
        m.eval()
    acc = [GlobalAccumulator() for _ in models]
    acc_all = [GlobalAccumulator() for _ in models]
    need = np.arange(SEQ_LEN) >= WARMUP
    batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    for b in Prefetcher(data, batches):
        x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
        for i, model in enumerate(models):
            h = None
            preds = []
            for s in range(0, SEQ_LEN, chunk):
                p, h = model(x[:, s:s + chunk], h)
                preds.append(p[:, :, :2])  # main targets only; auxiliary outputs are never scored
            preds = torch.cat(preds, 1).float().cpu().numpy()
            for k in range(len(preds)):
                mask = (b["is_scored"][k] & need) if use_is_scored else need
                acc[i].add(b["targets"][k], preds[k], mask)
                if use_is_scored:
                    acc_all[i].add(b["targets"][k], preds[k], need)
    for m in models:
        m.train()
    return [(a.result(), aa.result()) if use_is_scored else a.result() for a, aa in zip(acc, acc_all)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='"name:seed:arch:aux:aux_weight,..."')
    ap.add_argument("--data-seed", type=int, default=None, help="data stream seed (default: first model's seed)")
    ap.add_argument("--run", default=None, help="combined log name (default multi_<first name>)")
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
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--mask-p", default=None, help="float16 train p(scored) cache from mask_gru.py infer")
    ap.add_argument("--alpha", type=float, default=1.0, help="m = alpha + (1 - alpha) * min(p / mean_p, 20)")
    args = ap.parse_args()

    specs = parse_models(args.models)
    for sp in specs:  # resolved settings (the trailing _ marks resolved values)
        sp.update(alpha_=args.alpha if sp["alpha"] is None else sp["alpha"], layers_=sp["layers"] or 2,
                  dropout_=sp["dropout"] or 0.0, wd_=args.wd if sp["wd"] is None else sp["wd"])
    data_seed = specs[0]["seed"] if args.data_seed is None else args.data_seed
    sfx = "_smoke" if args.smoke else ""
    for sp in specs:
        sp["run_name"] = sp["name"] + sfx
    run = (args.run or f"multi_{specs[0]['name']}") + sfx
    if args.smoke and args.max_steps is None:
        args.max_steps = 50
    os.makedirs(CKPT_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(f"/home/jovyan/work/{run}.log"), logging.StreamHandler()])
    log = logging.info
    for sp in specs:
        lg = logging.getLogger(sp["run_name"])
        lg.propagate = False
        fh = logging.FileHandler(f"/home/jovyan/work/{sp['run_name']}.log")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
        lg.setLevel(logging.INFO)
        sp["logger"] = lg

    def mlog(sp, msg):
        sp["logger"].info(msg)
        log(f"[{sp['name']}] {msg}")

    def model_args(sp):
        """The checkpoint/log args of this model, in train_v5m.py's layout."""
        a = {k: v for k, v in vars(args).items() if k not in ("models", "data_seed", "run")}
        a.update(seed=sp["seed"], name=sp["name"], arch=sp["arch"], aux=sp["aux"], aux_weight=sp["aux_weight"],
                 alpha=sp["alpha_"], layers=sp["layers_"], dropout=sp["dropout_"], wd=sp["wd_"],
                 data_seed=data_seed, multi_run=run, multi_models=[s["name"] for s in specs])
        return a

    log(f"args: {vars(args)}; data seed {data_seed}; {len(specs)} models")
    for sp in specs:
        mlog(sp, f"args: {model_args(sp)}")

    set_seed(data_seed)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        GPU_MEM_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    log(f"device: {torch.cuda.get_device_name(0)}, memory cap {GPU_MEM_CAP_GB} GB")

    train, valid = load_split("train"), load_split("valid")
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(train["prices"].shape[0])
    holdout_idx = np.sort(perm[:args.n_holdout])
    train_idx = perm[args.n_holdout:]
    if args.smoke:
        train_idx = train_idx[:20]
    val_idx = np.sort(rng.choice(valid["prices"].shape[0], args.n_val, replace=False))
    log(f"train seqs {len(train_idx)}, train holdout {len(holdout_idx)}, valid subset {len(val_idx)}")

    train_fit = train
    mean_p = None
    if args.mask_p is not None:
        mp = np.load(os.path.expanduser(args.mask_p), mmap_mode="r")
        assert mp.shape == train["prices"].shape[:2], f"mask-p shape {mp.shape} != train {train['prices'].shape[:2]}"
        train_fit = dict(train, maskp=mp)
        srt = np.sort(train_idx)
        tot = sum(np.asarray(mp[srt[i:i + 500], WARMUP:]).sum(dtype=np.float64) for i in range(0, len(srt), 500))
        mean_p = tot / (len(srt) * (SEQ_LEN - WARMUP))
        log(f"mask weights from {args.mask_p}: alpha {args.alpha}, mean_p {mean_p:.6f} over {len(srt)} train seqs "
            f"(steps >= {WARMUP}); m = alpha + (1 - alpha) * min(p / mean_p, 20)")

    t0 = time.time()
    stats = compute_stats(train, np.sort(train_idx[:args.n_stats]), device)
    log(f"input stats from {min(args.n_stats, len(train_idx))} seqs in {time.time() - t0:.1f}s")

    chunks_per_seq = SEQ_LEN // args.chunk
    batches_per_epoch = math.ceil(len(train_idx) / args.batch)
    total_steps = int(args.epochs * batches_per_epoch * chunks_per_seq)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    for sp in specs:
        # train_v5m.py: set_seed(seed) at start; nothing before the model draws from torch's RNG.
        random.seed(sp["seed"])
        torch.manual_seed(sp["seed"])
        torch.cuda.manual_seed_all(sp["seed"])
        sp["model"] = build_model(sp, args.hidden).to(device)
        mlog(sp, f"settings: alpha {sp['alpha_']}, layers {sp['layers_']}, dropout {sp['dropout_']}, "
                 f"weight decay {sp['wd_']}")
        if sp["offsets"]:
            mlog(sp, f"auxiliary heads: offsets {sp['offsets']}, weight {sp['aux_weight']} each; "
                     f"head outputs {2 + 2 * len(sp['offsets'])}")
        mlog(sp, f"params: {sum(p.numel() for p in sp['model'].parameters()):,}")
        sp["opt"] = torch.optim.AdamW(sp["model"].parameters(), lr=args.lr, weight_decay=sp["wd_"])
        sp["sched"] = torch.optim.lr_scheduler.LambdaLR(
            sp["opt"], lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
        sp.update(best=-float("inf"), run_loss=0.0, run_main=0.0, run_aux=0.0, t_model=0.0, t_since=0.0)
    log(f"{batches_per_epoch} batches/epoch x {chunks_per_seq} chunks = {batches_per_epoch * chunks_per_seq} "
        f"steps/epoch; total optimizer steps {total_steps} per model")
    for sp in specs:
        mlog(sp, f"{batches_per_epoch} batches/epoch x {chunks_per_seq} chunks = {batches_per_epoch * chunks_per_seq} "
                 f"steps/epoch; total optimizer steps {total_steps}")

    steps_in_chunk = torch.arange(SEQ_LEN, device=device)
    step = 0
    run_n, t_train, t_last = 0, 0.0, time.time()
    t_wait = 0.0  # time the GPU loop waited for the loader

    def save(sp, path, rv, rva, rh):
        torch.save({"model": sp["model"].state_dict(), "stats": stats, "args": model_args(sp), "step": step,
                    "valid": rv, "valid_all": rva, "holdout": rh, "price_cols": PRICE_COLS, "other_cols": OTHER_COLS,
                    "flag_cols": FLAG_COLS}, path)

    def do_eval(final=False):
        torch.cuda.synchronize()
        te = time.time()
        models = [sp["model"] for sp in specs]
        rvs = evaluate_multi(models, valid, val_idx, stats, device, args.chunk, use_is_scored=True)
        tv = time.time() - te
        rhs = evaluate_multi(models, train, holdout_idx, stats, device, args.chunk, use_is_scored=False)
        th = time.time() - te - tv
        for sp, (rv, rva), rh in zip(specs, rvs, rhs):
            mlog(sp, f"[eval step {step}] valid is_scored wp={rv['weighted_pearson']:.4f} (t0 {rv['t0']:.4f}, "
                     f"t1 {rv['t1']:.4f}, rows {rv['selected_rows']:,}) | valid all>=99 wp={rva['weighted_pearson']:.4f} "
                     f"(t0 {rva['t0']:.4f}, t1 {rva['t1']:.4f}) | train-holdout wp={rh['weighted_pearson']:.4f} "
                     f"(t0 {rh['t0']:.4f}, t1 {rh['t1']:.4f}) | valid eval {tv:.1f}s, holdout eval {th:.1f}s "
                     f"(all {len(specs)} models)")
            if rv["weighted_pearson"] > sp["best"]:
                sp["best"] = rv["weighted_pearson"]
                save(sp, f"{CKPT_DIR}/{sp['run_name']}_best.pt", rv, rva, rh)
                mlog(sp, f"  saved new best ({sp['best']:.4f}) to {CKPT_DIR}/{sp['run_name']}_best.pt")
            if final:
                save(sp, f"{CKPT_DIR}/{sp['run_name']}_final.pt", rv, rva, rh)
                mlog(sp, f"  saved final to {CKPT_DIR}/{sp['run_name']}_final.pt")

    epoch = 0
    for sp in specs:
        sp["model"].train()
    while step < total_steps:
        order = rng.permutation(train_idx)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        loader = iter(Prefetcher(train_fit, batches))
        while True:
            tw = time.time()
            b = next(loader, None)
            t_wait += time.time() - tw
            if b is None:
                break
            x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
            y = torch.from_numpy(b["targets"]).to(device)
            m_all = {}  # per alpha: (B,T,1) mask weights, or None without --mask-p
            if mean_p is not None:
                p = torch.from_numpy(b["maskp"]).to(device).float()
            for sp in specs:
                a = sp["alpha_"]
                if a not in m_all:
                    m_all[a] = None if mean_p is None else (a + (1 - a) * (p / mean_p).clamp(max=20.0))[:, :, None]
            aux = {}
            for sp in specs:
                key = tuple(sp["offsets"])
                if key and key not in aux:
                    aux[key] = aux_targets(y, sp["offsets"])
                sp["h"] = None
            for s in range(0, SEQ_LEN, args.chunk):
                if step >= total_steps:
                    break
                ts = time.time()
                valid_mask = (steps_in_chunk[s:s + args.chunk] >= WARMUP).float()[None, :, None]
                for sp in specs:
                    tm = time.time()
                    m = None if m_all[sp["alpha_"]] is None else m_all[sp["alpha_"]][:, s:s + args.chunk]
                    pred, sp["h"] = sp["model"](x[:, s:s + args.chunk], sp["h"])
                    if sp["offsets"]:
                        y_aux, in_range = aux[tuple(sp["offsets"])]
                        main = loss_fn(pred[:, :, :2], y[:, s:s + args.chunk], valid_mask, m)
                        al = torch.stack(aux_loss(pred, y_aux[:, s:s + args.chunk], in_range[s:s + args.chunk],
                                                  valid_mask, m)).sum()
                        loss = main + sp["aux_weight"] * al
                        sp["run_main"] += main.item()
                        sp["run_aux"] += al.item()
                    else:
                        loss = loss_fn(pred, y[:, s:s + args.chunk], valid_mask, m)
                    sp["opt"].zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sp["model"].parameters(), 1.0)
                    sp["opt"].step()
                    sp["sched"].step()
                    sp["h"] = detach_state(sp["h"])
                    sp["run_loss"] += loss.item()  # .item() synchronises, so the timing below is real GPU time
                    dt = time.time() - tm
                    sp["t_model"] += dt
                    sp["t_since"] += dt
                step += 1
                run_n += 1
                t_train += time.time() - ts
                if step % 100 == 0 or (args.smoke and step % 10 == 0):
                    el = time.time() - t_last
                    for sp in specs:
                        parts = (f" (main {sp['run_main'] / run_n:.4f}, aux sum {sp['run_aux'] / run_n:.4f})"
                                 if sp["offsets"] else "")
                        mlog(sp, f"epoch {epoch} step {step}/{total_steps} loss {sp['run_loss'] / run_n:.4f}{parts} "
                                 f"lr {sp['sched'].get_last_lr()[0]:.2e} | {run_n / el:.1f} steps/s wall, "
                                 f"{run_n * args.batch * args.chunk / el:,.0f} seq-steps/s | "
                                 f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
                    log(f"[all] step {step}: {run_n / el:.1f} steps/s wall ({len(specs) * run_n / el:.1f} model-steps/s); "
                        f"per-model compute " + ", ".join(f"{sp['name']} {run_n / max(sp['t_since'], 1e-9):.1f}"
                                                          for sp in specs)
                        + f" steps/s; loader wait so far {t_wait:.1f}s")
                    for sp in specs:
                        sp.update(run_loss=0.0, run_main=0.0, run_aux=0.0, t_since=0.0)
                    run_n, t_last = 0, time.time()
                if step % args.eval_every == 0:
                    do_eval()
                    t_last = time.time()
            if step >= total_steps:
                break
        epoch += 1

    do_eval(final=True)
    for sp in specs:
        mlog(sp, f"done. best valid wp {sp['best']:.4f}; own train compute {sp['t_model']:.1f}s for {step} steps "
                 f"({step / max(sp['t_model'], 1e-9):.2f} steps/s)")
    log(f"done. {len(specs)} models x {step} steps; pure train compute {t_train:.1f}s "
        f"({step / max(t_train, 1e-9):.2f} steps/s for all models together, "
        f"{len(specs) * step / max(t_train, 1e-9):.2f} model-steps/s); loader wait {t_wait:.1f}s; gpu peak allocated "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
