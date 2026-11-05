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
      lr=R        peak learning rate of the cosine schedule (default: the shared --lr, 1e-3)
      hidden=H    hidden size (default: the shared --hidden)
      ema=D       EMA decay of the weights (default 0 = off). When D > 0 an EMA copy of the parameters is
                  updated after every optimizer step (p_ema += (1 - D) * (p - p_ema), starting from the
                  initial weights); evaluation and the best / final checkpoints use the EMA weights (a standard
                  state dict, so the exporters work unchanged), and the raw weights are evaluated in the same
                  pass and logged on a separate "raw (non-EMA)" line for comparison.
      noise=S     Gaussian input noise, training only (default 0 = off): N(0, S^2) is added to the standardised
                  input columns of this model's training chunks (all but the 12 fill-value flag columns). The
                  noise comes from the model's own torch.Generator (seed NOISE_SEED + init seed), so data order,
                  initialisation and every other random stream are unchanged; evaluation never adds noise.
      kd=K        distillation weight (default 0 = off; needs no aux): the training loss becomes
                  (1 - K) * main_loss + K * teacher_loss, where teacher_loss is the same weighted MSE (same row
                  weights, steps >= 99 mask and normalisation) between the clipped output and the teacher prediction
                  for the same rows (make_teacher.py: v18c + v19a, clipped, float16; read with the batch from --teacher)
      wpow=P      row-weight power (default 1; needs no aux): the row weight becomes (|clipped y|^P + 0.05) times the
                  mask weight instead of (|clipped y| + 0.05). With kd 0 and wpow 1 the loss is train_v5m.loss_fn.
      chunk=C     training chunk length in steps (default: the shared --chunk, 1000); a multiple of --chunk that
                  divides 20,000. The hidden state is carried across chunks as before and the batch stays 16
                  sequences, so the model takes 20,000 / C optimizer steps per batch (its cosine schedule and step
                  count follow that). Batches are still cut into --chunk pieces: the model steps when its chunk
                  ends, on the last C steps. Evaluation always runs in --chunk pieces (the state is carried, so the
                  outputs are the same). Eval lines and "step" in the checkpoint count --chunk pieces, as for every
                  model ("model_step" holds the model's own optimizer steps).
      fdrop=P     input-column dropout, training only (default 0 = off): each of the 156 standardised input columns
                  (never the 12 flag columns) is zeroed independently per sequence with probability P, constant over
                  the training chunk; kept columns are scaled by 1 / (1 - P). Own torch.Generator (seed
                  FDROP_SEED + init seed), so no other random stream moves.
      cls=W       auxiliary classification head (default 0 = off): a second linear layer on the GRU output gives one
                  21-way softmax per target over 21 equal bins of the clipped range [-2, 2]; its cross-entropy, with
                  the main loss's row weights, steps >= 99 mask and normalisation, is added with weight W. The head
                  is initialised from its own seed (CLS_SEED + init seed), trained with the model's optimizer, and
                  not saved: evaluation and the checkpoints use the main outputs only.
      drop=G[+G]  input-group ablation (default none): the model's input columns of the named groups (DROP_GROUPS,
                  built from the column names below) are set to 0 in training AND evaluation, i.e. the training mean
                  of a standardised column (flag columns become 0 = "not filled"). The input size stays 168, so the
                  checkpoints load as usual, but the exporters do not zero the columns: these are ablations, not
                  submissions. Groups (indices into train_v5m.build_inputs' 168 columns):
                    i1       instrument i1: i1_p0-21, their 22 differences, i1_v0-21, i1_dp0-3, i1_dv0-3, i1_dv flags (78)
                    i1px     i1_p0-21 and their differences (44)
                    vol      i0_v0-21 and i1_v0-21 (44; the volume-like columns have no fill flags)
                    trade    i0/i1 dp0-3 and dv0-3 and the 8 dv fill flags (24)
                    anon     a0-a7 and the a0 / a5 / a6 / a7 fill flags (12; not instrument-specific, so not in i1)
                    pxlevel  i0_p0-21 and i1_p0-21 (44; their differences are kept)
      xvd=S       extra input columns (default none): one-step changes of the volume-like columns, S = all (i0_v0-21
                  and i1_v0-21, 44 columns) or i1 (i1_v0-21, 22). change[t] = v[t] - v[t-1] of the cached values,
                  set to 0 at step 0 and wherever v[t] or v[t-1] is the fill value +-5.1992 (volume columns have no
                  fill flags, and a flag is set exactly when |value| is the fill value, so this covers both rules).
      xagg=1      extra input columns (default 0): OFI-style aggregates per instrument from those (zeroed) changes:
                  sum over the bid-side columns v0-10, sum over the ask-side columns v11-21, and bid - ask
                  (docs/data_overview.md: "bid and ask are told apart by index only (0-10 bid, 11-21 ask)"); 6 columns.
                  The extra columns are appended after the 168 (xvd first, then xagg) and standardised as the price
                  differences are: (value - mean) / std with mean and std from the same --n-stats training sequences,
                  steps >= 1 (compute_stats' procedure). Input noise and fdrop do not touch them; drop= indices refer
                  to the 168. The model's input size grows (168 + 44 / 22, + 6); checkpoints record "n_in", the extra
                  column names and their stats ("x_stats"), and load_model() rebuilds such a model. The exporters
                  do not build the extra columns: these models are experiments, not submissions yet.
      unitemb=E   unit-class input (default 0 = off): the sequence's target unit class (0-3 for 0.1254 / 0.2413 /
                  0.4685 / 0.6569) goes through a learned embedding of size E (torch.nn.Embedding with 5 rows, row 4
                  = "unknown"), appended to the input at every step after the 168 (and after any extra columns); the
                  input size grows by E. Labels: unit_labels.py (the true unit, read from the targets). The embedding
                  is initialised from its own seed (UNIT_SEED + init seed, on a forked CPU RNG, after the model),
                  trained with the model's optimizer (same weight decay) and saved in the checkpoints as
                  "unit_emb" (state dict) with "unit" (settings) and "n_in"; load_model() attaches it as
                  model.unit_emb. Not with ema.
      unitpred=1  with unitemb: use the causally PREDICTED class instead (unit_pred.py: LightGBM on the first
                  UNIT_FROM = 20 steps; train sequences out-of-fold, valid from the all-train classifier), and the
                  "unknown" class for steps < 20, in training and evaluation alike, exactly as at inference.
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

--unit-filter U (e.g. 0.6569) trains specialists for one target unit (labels from unit_labels.py, i.e.
unit_probe.py's labelling, including the 48 train sequences labelled from their full target sequence): after the
usual holdout split and valid subset draw (same RNG use), the training, holdout and valid sequences are cut to
those of unit U. The optimizer steps and the cosine schedule follow the filtered training set (--epochs are epochs
of it), the input stats come from its first --n-stats sequences, and the mask weights keep the unfiltered mean_p
(so every row gets the same weight as in a generalist run). Every training batch is checked to be of unit U (all
nonzero |targets| integer multiples of U), and the trained sequences are counted per unit. Evaluation reports
the in-group scores (global weighted Pearson over the unit's valid sequences, is_scored and all rows >= 99).

Usage (one process, one GPU):
  CUDA_VISIBLE_DEVICES=0 nohup python train_multi.py --models "..." --hidden 128 --epochs 5 --n-val 1873 \\
      --n-holdout 300 --eval-every 12800 --mask-p ~/work/cache/train_maskp.npy --alpha 0.5 > ~/work/x.out 2>&1 &
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import logging
import math
import random
import time

import numpy as np
import torch

from train_v5m import (ARCHS, CKPT_DIR, FILL, FLAG_COLS, GPU_MEM_CAP_GB, N_IN, OTHER_COLS, PRICE_COLS, SEQ_LEN,
                       WARMUP, GlobalAccumulator, Prefetcher, aux_loss, aux_targets, build_inputs, compute_stats,
                       detach_state, load_split, loss_fn, parse_offsets, read_batch, set_seed)

N_CONT = len(PRICE_COLS) * 2 + len(OTHER_COLS)  # standardised input columns (build_inputs puts the flags last)


OPTIONS = {"alpha": float, "layers": int, "dropout": float, "wd": float, "lr": float, "hidden": int, "ema": float,
           "noise": float, "kd": float, "wpow": float,
           "chunk": int, "fdrop": float, "cls": float, "drop": str, "xvd": str, "xagg": int,
           "unitemb": int, "unitpred": int}
NOISE_SEED = 1_000_000  # noise generator seed = NOISE_SEED + the model's init seed
FDROP_SEED = 2_000_000  # input-column dropout generator seed = FDROP_SEED + init seed
CLS_SEED = 3_000_000  # classification head init seed = CLS_SEED + init seed
N_BINS = 21  # classification head bins over the clipped target range [-2, 2]
UNIT_SEED = 4_000_000  # unit embedding init seed = UNIT_SEED + init seed
UNIT_FROM = 20  # unitpred=1: the predicted class is used from this step on ("unknown" before)
UNIT_UNKNOWN = 4  # embedding row of the "unknown" class (rows 0-3 = the four units)


def input_names():
    """Names of the 168 build_inputs columns: prices, their differences, the other columns, then the fill flags."""
    return (list(PRICE_COLS) + [f"d_{c}" for c in PRICE_COLS] + list(OTHER_COLS) + [f"flag_{c}" for c in FLAG_COLS])


def drop_groups():
    """Input-group index lists for drop=, from the column names (parquet schema via train_v5m)."""
    names = input_names()
    assert len(names) == N_CONT + len(FLAG_COLS) == 168
    base = [n[2:] if n.startswith("d_") else n[5:] if n.startswith("flag_") else n for n in names]
    kind = ["diff" if n.startswith("d_") else "flag" if n.startswith("flag_") else "col" for n in names]

    def is_(b, inst=None, grp=None):
        if inst is not None and not b.startswith(inst + "_"):
            return False
        if grp is None:
            return True
        g = b.split("_", 1)[1] if "_" in b else b
        return g.rstrip("0123456789") == grp
    g = {"i1": [j for j, b in enumerate(base) if is_(b, "i1")],
         "i1px": [j for j, b in enumerate(base) if is_(b, "i1", "p")],
         "vol": [j for j, b in enumerate(base) if is_(b, None, "v") and "_" in b],
         "trade": [j for j, b in enumerate(base) if "_" in b and (is_(b, None, "dp") or is_(b, None, "dv"))],
         "anon": [j for j, b in enumerate(base) if b.startswith("a") and b[1:].isdigit()],
         "pxlevel": [j for j, (b, k) in enumerate(zip(base, kind)) if k == "col" and "_" in b and is_(b, None, "p")]}
    for k, v in g.items():
        assert v, f"drop group {k} is empty"
    return g, names


DROP_GROUPS, INPUT_NAMES = drop_groups()


def drop_index(spec):
    """'i1+vol' -> sorted index list of the union of the named groups."""
    idx = set()
    for k in spec.split("+"):
        assert k in DROP_GROUPS, f"unknown drop group {k!r} (known: {sorted(DROP_GROUPS)})"
        idx |= set(DROP_GROUPS[k])
    return sorted(idx)


def apply_drop(x, idx):
    """x (B, T, 168) with the columns idx set to 0 (a copy; idx None = unchanged, same tensor)."""
    if idx is None:
        return x
    x = x.clone()
    x[:, :, idx] = 0
    return x


VOL_COLS = [f"i{i}_v{j}" for i in range(2) for j in range(22)]  # the 44 volume-like columns, i0 then i1
VOL_IDX = [OTHER_COLS.index(c) for c in VOL_COLS]  # into the 68 other features
assert not set(VOL_COLS) & set(FLAG_COLS)
EXTRA_NAMES = ([f"xvd_{c}" for c in VOL_COLS]
               + [f"xagg_i{i}_{k}" for i in range(2) for k in ("bid", "ask", "bid_minus_ask")])  # 50 extra columns


def extra_raw(other):
    """other (B,T,68) -> (B,T,50) f32 unstandardised extra columns: 44 volume changes, then per instrument the bid sum
    (v0-10), ask sum (v11-21) and their difference of the changes. 0 at step 0 and where either value is the fill."""
    v = other[:, :, VOL_IDX].float()
    fill = v.abs() == FILL
    dv = torch.zeros_like(v)
    dv[:, 1:] = v[:, 1:] - v[:, :-1]
    dv[:, 1:][fill[:, 1:] | fill[:, :-1]] = 0
    agg = []
    for i in range(2):
        bid = dv[:, :, 22 * i:22 * i + 11].sum(2)
        ask = dv[:, :, 22 * i + 11:22 * i + 22].sum(2)
        agg += [bid, ask, bid - ask]
    return torch.cat([dv, torch.stack(agg, 2)], 2)


def extra_index(xvd, xagg):
    """Indices into the 50 EXTRA_NAMES columns for xvd= / xagg= (None = no extra columns)."""
    idx = {"": [], "all": list(range(44)), "i1": list(range(22, 44))}[xvd] + (list(range(44, 50)) if xagg else [])
    return idx or None


def compute_extra_stats(data, idx, device):
    """compute_stats' procedure for the extra columns: per sequence, steps >= 1, float64 sums."""
    s1 = torch.zeros(len(EXTRA_NAMES), dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    n = 0
    for i in idx:
        b = read_batch(data, [i])
        x = extra_raw(torch.from_numpy(b["features"]).to(device))[0, 1:].double()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        n += x.shape[0]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-6)
    return {"mean": mean.float(), "std": std.float()}


def build_extra(other, xstats):
    """(B,T,50) standardised extra columns."""
    return (extra_raw(other) - xstats["mean"]) / xstats["std"]


def with_extra(x, xe, idx):
    """x (B,T,168) with the extra columns idx of xe appended (idx None = x itself)."""
    return x if idx is None else torch.cat([x, xe[:, :, idx]], 2)


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
        assert m["ema"] is None or 0 <= m["ema"] < 1, f"ema {m['ema']} not in [0, 1)"
        assert m["noise"] is None or m["noise"] >= 0, f"noise {m['noise']} < 0"
        assert m["kd"] is None or 0 <= m["kd"] <= 1, f"kd {m['kd']} not in [0, 1]"
        assert m["wpow"] is None or m["wpow"] > 0, f"wpow {m['wpow']} <= 0"
        assert not aux or (m["kd"] or 0) == 0 and (m["wpow"] or 1) == 1, "kd / wpow need a model without aux heads"
        assert m["fdrop"] is None or 0 <= m["fdrop"] < 1, f"fdrop {m['fdrop']} not in [0, 1)"
        assert m["cls"] is None or m["cls"] >= 0, f"cls {m['cls']} < 0"
        assert not aux or (m["cls"] or 0) == 0, "cls needs a model without aux heads"
        assert m["xvd"] in (None, "all", "i1"), f"xvd {m['xvd']!r} not all / i1"
        assert m["xagg"] in (None, 0, 1), f"xagg {m['xagg']} not 0 / 1"
        assert m["unitemb"] is None or m["unitemb"] >= 0, f"unitemb {m['unitemb']} < 0"
        assert m["unitpred"] in (None, 0, 1), f"unitpred {m['unitpred']} not 0 / 1"
        assert not m["unitpred"] or (m["unitemb"] or 0) > 0, "unitpred needs unitemb > 0"
        assert not m["unitemb"] or (m["ema"] or 0) == 0, "unitemb is not supported with ema"
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


def kd_loss_fn(pred, y, t, valid, m, kd, wpow):
    """train_v5m.loss_fn with the row weight (|yc|^wpow + 0.05) * valid * m, mixed with the same weighted MSE against
    the teacher t (None when kd == 0). Returns (loss, main loss, teacher loss or None)."""
    yc = y.clamp(-2, 2)
    pc = pred.clamp(-2, 2)
    w = ((yc.abs() if wpow == 1 else yc.abs() ** wpow) + 0.05) * valid
    if m is not None:
        w = w * m
    den = w.sum().clamp_min(1e-8)
    main = (w * (pc - yc) ** 2).sum() / den
    if kd == 0:
        return main, main, None
    teach = (w * (pc - t.clamp(-2, 2)) ** 2).sum() / den
    return (1 - kd) * main + kd * teach, main, teach


def cls_loss_fn(logits, y, valid, m, wpow):
    """Cross-entropy of the (B, T, 2 * N_BINS) classification logits against the bin of the clipped target, with the
    main loss's row weights (|yc|^wpow + 0.05) * valid * m and normalisation."""
    yc = y.clamp(-2, 2)
    label = ((yc + 2) / 4 * N_BINS).floor().clamp(0, N_BINS - 1).long()
    logp = torch.log_softmax(logits.view(*logits.shape[:2], 2, N_BINS), dim=-1)
    nll = -logp.gather(-1, label[..., None])[..., 0]
    w = ((yc.abs() if wpow == 1 else yc.abs() ** wpow) + 0.05) * valid
    if m is not None:
        w = w * m
    return (w * nll).sum() / w.sum().clamp_min(1e-8)


def build_model(sp, hidden, n_in=N_IN):
    n_out, layers, dropout = 2 + 2 * len(sp["offsets"]), sp["layers"] or 2, sp["dropout"] or 0.0
    if dropout == 0:
        if n_in == N_IN:
            return ARCHS[sp["arch"]](hidden=hidden, layers=layers, n_out=n_out)
        return ARCHS[sp["arch"]](n_in=n_in, hidden=hidden, layers=layers, n_out=n_out)
    return RNNModel(sp["arch"], n_in, hidden, layers, n_out, dropout)


def load_model(path, device="cpu"):
    """Rebuild a train_multi checkpoint's model (input size from "n_in", 168 for older checkpoints).
    Returns (model in eval mode, checkpoint dict). A unitemb= model also gets its embedding as model.unit_emb."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    sp = {"offsets": parse_offsets(a.get("aux", "")), "layers": a.get("layers", 2), "dropout": 0.0, "arch": a["arch"]}
    model = build_model(sp, a["hidden"], ck.get("n_in", N_IN)).to(device)
    model.load_state_dict(ck["model"])
    if "unit_emb" in ck:
        emb = torch.nn.Embedding(UNIT_UNKNOWN + 1, ck["unit"]["dim"]).to(device)
        emb.load_state_dict(ck["unit_emb"])
        model.unit_emb = emb
    return model.eval(), ck


def unit_input(emb, cls, s, e, pred):
    """(B, e - s, E) embedding of the per-sequence unit classes cls (B,) long tensor for steps s..e-1; with pred,
    the "unknown" class for steps < UNIT_FROM."""
    c = cls[:, None].expand(-1, e - s)
    if pred and s < UNIT_FROM:
        steps = torch.arange(s, e, device=cls.device)[None, :]
        c = torch.where(steps < UNIT_FROM, torch.full_like(c, UNIT_UNKNOWN), c)
    return emb(c)


@torch.no_grad()
def evaluate_multi(models, data, idx, stats, device, chunk, use_is_scored, batch=25, drops=None, extras=None,
                   xstats=None, units=None):
    """train_v5m.evaluate for K models; each batch is read once. Returns a list of per-model results.
    drops: per model None or the input column indices to zero (drop=). extras: per model None or the extra
    column indices to append (xvd= / xagg=; xstats their standardisation). units: per model None or
    (embedding, class per sequence of data, pred flag) for unitemb= (appended last)."""
    for m in models:
        m.eval()
    for u in units or []:
        if u is not None:
            u[0].eval()
    acc = [GlobalAccumulator() for _ in models]
    acc_all = [GlobalAccumulator() for _ in models]
    need = np.arange(SEQ_LEN) >= WARMUP
    batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    for bi, b in enumerate(Prefetcher(data, batches)):
        x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
        xe = None if extras is None else build_extra(torch.from_numpy(b["features"]).to(device), xstats)
        for i, model in enumerate(models):
            h = None
            preds = []
            xi = x if drops is None else apply_drop(x, drops[i])
            ei = None if extras is None else extras[i]
            ui = None if units is None else units[i]
            if ui is not None:
                cls = torch.from_numpy(np.asarray(ui[1][batches[bi]], dtype=np.int64)).to(device)
            for s in range(0, SEQ_LEN, chunk):
                xs = xi[:, s:s + chunk] if ei is None else with_extra(xi[:, s:s + chunk], xe[:, s:s + chunk], ei)
                if ui is not None:
                    xs = torch.cat([xs, unit_input(ui[0], cls, s, s + xs.shape[1], ui[2])], 2)
                p, h = model(xs, h)
                preds.append(p[:, :, :2])  # main targets only; auxiliary outputs are never scored
            preds = torch.cat(preds, 1).float().cpu().numpy()
            for k in range(len(preds)):
                mask = (b["is_scored"][k] & need) if use_is_scored else need
                acc[i].add(b["targets"][k], preds[k], mask)
                if use_is_scored:
                    acc_all[i].add(b["targets"][k], preds[k], need)
    for m in models:
        m.train()
    for u in units or []:
        if u is not None:
            u[0].train()
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
    ap.add_argument("--teacher", default="/home/jovyan/work/cache/train_teacher_t{t}.npy",
                    help="teacher prediction files for kd= ({t} = 0, 1), make_teacher.py")
    ap.add_argument("--unit-filter", type=float, default=None,
                    help="train / evaluate only on the sequences of this target unit (e.g. 0.6569)")
    args = ap.parse_args()

    specs = parse_models(args.models)
    for sp in specs:  # resolved settings (the trailing _ marks resolved values)
        sp.update(alpha_=args.alpha if sp["alpha"] is None else sp["alpha"], layers_=sp["layers"] or 2,
                  dropout_=sp["dropout"] or 0.0, wd_=args.wd if sp["wd"] is None else sp["wd"],
                  lr_=args.lr if sp["lr"] is None else sp["lr"], hidden_=sp["hidden"] or args.hidden,
                  ema_=sp["ema"] or 0.0, noise_=sp["noise"] or 0.0,
                  kd_=sp["kd"] or 0.0, wpow_=1.0 if sp["wpow"] is None else sp["wpow"],
                  chunk_=sp["chunk"] or args.chunk, fdrop_=sp["fdrop"] or 0.0, cls_=sp["cls"] or 0.0,
                  drop_=sp["drop"] or "", xvd_=sp["xvd"] or "", xagg_=sp["xagg"] or 0,
                  unitemb_=sp["unitemb"] or 0, unitpred_=sp["unitpred"] or 0)
        sp["drop_idx"] = drop_index(sp["drop_"]) if sp["drop_"] else None
        sp["extra_idx"] = extra_index(sp["xvd_"], sp["xagg_"])
        sp["n_in"] = N_IN + (0 if sp["extra_idx"] is None else len(sp["extra_idx"])) + sp["unitemb_"]
        assert sp["chunk_"] % args.chunk == 0 and SEQ_LEN % sp["chunk_"] == 0, \
            f"chunk {sp['chunk_']}: must be a multiple of --chunk {args.chunk} and divide {SEQ_LEN}"
        sp["custom_loss"] = sp["kd_"] > 0 or sp["wpow_"] != 1
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
                 lr=sp["lr_"], hidden=sp["hidden_"], ema=sp["ema_"], noise=sp["noise_"], kd=sp["kd_"], wpow=sp["wpow_"],
                 train_chunk=sp["chunk_"], fdrop=sp["fdrop_"], cls=sp["cls_"], drop=sp["drop_"],
                 drop_idx=sp["drop_idx"], xvd=sp["xvd_"], xagg=sp["xagg_"], unitemb=sp["unitemb_"],
                 unitpred=sp["unitpred_"], n_in=sp["n_in"],
                 extra_cols=None if sp["extra_idx"] is None else [EXTRA_NAMES[j] for j in sp["extra_idx"]],
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
    val_idx = np.sort(rng.choice(valid["prices"].shape[0], args.n_val, replace=False))
    pre_idx = train_idx  # unfiltered training sequences (mean_p of the mask weights)
    unit_k = unit_value = unit_tr = None
    if args.unit_filter is not None:
        from unit_labels import unit_index, unit_labels
        from unit_probe import UNITS
        unit_k = unit_index(args.unit_filter)
        unit_value = UNITS[unit_k]
        unit_tr, unit_va = unit_labels("train"), unit_labels("valid")
        assert len(unit_tr) == train["prices"].shape[0] and len(unit_va) == valid["prices"].shape[0]
        log(f"unit filter {args.unit_filter} -> unit {unit_value}: train seqs {len(train_idx)} -> "
            f"{np.sum(unit_tr[train_idx] == unit_k)} (per unit " + ", ".join(
                f"{u} {np.sum(unit_tr[train_idx] == k)}" for k, u in enumerate(UNITS)) + f"), holdout {len(holdout_idx)} -> "
            f"{np.sum(unit_tr[holdout_idx] == unit_k)}, valid {len(val_idx)} -> {np.sum(unit_va[val_idx] == unit_k)}")
        train_idx = train_idx[unit_tr[train_idx] == unit_k]
        holdout_idx = holdout_idx[unit_tr[holdout_idx] == unit_k]
        val_idx = val_idx[unit_va[val_idx] == unit_k]
    if args.smoke:
        train_idx = train_idx[:20]
        pre_idx = pre_idx[:20]
    log(f"train seqs {len(train_idx)}, train holdout {len(holdout_idx)}, valid subset {len(val_idx)}")
    use_unit = any(sp["unitemb_"] > 0 for sp in specs)
    if use_unit:  # per-sequence unit classes: true (unit_labels.py) or causally predicted (unit_pred.py)
        from unit_labels import unit_labels
        from unit_pred import pred_path, unit_pred
        lab = {"true": (unit_labels("train"), unit_labels("valid"))}
        if any(sp["unitpred_"] for sp in specs):
            lab["pred"] = (unit_pred("train"), unit_pred("valid"))
        for k, (ltr, lva) in lab.items():
            assert len(ltr) == train["prices"].shape[0] and len(lva) == valid["prices"].shape[0], k
            assert ltr.min() >= 0 and ltr.max() < UNIT_UNKNOWN and lva.min() >= 0 and lva.max() < UNIT_UNKNOWN, k
        if "pred" in lab:
            (ttr, tva), (ptr, pva) = lab["true"], lab["pred"]
            log(f"unit classes: predicted ({pred_path('train')}, out-of-fold; {pred_path('valid')}) vs true "
                f"(unit_labels.py): train accuracy {np.mean(ptr == ttr):.5f} ({np.sum(ptr != ttr)} wrong), on the "
                f"training seqs {np.mean(ptr[train_idx] == ttr[train_idx]):.5f}, holdout "
                f"{np.mean(ptr[holdout_idx] == ttr[holdout_idx]):.5f}; valid {np.mean(pva == tva):.5f}, valid subset "
                f"{np.mean(pva[val_idx] == tva[val_idx]):.5f}")
        for sp in specs:
            if sp["unitemb_"] > 0:
                sp["unit_tr"], sp["unit_va"] = lab["pred" if sp["unitpred_"] else "true"]

    train_fit = train
    mean_p = None
    if args.mask_p is not None:
        mp = np.load(os.path.expanduser(args.mask_p), mmap_mode="r")
        assert mp.shape == train["prices"].shape[:2], f"mask-p shape {mp.shape} != train {train['prices'].shape[:2]}"
        train_fit = dict(train, maskp=mp)
        srt = np.sort(pre_idx)
        tot = sum(np.asarray(mp[srt[i:i + 500], WARMUP:]).sum(dtype=np.float64) for i in range(0, len(srt), 500))
        mean_p = tot / (len(srt) * (SEQ_LEN - WARMUP))
        log(f"mask weights from {args.mask_p}: alpha {args.alpha}, mean_p {mean_p:.6f} over {len(srt)} train seqs "
            f"(steps >= {WARMUP}); m = alpha + (1 - alpha) * min(p / mean_p, 20)"
            + ("" if unit_k is None else " (unfiltered training set, as a generalist run)"))

    use_teacher = any(sp["kd_"] > 0 for sp in specs)
    if use_teacher:
        for t in range(2):
            tp = args.teacher.format(t=t)
            tm = np.load(tp, mmap_mode="r")
            assert tm.shape == train["prices"].shape[:2] and tm.dtype == np.float16, f"teacher {tp} {tm.shape} {tm.dtype}"
            train_fit = dict(train_fit, **{f"teacher_t{t}": tm})
        log(f"teacher predictions from {args.teacher} (read with each training batch)")

    grp = "" if unit_k is None else f" | in-group: unit {unit_value} only ({len(val_idx)} valid seqs)"
    seen = np.zeros(4, np.int64)  # trained sequences per unit (with --unit-filter)
    t0 = time.time()
    stats = compute_stats(train, np.sort(train_idx[:args.n_stats]), device)
    log(f"input stats from {min(args.n_stats, len(train_idx))} seqs in {time.time() - t0:.1f}s")
    use_extra = any(sp["extra_idx"] is not None for sp in specs)
    xstats = None
    if use_extra:
        t0 = time.time()
        xstats = compute_extra_stats(train, np.sort(train_idx[:args.n_stats]), device)
        log(f"extra-column stats from the same {min(args.n_stats, len(train_idx))} seqs in {time.time() - t0:.1f}s: "
            + ", ".join(f"{n} {m:.4g}/{s:.4g}" for n, m, s in zip(EXTRA_NAMES, xstats["mean"].tolist(),
                                                                   xstats["std"].tolist())) + " (mean/std)")

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
        sp["model"] = build_model(sp, sp["hidden_"], sp["n_in"]).to(device)
        # EMA copy (deepcopy draws no random numbers, so the other models' streams are unaffected)
        sp["ema_model"] = copy.deepcopy(sp["model"]) if sp["ema_"] > 0 else None
        if sp["ema_model"] is not None:
            for q in sp["ema_model"].parameters():
                q.requires_grad_(False)
            for mod in sp["ema_model"].modules():
                if isinstance(mod, torch.nn.RNNBase):
                    mod.flatten_parameters()  # keep the copy's cuDNN weights in one chunk (lerp_ is in place)
        # input noise: the model's own generator, so no other random stream moves
        sp["noise_gen"] = None
        if sp["noise_"] > 0:
            sp["noise_gen"] = torch.Generator(device=device)
            sp["noise_gen"].manual_seed(NOISE_SEED + sp["seed"])
        sp["fdrop_gen"] = None
        if sp["fdrop_"] > 0:
            sp["fdrop_gen"] = torch.Generator(device=device)
            sp["fdrop_gen"].manual_seed(FDROP_SEED + sp["seed"])
        sp["cls_head"] = None
        if sp["cls_"] > 0:  # own seed on a forked CPU RNG, so the global torch stream is untouched
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(CLS_SEED + sp["seed"])
                sp["cls_head"] = torch.nn.Linear(sp["hidden_"], 2 * N_BINS)
            sp["cls_head"] = sp["cls_head"].to(device)
        sp["unit_emb"] = None
        if sp["unitemb_"] > 0:  # own seed on a forked CPU RNG, so the global torch stream is untouched
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(UNIT_SEED + sp["seed"])
                sp["unit_emb"] = torch.nn.Embedding(UNIT_UNKNOWN + 1, sp["unitemb_"])
            sp["unit_emb"] = sp["unit_emb"].to(device)
        sp["total"] = int(args.epochs * batches_per_epoch * (SEQ_LEN // sp["chunk_"]))
        if args.max_steps is not None:
            sp["total"] = min(sp["total"], args.max_steps * args.chunk // sp["chunk_"])
        mlog(sp, f"settings: alpha {sp['alpha_']}, layers {sp['layers_']}, dropout {sp['dropout_']}, "
                 f"weight decay {sp['wd_']}, lr {sp['lr_']}, hidden {sp['hidden_']}, ema {sp['ema_']}"
                 + ("" if sp["noise_gen"] is None else f", input noise {sp['noise_']} (training only, generator seed "
                                                       f"{NOISE_SEED + sp['seed']}, {N_CONT} standardised columns)")
                 + (f", distillation kd {sp['kd_']} (loss = (1 - kd) * main + kd * teacher)" if sp["kd_"] > 0 else "")
                 + (f", row weight (|y|^{sp['wpow_']} + 0.05) * m" if sp["wpow_"] != 1 else "")
                 + (f", training chunk {sp['chunk_']} ({SEQ_LEN // sp['chunk_']} optimizer steps per batch)"
                    if sp["chunk_"] != args.chunk else "")
                 + (f", input-column dropout {sp['fdrop_']} (training only, per sequence and chunk, {N_CONT} "
                    f"standardised columns, generator seed {FDROP_SEED + sp['seed']})" if sp["fdrop_"] > 0 else "")
                 + (f", classification head weight {sp['cls_']} ({N_BINS} bins on [-2, 2] per target, init seed "
                    f"{CLS_SEED + sp['seed']}, not saved)" if sp["cls_"] > 0 else "")
                 + (f", input groups dropped (set to 0 in training and evaluation): {sp['drop_']} = "
                    f"{len(sp['drop_idx'])} columns {sp['drop_idx']}" if sp["drop_idx"] else "")
                 + (f", extra input columns (xvd {sp['xvd_'] or 'none'}, xagg {sp['xagg_']}): {len(sp['extra_idx'])} "
                    f"appended, input size {sp['n_in']}: {[EXTRA_NAMES[j] for j in sp['extra_idx']]}"
                    if sp["extra_idx"] is not None else "")
                 + (f", unit embedding size {sp['unitemb_']} ({'predicted class from step ' + str(UNIT_FROM) + ', unknown before' if sp['unitpred_'] else 'true class'}; "
                    f"init seed {UNIT_SEED + sp['seed']}), input size {sp['n_in']}" if sp["unit_emb"] is not None else ""))
        if sp["offsets"]:
            mlog(sp, f"auxiliary heads: offsets {sp['offsets']}, weight {sp['aux_weight']} each; "
                     f"head outputs {2 + 2 * len(sp['offsets'])}")
        mlog(sp, f"params: {sum(p.numel() for p in sp['model'].parameters()):,}")
        if sp["unit_emb"] is not None:
            mlog(sp, f"unit embedding params: {sum(p.numel() for p in sp['unit_emb'].parameters()):,}")
        sp["params"] = list(sp["model"].parameters()) + (
            [] if sp["cls_head"] is None else list(sp["cls_head"].parameters())) + (
            [] if sp["unit_emb"] is None else list(sp["unit_emb"].parameters()))
        sp["extra_params"] = sp["cls_head"] is not None or sp["unit_emb"] is not None
        sp["opt"] = torch.optim.AdamW(sp["params"] if sp["extra_params"] else sp["model"].parameters(),
                                      lr=sp["lr_"], weight_decay=sp["wd_"])
        sp["sched"] = torch.optim.lr_scheduler.LambdaLR(
            sp["opt"], lambda s, T=sp["total"]: 0.5 * (1 + math.cos(math.pi * min(s, T) / T)))
        sp.update(best=-float("inf"), run_loss=0.0, run_main=0.0, run_aux=0.0, run_teach=0.0, run_cls=0.0, run_k=0,
                  step=0, t_model=0.0, t_since=0.0)
    log(f"{batches_per_epoch} batches/epoch x {chunks_per_seq} chunks = {batches_per_epoch * chunks_per_seq} "
        f"steps/epoch; total optimizer steps {total_steps} per model")
    for sp in specs:
        cps = SEQ_LEN // sp["chunk_"]
        mlog(sp, f"{batches_per_epoch} batches/epoch x {cps} chunks = {batches_per_epoch * cps} "
                 f"steps/epoch; total optimizer steps {sp['total']}")

    steps_in_chunk = torch.arange(SEQ_LEN, device=device)
    step = 0
    run_n, t_train, t_last = 0, 0.0, time.time()
    t_wait = 0.0  # time the GPU loop waited for the loader

    def eval_model(sp):
        return sp["model"] if sp["ema_model"] is None else sp["ema_model"]

    def save(sp, path, rv, rva, rh):
        extra = {} if sp["extra_idx"] is None else {
            "n_in": sp["n_in"], "extra_cols": [EXTRA_NAMES[j] for j in sp["extra_idx"]],
            "x_stats": {k: v[sp["extra_idx"]] for k, v in xstats.items()}}
        if sp["unit_emb"] is not None:
            extra.update(n_in=sp["n_in"], unit_emb=sp["unit_emb"].state_dict(), unit={
                "dim": sp["unitemb_"], "pred": bool(sp["unitpred_"]), "from_step": UNIT_FROM if sp["unitpred_"] else 0,
                "unknown": UNIT_UNKNOWN, "classes": [0.125433, 0.241330, 0.468536, 0.656868],
                "labels": "unit_pred.py (LightGBM, first 20 steps)" if sp["unitpred_"] else "unit_labels.py (true)"})
        torch.save({"model": eval_model(sp).state_dict(), "stats": stats, "args": model_args(sp), "step": step, **extra,
                    "model_step": sp["step"],
                    "valid": rv, "valid_all": rva, "holdout": rh, "price_cols": PRICE_COLS, "other_cols": OTHER_COLS,
                    "flag_cols": FLAG_COLS}, path)

    def do_eval(final=False):
        torch.cuda.synchronize()
        te = time.time()
        # EMA models are evaluated with their EMA weights; their raw weights are appended for comparison
        raw = [sp for sp in specs if sp["ema_model"] is not None]
        models = [eval_model(sp) for sp in specs] + [sp["model"] for sp in raw]
        drops = [sp["drop_idx"] for sp in specs] + [sp["drop_idx"] for sp in raw]
        drops = None if all(d is None for d in drops) else drops
        extras = [sp["extra_idx"] for sp in specs] + [sp["extra_idx"] for sp in raw] if use_extra else None
        # unit models never have an EMA copy (parse_models), so the raw entries get None
        u_va = [None if sp["unit_emb"] is None else (sp["unit_emb"], sp["unit_va"], sp["unitpred_"]) for sp in specs]
        u_tr = [None if sp["unit_emb"] is None else (sp["unit_emb"], sp["unit_tr"], sp["unitpred_"]) for sp in specs]
        u_va, u_tr = (u_va + [None] * len(raw), u_tr + [None] * len(raw)) if use_unit else (None, None)
        rvs = evaluate_multi(models, valid, val_idx, stats, device, args.chunk, use_is_scored=True, drops=drops,
                             extras=extras, xstats=xstats, units=u_va)
        tv = time.time() - te
        rhs = evaluate_multi(models, train, holdout_idx, stats, device, args.chunk, use_is_scored=False, drops=drops,
                             extras=extras, xstats=xstats, units=u_tr)
        th = time.time() - te - tv
        for sp, (rv, rva), rh in zip(raw, rvs[len(specs):], rhs[len(specs):]):
            mlog(sp, f"[eval step {step}] raw (non-EMA) weights: valid is_scored wp={rv['weighted_pearson']:.4f} "
                     f"(t0 {rv['t0']:.4f}, t1 {rv['t1']:.4f}) | valid all>=99 wp={rva['weighted_pearson']:.4f} | "
                     f"train-holdout wp={rh['weighted_pearson']:.4f}")
        for sp, (rv, rva), rh in zip(specs, rvs[:len(specs)], rhs[:len(specs)]):
            mlog(sp, f"[eval step {step}] valid is_scored wp={rv['weighted_pearson']:.4f} (t0 {rv['t0']:.4f}, "
                     f"t1 {rv['t1']:.4f}, rows {rv['selected_rows']:,}) | valid all>=99 wp={rva['weighted_pearson']:.4f} "
                     f"(t0 {rva['t0']:.4f}, t1 {rva['t1']:.4f}) | train-holdout wp={rh['weighted_pearson']:.4f} "
                     f"(t0 {rh['t0']:.4f}, t1 {rh['t1']:.4f}) | valid eval {tv:.1f}s, holdout eval {th:.1f}s "
                     f"(all {len(specs)} models){grp}")
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
        bi = -1
        while True:
            tw = time.time()
            b = next(loader, None)
            t_wait += time.time() - tw
            if b is None:
                break
            bi += 1
            if unit_k is not None:
                assert (unit_tr[batches[bi]] == unit_k).all(), f"batch {bi}: sequence of another unit"
                a = np.abs(b["targets"][b["targets"] != 0]).astype(np.float64) / unit_value
                assert a.size == 0 or a.min() > 0.5 and np.abs(a - np.round(a)).max() < 1e-3, f"batch {bi}: targets not of unit {unit_value}"
                seen += np.bincount(unit_tr[batches[bi]], minlength=4)
            x = build_inputs(torch.from_numpy(b["prices"]).to(device), torch.from_numpy(b["features"]).to(device), stats)
            xe = build_extra(torch.from_numpy(b["features"]).to(device), xstats) if use_extra else None
            y = torch.from_numpy(b["targets"]).to(device)
            if use_teacher:
                y_t = torch.stack([torch.from_numpy(b["teacher_t0"]), torch.from_numpy(b["teacher_t1"])], 2).to(device).float()
            assert x.shape[2] == N_CONT + len(FLAG_COLS)
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
                if sp["unit_emb"] is not None:
                    sp["unit_cls"] = torch.from_numpy(np.asarray(sp["unit_tr"][batches[bi]], dtype=np.int64)).to(device)
            for s in range(0, SEQ_LEN, args.chunk):
                if step >= total_steps:
                    break
                ts = time.time()
                valid_mask = (steps_in_chunk[s:s + args.chunk] >= WARMUP).float()[None, :, None]
                e = s + args.chunk  # end of this --chunk piece
                for sp in specs:
                    c = sp["chunk_"]
                    if e % c != 0 or sp["step"] >= sp["total"]:
                        continue  # the model's chunk has not ended yet (or its step budget is used up)
                    sl = slice(e - c, e)
                    tm = time.time()
                    vm = valid_mask if c == args.chunk else (steps_in_chunk[sl] >= WARMUP).float()[None, :, None]
                    m = None if m_all[sp["alpha_"]] is None else m_all[sp["alpha_"]][:, sl]
                    xc = x[:, sl]
                    if sp["noise_gen"] is not None:  # training only; flags (the last 12 columns) stay exact
                        xc = torch.cat([xc[:, :, :N_CONT] + sp["noise_"] * torch.randn(
                            xc.shape[0], xc.shape[1], N_CONT, device=device, generator=sp["noise_gen"]),
                            xc[:, :, N_CONT:]], dim=2)
                    if sp["fdrop_gen"] is not None:  # training only; one keep mask per sequence and column
                        q = sp["fdrop_"]
                        keep = (torch.rand(xc.shape[0], 1, N_CONT, device=device, generator=sp["fdrop_gen"]) >= q)
                        xc = torch.cat([xc[:, :, :N_CONT] * keep.float() / (1 - q), xc[:, :, N_CONT:]], dim=2)
                    xc = apply_drop(xc, sp["drop_idx"])  # after noise / fdrop, so the dropped columns are exactly 0
                    if sp["extra_idx"] is not None:  # appended last: noise, fdrop and drop never touch them
                        xc = with_extra(xc, xe[:, sl], sp["extra_idx"])
                    if sp["unit_emb"] is not None:  # after the extra columns
                        xc = torch.cat([xc, unit_input(sp["unit_emb"], sp["unit_cls"], e - c, e, sp["unitpred_"])], 2)
                    if sp["cls_head"] is None:
                        pred, sp["h"] = sp["model"](xc, sp["h"])
                    else:  # the model's own forward, split to reach the GRU output
                        out, sp["h"] = getattr(sp["model"], sp["arch"])(xc, sp["h"])
                        pred = sp["model"].head(out)
                        logits = sp["cls_head"](out)
                    if sp["offsets"]:
                        y_aux, in_range = aux[tuple(sp["offsets"])]
                        main = loss_fn(pred[:, :, :2], y[:, sl], vm, m)
                        al = torch.stack(aux_loss(pred, y_aux[:, sl], in_range[sl], vm, m)).sum()
                        loss = main + sp["aux_weight"] * al
                        sp["run_main"] += main.item()
                        sp["run_aux"] += al.item()
                    elif sp["custom_loss"]:
                        loss, main, teach = kd_loss_fn(pred, y[:, sl], y_t[:, sl] if sp["kd_"] > 0 else None,
                                                       vm, m, sp["kd_"], sp["wpow_"])
                        sp["run_main"] += main.item()
                        if teach is not None:
                            sp["run_teach"] += teach.item()
                    else:
                        loss = loss_fn(pred, y[:, sl], vm, m)
                    if sp["cls_head"] is not None:
                        if not sp["custom_loss"]:
                            sp["run_main"] += loss.item()
                        cl = cls_loss_fn(logits, y[:, sl], vm, m, sp["wpow_"])
                        loss = loss + sp["cls_"] * cl
                        sp["run_cls"] += cl.item()
                    sp["opt"].zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sp["params"] if sp["extra_params"] else sp["model"].parameters(),
                                                   1.0)
                    sp["opt"].step()
                    sp["sched"].step()
                    if sp["ema_model"] is not None:
                        with torch.no_grad():
                            torch._foreach_lerp_(list(sp["ema_model"].parameters()), list(sp["model"].parameters()),
                                                 1.0 - sp["ema_"])
                    sp["h"] = detach_state(sp["h"])
                    sp["run_loss"] += loss.item()  # .item() synchronises, so the timing below is real GPU time
                    sp["step"] += 1
                    sp["run_k"] += 1
                    dt = time.time() - tm
                    sp["t_model"] += dt
                    sp["t_since"] += dt
                step += 1
                run_n += 1
                t_train += time.time() - ts
                if step % 100 == 0 or (args.smoke and step % 10 == 0):
                    el = time.time() - t_last
                    for sp in specs:
                        k = max(sp["run_k"], 1)  # the model's own optimizer steps since the last line (run_n by default)
                        parts = (f" (main {sp['run_main'] / k:.4f}, aux sum {sp['run_aux'] / k:.4f})"
                                 if sp["offsets"] else "")
                        if sp["kd_"] > 0:
                            parts = f" (main {sp['run_main'] / k:.4f}, teacher {sp['run_teach'] / k:.4f})"
                        if sp["cls_head"] is not None:
                            parts = (parts[:-1] if parts else f" (main {sp['run_main'] / k:.4f}") \
                                + f", cls {sp['run_cls'] / k:.4f})"
                        mlog(sp, f"epoch {epoch} step {sp['step']}/{sp['total']} loss {sp['run_loss'] / k:.4f}{parts} "
                                 f"lr {sp['sched'].get_last_lr()[0]:.2e} | {run_n / el:.1f} steps/s wall, "
                                 f"{run_n * args.batch * args.chunk / el:,.0f} seq-steps/s | "
                                 f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
                    log(f"[all] step {step}: {run_n / el:.1f} steps/s wall ({len(specs) * run_n / el:.1f} model-steps/s); "
                        f"per-model compute " + ", ".join(f"{sp['name']} {run_n / max(sp['t_since'], 1e-9):.1f}"
                                                          for sp in specs)
                        + f" steps/s; loader wait so far {t_wait:.1f}s")
                    for sp in specs:
                        sp.update(run_loss=0.0, run_main=0.0, run_aux=0.0, run_teach=0.0, run_cls=0.0, run_k=0, t_since=0.0)
                    run_n, t_last = 0, time.time()
                if step % args.eval_every == 0:
                    do_eval()
                    t_last = time.time()
            if step >= total_steps:
                break
        epoch += 1

    do_eval(final=True)
    if unit_k is not None:
        log(f"trained sequence-passes per unit {UNITS}: {seen.tolist()} (each batch checked: targets are multiples "
            f"of {unit_value})")
    for sp in specs:
        mlog(sp, f"done. best valid wp {sp['best']:.4f}; own train compute {sp['t_model']:.1f}s for {sp['step']} steps "
                 f"({sp['step'] / max(sp['t_model'], 1e-9):.2f} steps/s)")
    log(f"done. {len(specs)} models x {step} steps; pure train compute {t_train:.1f}s "
        f"({step / max(t_train, 1e-9):.2f} steps/s for all models together, "
        f"{len(specs) * step / max(t_train, 1e-9):.2f} model-steps/s); loader wait {t_wait:.1f}s; gpu peak allocated "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
