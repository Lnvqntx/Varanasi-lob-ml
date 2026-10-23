"""Checks for train_v5m.py's auxiliary targets and for strip_aux.py (CPU only).

  python aux_checks.py align
      (b) alignment: for random (sequence, step, offset) triples, including steps near the sequence start and
      end, aux_targets() on a whole batch read with read_batch() and sliced per 1000-step chunk as in training
      equals the train target array at t + o, and the auxiliary weight is zero exactly where t + o is out of
      range (also: a prediction change at an out-of-range row leaves the loss unchanged).
  python aux_checks.py strip V5M.pt STRIPPED.pt
      first-two-output predictions of the v5m model vs the stripped checkpoint in train_v1.GRUModel on the
      first N_SEQ valid sequences: max abs diff and both scores.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "4"

import sys

import numpy as np
import torch

import train_v5m as t5

OFFSETS = [50, 200, -50, -200]
CHUNK = 1000
N_SEQ = 100


def align():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train = t5.load_split("train")
    n, T = train["targets"].shape[:2]
    seqs = np.sort(rng.choice(n, 4, replace=False))
    b = t5.read_batch(train, seqs)  # sorted indices, as in training
    y = torch.from_numpy(b["targets"])
    y_aux, in_range = t5.aux_targets(y, OFFSETS)
    steps = [0, 1, 49, 50, 99, 150, 199, 200, 999, 1000, 1001, 19799, 19800, 19949, 19950, 19999] + \
        list(rng.integers(0, T, 8))
    n_ok = n_out = 0
    for bi, j in enumerate(seqs):
        for t in steps:
            s = (t // CHUNK) * CHUNK
            ya_chunk, ir_chunk = y_aux[:, s:s + CHUNK], in_range[s:s + CHUNK]  # the training slice
            for k, o in enumerate(OFFSETS):
                inside = 0 <= t + o < T
                assert ir_chunk[t - s, k].item() == float(inside), (j, t, o)
                if inside:
                    ref = np.asarray(train["targets"][j, t + o])
                    assert np.array_equal(ya_chunk[bi, t - s, k].numpy(), ref), (j, t, o)
                    n_ok += 1
                else:
                    n_out += 1
    print(f"alignment: {n_ok} in-range (sequence, step, offset) triples equal the target array at t + o; "
          f"{n_out} out-of-range triples flagged; sequences {seqs.tolist()}, steps {sorted(set(int(t) for t in steps))}")

    # weights: zero exactly where out of range (and in warm-up); perturbing an out-of-range prediction changes nothing
    valid = (torch.arange(T) >= t5.WARMUP).float()[None, :, None]
    m = torch.rand(len(seqs), T, 1) + 0.5
    for s in [0, 19 * CHUNK]:
        sl = slice(s, s + CHUNK)
        pred = torch.randn(len(seqs), CHUNK, 2 + 2 * len(OFFSETS))
        base = t5.aux_loss(pred, y_aux[:, sl], in_range[sl], valid[:, sl], m[:, sl])
        for k, o in enumerate(OFFSETS):
            w = (y_aux[:, sl, k].clamp(-2, 2).abs() + 0.05) * valid[:, sl] * in_range[None, sl, k, None] * m[:, sl]
            out_rows = in_range[sl, k] == 0
            assert (w[:, out_rows] == 0).all()
            expect_out = [t for t in range(s, s + CHUNK) if not 0 <= t + o < T]
            assert out_rows.nonzero().flatten().add(s).tolist() == expect_out
            if expect_out:
                p2 = pred.clone()
                p2[:, expect_out[-1] - s, 2 + 2 * k:4 + 2 * k] += 5.0  # last out-of-range row (199 for -200: past warm-up)
                after = t5.aux_loss(p2, y_aux[:, sl], in_range[sl], valid[:, sl], m[:, sl])
                assert after[k] == base[k], (s, o)
            print(f"  chunk {s}-{s + CHUNK - 1}, offset {o:+d}: {len(expect_out)} out-of-range rows, weight 0 there; "
                  f"loss {base[k].item():.4f}" + (", unchanged by an out-of-range perturbation" if expect_out else ""))
    print("alignment check passed")


def strip_check(v5_path, stripped_path):
    import train_v1
    from utils import GlobalAccumulator
    ck5 = torch.load(v5_path, map_location="cpu", weights_only=False)
    cks = torch.load(stripped_path, map_location="cpu", weights_only=False)
    k = len(t5.parse_offsets(ck5["args"]["aux"]))
    m5 = t5.ARCHS[ck5["args"]["arch"]](hidden=ck5["args"]["hidden"], n_out=2 + 2 * k)
    m5.load_state_dict(ck5["model"])
    m1 = train_v1.GRUModel(hidden=cks["args"]["hidden"])
    m1.load_state_dict(cks["model"])
    valid = t5.load_split("valid")
    stats = {kk: v.cpu() for kk, v in ck5["stats"].items()}
    need = np.arange(t5.SEQ_LEN) >= t5.WARMUP
    acc5, acc1 = GlobalAccumulator(), GlobalAccumulator()
    dmax = 0.0
    with torch.no_grad():
        for i in range(0, N_SEQ, 25):
            b = t5.read_batch(valid, np.arange(i, i + 25))
            x = t5.build_inputs(torch.from_numpy(b["prices"]), torch.from_numpy(b["features"]), stats)
            outs = []
            for model in (m5, m1):
                h, ps = None, []
                for s in range(0, t5.SEQ_LEN, CHUNK):
                    p, h = model(x[:, s:s + CHUNK], h)
                    ps.append(p[:, :, :2])
                outs.append(torch.cat(ps, 1).numpy())
            dmax = max(dmax, float(np.abs(outs[0] - outs[1]).max()))
            for q in range(len(outs[0])):
                mask = b["is_scored"][q] & need
                acc5.add(b["targets"][q], outs[0][q], mask)
                acc1.add(b["targets"][q], outs[1][q], mask)
    r5, r1 = acc5.result(), acc1.result()
    print(f"first {N_SEQ} valid sequences: v5m first two outputs wp {r5['weighted_pearson']:.6f}, stripped "
          f"(train_v1.GRUModel) wp {r1['weighted_pearson']:.6f}; max abs diff {dmax:.3e}")


if __name__ == "__main__":
    if sys.argv[1] == "align":
        align()
    else:
        strip_check(sys.argv[2], sys.argv[3])
