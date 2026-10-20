"""Turn a train_v5m.py checkpoint (head with 2 + 2K outputs) into a standard 2-output checkpoint.

Keeps head rows 0-1 (t0, t1), drops the auxiliary rows, and records the removed offsets in args
("aux" becomes "", "aux_stripped" holds the original). Everything else is copied unchanged, so
export_v2.py (GRU) and export_v4.py (LSTM) read the result like any train_v2m.py checkpoint. CPU only.

  python strip_aux.py IN.pt OUT.pt
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys

import torch


def strip(ck):
    sd = dict(ck["model"])
    n_out = sd["head.weight"].shape[0]
    offsets = [o for o in ck["args"].get("aux", "").split(",") if o.strip()]
    assert n_out == 2 + 2 * len(offsets), f"head has {n_out} outputs, args aux {ck['args'].get('aux')!r}"
    sd["head.weight"] = sd["head.weight"][:2].clone()
    sd["head.bias"] = sd["head.bias"][:2].clone()
    args = dict(ck["args"], aux="", aux_stripped=ck["args"].get("aux", ""))
    return dict(ck, model=sd, args=args), n_out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst):
        sys.exit(f"{dst} exists; not overwriting")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out, n_out = strip(ck)
    torch.save(out, dst)
    print(f"{src}: step {ck['step']}, head {n_out} -> 2 outputs (aux {ck['args'].get('aux', '')!r} removed); "
          f"saved {dst}")


if __name__ == "__main__":
    main()
