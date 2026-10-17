"""Convert train/valid parquet to memory-mappable .npy files, one row group (= one sequence) at a time.

Outputs in OUT_DIR, per split:
  {split}_prices.npy    float32 (n_seq, 20000, 44)  i0_p*, i1_p* columns
  {split}_features.npy  float16 (n_seq, 20000, 68)  all other feature columns
  {split}_targets.npy   float32 (n_seq, 20000, 2)   [t0, t1]
  {split}_seq_ix.npy    int64   (n_seq,)            seq_ix of each sequence, in parquet order
  valid_is_scored.npy   bool    (n_seq, 20000)      valid only

Column order within prices/features follows the parquet schema (see price_and_other_cols).

Files are written as *.tmp and renamed only after the split finishes, so a partial run never
leaves files that look complete.

Usage:
  python make_cache.py --limit 20 --out ~/work/cache/test20   # test run
  python make_cache.py --dry-run                              # print exact sizes, write nothing
  python make_cache.py                                        # full conversion
"""
import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = "/home/jovyan/work/wnn/wnn_connectome_starterpack/datasets"
OUT_DIR = "/home/jovyan/work/cache"
SPLITS = ["train", "valid"]
SEQ_LEN = 20000
TARGET_COLS = ["t0", "t1"]
META_COLS = {"seq_ix", "step_in_seq", "need_prediction", "is_scored"}


def feature_cols(schema_names):
    cols = [c for c in schema_names if c not in META_COLS and c not in TARGET_COLS]
    assert len(cols) == 112, f"expected 112 feature columns, got {len(cols)}"
    return cols


def price_and_other_cols(schema_names):
    cols = feature_cols(schema_names)
    is_price = lambda c: c.startswith(("i0_p", "i1_p")) and c[4:].isdigit()
    prices = [c for c in cols if is_price(c)]
    others = [c for c in cols if not is_price(c)]
    assert len(prices) == 44 and len(others) == 68, (len(prices), len(others))
    return prices, others


def npy_size(shape, dtype):
    """Exact on-disk size of a .npy file: header + data."""
    header = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": shape}
    import io
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, header)
    return buf.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize


def outputs(split, n_seq):
    out = {
        "prices": ((n_seq, SEQ_LEN, 44), np.float32),
        "features": ((n_seq, SEQ_LEN, 68), np.float16),
        "targets": ((n_seq, SEQ_LEN, 2), np.float32),
        "seq_ix": ((n_seq,), np.int64),
    }
    if split == "valid":
        out["is_scored"] = ((n_seq, SEQ_LEN), np.bool_)
    return out


def convert(split, out_dir, limit, dry_run):
    pf = pq.ParquetFile(os.path.join(DATA_DIR, f"{split}.parquet"))
    md = pf.metadata
    n_seq = md.num_row_groups if limit is None else min(limit, md.num_row_groups)
    prices, others = price_and_other_cols(pf.schema_arrow.names)
    specs = outputs(split, n_seq)

    total = 0
    for name, (shape, dtype) in specs.items():
        size = npy_size(shape, dtype)
        total += size
        print(f"  {split}_{name}.npy  {str(np.dtype(dtype)):>8} {str(shape):>22}  {size:>15,} bytes")
    print(f"  {split} total: {total:,} bytes ({total / 2**30:.3f} GiB)")
    if dry_run:
        return total

    os.makedirs(out_dir, exist_ok=True)
    tmp = {name: os.path.join(out_dir, f"{split}_{name}.npy.tmp") for name in specs}
    arrs = {name: np.lib.format.open_memmap(tmp[name], mode="w+", dtype=dtype, shape=shape)
            for name, (shape, dtype) in specs.items()}
    read_cols = ["seq_ix", "step_in_seq"] + prices + others + TARGET_COLS + (["is_scored"] if split == "valid" else [])

    t_start = time.time()
    for i in range(n_seq):
        tbl = pf.read_row_group(i, columns=read_cols)
        assert tbl.num_rows == SEQ_LEN, f"row group {i}: {tbl.num_rows} rows"
        seq = tbl.column("seq_ix").to_numpy()
        assert (seq == seq[0]).all(), f"row group {i} has more than one seq_ix"
        assert (tbl.column("step_in_seq").to_numpy() == np.arange(SEQ_LEN)).all(), f"row group {i}: steps out of order"

        p = np.empty((SEQ_LEN, 44), dtype=np.float32)
        for j, c in enumerate(prices):
            p[:, j] = tbl.column(c).to_numpy()
        arrs["prices"][i] = p
        x = np.empty((SEQ_LEN, 68), dtype=np.float32)
        for j, c in enumerate(others):
            x[:, j] = tbl.column(c).to_numpy()
        x16 = x.astype(np.float16)
        assert np.isfinite(x16).all(), f"row group {i}: non-finite feature after float16 cast"
        arrs["features"][i] = x16
        arrs["targets"][i] = np.stack([tbl.column(c).to_numpy() for c in TARGET_COLS], axis=1)
        arrs["seq_ix"][i] = seq[0]
        if split == "valid":
            arrs["is_scored"][i] = tbl.column("is_scored").to_numpy()
        del tbl, p, x, x16

        if (i + 1) % 200 == 0 or i + 1 == n_seq:
            for a in arrs.values():
                a.flush()
            el = time.time() - t_start
            print(f"  {split}: {i + 1}/{n_seq} seqs, {el:.0f}s elapsed, eta {el / (i + 1) * (n_seq - i - 1):.0f}s",
                  flush=True)

    for name, a in arrs.items():
        a.flush()
        del a
    arrs.clear()
    for name, path in tmp.items():
        os.replace(path, path[:-4])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only convert the first N sequences per split")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--dry-run", action="store_true", help="print exact output sizes and exit")
    args = ap.parse_args()

    grand = 0
    for split in args.splits:
        print(f"[{split}]")
        grand += convert(split, os.path.expanduser(args.out), args.limit, args.dry_run)
    print(f"grand total: {grand:,} bytes ({grand / 2**30:.3f} GiB, {grand / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()
