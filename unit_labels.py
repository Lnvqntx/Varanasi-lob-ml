"""Per-sequence target unit labels (0.125433 / 0.241330 / 0.468536 / 0.656868), as in unit_probe.py.

unit_probe.label_units on the first LABEL_ROWS = 2,000 target steps of every sequence; a sequence whose smallest
nonzero |target| there is not a unit (48 of 10,607 train sequences) is labelled from its full target sequence.
Every nonzero |target| read must be an integer multiple of the label. The labels are stored once as new files
{CACHE}/unit_labels_{split}.npy (int8 unit index into UNITS) and loaded from there afterwards (never overwritten).
The head of the targets is taken from unit_probe.py's scratch cache when it exists (same bytes), else read
from the targets cache (2,000 steps per sequence, pread).
On valid the label is also checked against the full-sequence label of every sequence and the unit counts of
unit_analysis.py (398 / 347 / 431 / 697).

  python unit_labels.py   # writes the two label files (CPU, about 1 min)
"""
import os

import numpy as np

from unit_probe import CACHE, LABEL_ROWS, SCRATCH, UNITS, label_units, match, read_head

VALID_COUNTS = [398, 347, 431, 697]  # unit_analysis.py


def labels_path(split):
    return f"{CACHE}/unit_labels_{split}.npy"


def unit_index(value):
    """0.6569 -> 3: the unit within 5e-4 of value."""
    k = int(np.argmin([abs(u - value) for u in UNITS]))
    assert abs(UNITS[k] - value) < 5e-4, f"{value} is not one of the units {UNITS}"
    return k


def make_labels(split):
    cache = f"{SCRATCH}/unit_probe_heads_{split}.npz"
    yh = np.load(cache)["targets"] if os.path.exists(cache) else read_head(f"{CACHE}/{split}_targets.npy", LABEL_ROWS)
    assert yh.shape[1] == LABEL_ROWS
    full = np.load(f"{CACHE}/{split}_targets.npy", mmap_mode="r")
    lab, ok, fallback = label_units(yh, full)
    assert (lab >= 0).all() and ok.all(), f"{split}: bad labels {np.flatnonzero((lab < 0) | ~ok)[:20]}"
    if split == "valid":
        full_lab = np.array([match(np.asarray(full[i]))[0] for i in range(len(full))])
        assert np.array_equal(full_lab, lab), "valid: head label differs from the full-sequence label"
        assert [int(np.sum(lab == k)) for k in range(len(UNITS))] == VALID_COUNTS
    return lab.astype(np.int8), fallback


def unit_labels(split):
    """int8 unit index per sequence of split ('train' / 'valid')."""
    return np.load(labels_path(split))


def main():
    for split in ["train", "valid"]:
        path = labels_path(split)
        if os.path.exists(path):
            lab = np.load(path)
            print(f"{path} exists; kept. Recomputing to compare")
            new, fallback = make_labels(split)
            assert np.array_equal(lab, new), f"{path} differs from a fresh labelling"
        else:
            lab, fallback = make_labels(split)
            np.save(path, lab)
        print(f"{split}: {len(lab)} sequences: " + ", ".join(f"{u} {np.sum(lab == k)}" for k, u in enumerate(UNITS))
              + f"; {len(fallback)} labelled from the full sequence; saved {path}")


if __name__ == "__main__":
    main()
