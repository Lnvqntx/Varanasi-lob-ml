# Reproducing the submitted pair `submission_ens_v18cv19a_int8.zip`

This file rebuilds our current best submission from the raw parquet files: two GRU-128 models, v18c and v19a, in one int8 ONNX graph. It scored 0.65684 on the public leaderboard, and its local score is 0.6867 on all 1,873 valid sequences.

Every command below was checked against the script's real options (`--help`, or the usage text for scripts without argparse) on Oct 31. The training commands are copied from the argument lines in the real logs. Section 9 lists plainly what was verified and what was not.

## 1. Machine and software

| Item | What we used |
|---|---|
| GPU | 1 x NVIDIA RTX 2080 Ti (11 GB), driver 580.178.04. Training uses about 1.8 GB (capped at 6 GB in the scripts) |
| CPU / RAM | any x86-64; training needs about 30 GB RAM including page cache. The export and the solution run on 1 core |
| Disk | about 34 GB for the starter pack plus 80 GB for the `.npy` cache. Training reads about 70 GB per run, so the cache disk sets the speed (our 7200 rpm disk: about 82 MB/s, about 17 steps/s) |
| Python | 3.11.6 (same as the competition container) |
| Packages (venv `~/work/venv`) | numpy 2.2.6, pyarrow 19.0.1, onnxruntime 1.23.2 (the starter pack's pins), torch 2.14.1+cu126 (CUDA 12.6, cuDNN 9.10.2), onnx 1.23.2 (export only), scipy 1.17.1 (mask GRU AUC) |

**Paths are hard-coded.** The scripts expect:
- the starter pack at `/home/jovyan/work/wnn/wnn_connectome_starterpack` (`datasets/train.parquet`, `datasets/valid.parquet`, `utils.py`);
- the cache at `/home/jovyan/work/cache`;
- checkpoints in `/home/jovyan/work/checkpoints`;
- logs in `/home/jovyan/work`.

On another machine, create the same paths, for example with a symlink.

**Determinism.** Every training script sets all seeds, forces deterministic cuDNN (`torch.use_deterministic_algorithms(True, warn_only=True)`) and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Results reproduce exactly on the same GPU type and software versions. On a different GPU or cuDNN version, expect small floating-point differences and scores within about 0.001.

**Rules while reproducing.** Run one job at a time. Training is disk-bound, and any second job that reads the cache slows both jobs by about 5x.

```
source ~/work/venv/bin/activate
cd ~/work/varanasi-lob-ml
```

## 2. Step 1: build the `.npy` cache (`make_cache.py`, about 30 minutes, CPU)

```
python make_cache.py --dry-run          # optional: prints the exact sizes, writes nothing
python make_cache.py                    # train and valid -> /home/jovyan/work/cache
```

Expected output: `grand total: 79,909,560,992 bytes (74.422 GiB, 79.910 GB)`. Files written:
- `train_prices.npy` float32 (10607, 20000, 44), 37,336,640,128 bytes;
- `train_features.npy` float16 (10607, 20000, 68), 28,851,040,128 bytes;
- `train_targets.npy`, `train_seq_ix.npy`;
- the same four files for `valid` (1,873 sequences), plus `valid_is_scored.npy`.

Files are written as `.tmp` and renamed when complete. There are no random choices in this step. Our log is `~/work/cache_build.log` (Oct 17).

## 3. Step 2: the v4 checkpoint, needed only for its input statistics (`train_v1.py`, about 77 minutes, GPU)

`mask_gru.py` standardises its inputs with the stats stored in `train_v4_h128_e5_fullval_best.pt`. These stats come from `compute_stats` on the first 64 sequences of the seed-0 train permutation (holdout 300). They are bitwise equal to the stats inside v18c's checkpoint (checked Oct 31), so they depend only on the seed and the split, not on training. The exact command, from the args line of `~/work/train_v4_h128_e5_fullval.log`:

```
CUDA_VISIBLE_DEVICES=0 nohup python train_v1.py --hidden 128 --epochs 5 --n-val 1873 --n-holdout 300 \
    --eval-every 12800 --seed 0 --name train_v4_h128_e5_fullval > ~/work/train_v4.nohup.out 2>&1 &
```

Expected: best valid 0.6788 and final 0.6778 (masked, all 1,873). The log ends with `done. best valid wp 0.6788`.

Untested shortcut: adding `--max-steps 100` should give the same stats, because they are computed before any training, but we did not run it.

## 4. Step 3: the mask GRU and the mask probabilities (`mask_gru.py`, about 5 + 12 minutes, GPU 0)

```
python mask_gru.py train   > ~/work/mask_gru.log 2>&1
python mask_gru.py infer   > ~/work/mask_gru_infer.log 2>&1
```

**`train`.** A 1-layer GRU (hidden 64, seed 0) is fitted on the 1,404 valid sequences with index % 4 != 0, using early stopping on the held-out 469 (index % 4 == 0). It writes `checkpoints/mask_gru_h64_best.pt` and `~/work/mask_gru_results.txt`, and refuses to run if either already exists.

Expected: `best epoch 13 (step 24640), held-out AUC 0.9414`.

**`infer`.** It writes p(scored) for every row as float16:
- `cache/train_maskp.npy` (10607, 20000), 424,280,128 bytes;
- `cache/valid_maskp.npy` (1873, 20000), 74,920,128 bytes.

Warm-up rows get the mean p. Our run gave a mean p of 0.104 on train and 0.0947 on valid. It writes `.tmp` first, then renames, and refuses to overwrite.

The training runs below log `mean_p 0.104154 over 10307 train seqs`, which is a direct check of this file. Only the train file is used by the submitted pair.

## 5. Step 4: train v18c and v19a (`train_multi.py`, about 85 and 81 minutes, GPU 0)

Both members came from K=3 shared-batch runs. v18c is the third model of `train_v18`, and v19a is the first model of `train_v19`. The commands below are rebuilt exactly from the first line of `~/work/train_v18.nohup.out` and `~/work/train_v19.nohup.out`; the v19 command is also the one in `~/work/queue_v19.sh`. The other two models in each run (wd 0.1 / 0.3 and wd 0.3 / 0.05) do not affect v18c or v19a: train_multi.py's checks show a model's weights do not depend on which other models share the run. We kept the runs as they were, so the logs can be compared line by line.

```
# v18 (data seed 0, init seed 0): v18c = train_v18c_wd05_nodrop, weight decay 0.05
M="train_v18a_wd10:0:gru::0:alpha=0.5:layers=2:dropout=0:wd=0.1"
M="$M,train_v18b_wd30:0:gru::0:alpha=0.5:layers=2:dropout=0:wd=0.3"
M="$M,train_v18c_wd05_nodrop:0:gru::0:alpha=0.5:layers=2:dropout=0:wd=0.05"
CUDA_VISIBLE_DEVICES=0 nohup python train_multi.py --models "$M" --data-seed 0 --run train_v18 --hidden 128 \
    --epochs 5 --n-val 1873 --n-holdout 300 --eval-every 12800 --mask-p /home/jovyan/work/cache/train_maskp.npy \
    --alpha 0.5 > ~/work/train_v18.nohup.out 2>&1 &

# after v18 has finished: v19 (data seed 1, init seed 1): v19a = train_v19a_wd10_ds1, weight decay 0.1
M="train_v19a_wd10_ds1:1:gru::0:alpha=0.5:layers=2:dropout=0:wd=0.1"
M="$M,train_v19b_wd30_ds1:1:gru::0:alpha=0.5:layers=2:dropout=0:wd=0.3"
M="$M,train_v19c_wd05_nodrop_ds1:1:gru::0:alpha=0.5:layers=2:dropout=0:wd=0.05"
CUDA_VISIBLE_DEVICES=0 nohup python train_multi.py --models "$M" --data-seed 1 --run train_v19 --hidden 128 \
    --epochs 5 --n-val 1873 --n-holdout 300 --eval-every 12800 --mask-p /home/jovyan/work/cache/train_maskp.npy \
    --alpha 0.5 > ~/work/train_v19.nohup.out 2>&1 &
```

**Settings.** Everything else is a default:
- batch 16, chunk 1000, lr 1e-3 with cosine decay to 0, n-stats 64, 2 layers, no dropout;
- the shared `--wd` default 0.01 is overridden per model by `wd=`;
- 645 batches per epoch x 20 chunks = 64,500 optimizer steps;
- the row weight is (|clipped y| + 0.05) x (0.5 + 0.5 x min(p / mean_p, 20)).

The `--models` field order is name : init seed : arch : aux : aux weight : options; see the `train_multi.py` docstring.

**Code version.** v18 and v19 ran with `train_multi.py` as of commit `7e64328` (Oct 20 10:58 UTC). The current version adds options whose defaults reproduce the old behaviour:
- K=1 still reproduces `train_v2m.py` bit for bit (re-checked Oct 26);
- v20a (run with the version of `6e88398`, v18c's spec plus an EMA) repeats v18c's raw-weight evaluations and all 646 loss lines exactly.

For the exact historical file: `git show 7e64328:train_multi.py > ~/work/train_multi_7e64328.py`. Place it in the repo directory, because it imports `train_v5m`.

**Expected results.** Each line below is the `[eval step N] valid is_scored wp=` value, masked, on all 1,873 valid sequences, at steps 12.8k / 25.6k / 38.4k / 51.2k / 64k / final. Then come the final all-rows-≥99 score and the train-holdout score.

| model | evals | final all rows >= 99 | train holdout |
|---|---|---|---|
| v18c | 0.6711 / 0.6796 / 0.6830 / 0.6848 / 0.6857 / 0.6857 | 0.5138 | 0.4524 |
| v19a | 0.6713 / 0.6710 / 0.6821 / 0.6844 / 0.6856 / 0.6856 | 0.5144 | 0.4502 |

The checkpoints store `valid` 0.6857267 (v18c) and 0.6855678 (v19a), and `valid_all` 0.5138374 / 0.5144045. The files used are `checkpoints/train_v18c_wd05_nodrop_final.pt` and `checkpoints/train_v19a_wd10_ds1_final.pt` (step 64,500; best = final for both).

## 6. Step 5: export the int8 pair (`export_ens.py` + `solution_ens.py`, seconds, CPU)

```
CUDA_VISIBLE_DEVICES= python export_ens.py \
    --ckpt-a ~/work/checkpoints/train_v18c_wd05_nodrop_final.pt \
    --ckpt-b ~/work/checkpoints/train_v19a_wd10_ds1_final.pt \
    --out ~/work/submission_ens_v18cv19a_int8 --zip ~/work/submissions/submission_ens_v18cv19a_int8.zip
```

int8 is the default (`--fp32` turns it off). The graph quantizes every MatMul as `export_v2.py --int8 --cell matmul`, and keeps each model's standardization as explicit Sub/Div nodes. `solution_ens.py` is copied into the zip as `solution.py`. The usage line in `export_ens.py` mentions `PYTHONPATH=<dir with onnx>`; that is no longer needed, because onnx is in the venv.

**Expected output.**
- `solution.py` 4,161 bytes, `model.onnx` 586,673 bytes, `stats.npz` 5,466 bytes (zip 338,570 bytes).
- `model.onnx` SHA-256: `94a634a2f63ad2d87f7db4f969a2ba7244c9cd0e9b9a9960cc9a27b6ba524410`.

We re-ran this export from the two checkpoints on Oct 31 into a new directory. All three files were byte-identical to the submitted zip's contents. The zip itself differs in its timestamps, so compare the files, not the zip bytes.

## 7. Step 6: check the zip

```
mkdir ~/work/sub_check && cd ~/work/sub_check && unzip ~/work/submissions/submission_ens_v18cv19a_int8.zip && cd -

# correctness: int8 zip (step by step, CPU) vs the fp32 PyTorch average (GPU 0) on the first 100 valid sequences
SUB=~/work/sub_check python check_ens.py --ckpt-a ~/work/checkpoints/train_v18c_wd05_nodrop_final.pt \
    --ckpt-b ~/work/checkpoints/train_v19a_wd10_ds1_final.pt --n 100

# speed on one pinned core, GPUs hidden (nothing else running)
CUDA_VISIBLE_DEVICES= taskset -c 31 python bench_ens.py --n 100 ~/work/sub_check

# optional: official step-by-step scorer on all 1,873 valid sequences (about 75 minutes at 116 us/row)
taskset -c 31 python run_full_valid.py ~/work/submissions/submission_ens_v18cv19a_int8.zip ~/work/valid_check_v18cv19a
```

**Expected results** (from our Oct 20 logs `~/work/check_ens_v18cv19a.log` and `~/work/bench_ens_v18cv19a.log`):
- **check_ens:** int8 ensemble wp 0.689219 (t0 0.696426, t1 0.682012), fp32 average 0.689201, scored rows 190,292, int8 loss -0.000017 (limit 0.0010).
- **bench_ens:** 114.0 us/row in `predict()`, 116.3 us/row for the whole loop. The projection is 39.6 minutes on the platform, using the factor 1.89 measured with earlier submissions; the v4+v5 pair, projected at 39.7, took 34m16s.

**Scores to check.**

| what | value | where it comes from |
|---|---|---|
| v18c alone, all 1,873 masked | 0.6857 | its last eval line and the checkpoint `valid` |
| v19a alone, all 1,873 masked | 0.6856 | the same |
| pair (fp32 average of raw outputs, clipped), all 1,873 masked | 0.6867 (0.6867397) | `ensemble_check9_results.txt`; repeated by ensemble_check10 |
| pair on the 469 held-out valid sequences (index % 4 == 0) | 0.6938 (0.6937848) | the same |
| public leaderboard | 0.65684 | platform, Oct 20 |

`ensemble_check9.py` needs seven other checkpoints as well (v7, v9, v16a, v17a, v17c, v18a, v19c). To check the pair alone, use `check_ens.py --n 1873`: its fp32 line is the same average on all 1,873 sequences. The int8 half of that check runs step by step on the CPU for about 75 minutes. We have not run `--n 1873` for this pair (see section 9).

## 8. Order and total time

| step | command | time | device |
|---|---|---|---|
| 1 | `make_cache.py` | about 30 min | CPU, disk |
| 2 | `train_v1.py ... --name train_v4_h128_e5_fullval` | about 77 min | GPU |
| 3 | `mask_gru.py train`, then `mask_gru.py infer` | about 5 + 12 min | GPU |
| 4 | `train_multi.py` v18, then v19 | about 85 + 81 min | GPU (disk-bound) |
| 5 | `export_ens.py` | seconds | CPU |
| 6 | `check_ens.py`, `bench_ens.py` (optional `run_full_valid.py`) | about 10 + 10 min (75 min) | CPU (+ GPU 0 for the reference) |

That is about 5.5 hours in all, plus the download of the starter pack (about 50 minutes at 11 MB/s).

## 9. What was and was not verified

**Verified:**
- **Options.** Every command's options match the script's real options. `--help` was run for `make_cache.py`, `train_v1.py`, `train_multi.py`, `export_ens.py`, `check_ens.py` and `bench_ens.py`. `mask_gru.py` (which prints its usage) and `run_full_valid.py` take positional arguments and were checked against their source.
- **Training commands.** The v4, v18 and v19 commands match the args lines of their real logs. The v19 command also matches `queue_v19.sh`.
- **Export.** Re-exporting from the two final checkpoints gives byte-identical `model.onnx`, `solution.py` and `stats.npz`. The repo's `solution_ens.py` equals the zip's `solution.py`.
- **Input stats.** v4's stats (used by the mask GRU) are bitwise equal to v18c's.
- **Code changes since v18/v19.** The current `train_multi.py` with K=1 defaults reproduces `train_v2m.py`'s smoke-test loss lines, evaluation and weights bit for bit (Oct 26).
- **Determinism.** v20a used v18c's data seed, init seed and settings, plus an EMA copy that does not touch training. It reproduced v18c's raw-weight evaluation at all six evals (masked, t0, t1, all rows, train holdout) and all 646 loss lines exactly. These numbers are logged to 4 decimals. The raw weights of v20a were not saved (its checkpoints hold the EMA weights), so "bit for bit" is shown through the logs, not by comparing weight files. The same kind of check on smoke runs (K=1 train_multi vs train_v2m) did compare weight files and found them bitwise equal. ensemble_check10 reproduced ensemble_check9's saved per-sequence statistics with a maximum difference of about 2e-16.

**Not verified (not rerun on Oct 31: no training and no cache reads were allowed):**
- **The pipeline was not run end to end from the parquet files.** The cache, the v4 run, the mask GRU, `train_maskp.npy` and the two training runs were each run once (Oct 17-20) and are documented from their logs and outputs, but were not rebuilt.
- **Mask GRU determinism.** It sets the same seeds and deterministic flags, but a second `train`/`infer` run was never compared bitwise with the first. If `train_maskp.npy` differs even slightly, v18c and v19a will differ slightly too (expect scores within seed noise, about 0.001).
- **v18c retrained with the current `train_multi.py`.** This was not run at full length. The evidence that it would be identical is indirect (the K=1 smoke equality and the v20a match above, which used an intermediate version). Use the `7e64328` version for a faithful rerun.
- **`check_ens.py --n 1873` and `run_full_valid.py` on this zip** were never run. The full-valid pair score 0.6867 is the fp32 PyTorch average from ensemble_check9/10, not the int8 zip through the official scorer. The int8 loss on the first 100 sequences was -0.000017, so the official int8 score should be very close.
- **The `--max-steps 100` shortcut** for the v4 stats was not run.
- **Results on a different GPU or software versions** were not tested.
