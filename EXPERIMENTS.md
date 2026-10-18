# Experiment log (Team Varanasi)

One row per run. Add yours before you launch. Never reuse a `--name`.

## Team rules (shared `user_9` login, shared GPUs)

- One GPU each, and check `nvidia-smi` first. Suggested: GPU 0 = Lnv, GPU 1 = Mukesh, GPU 2 = Tejuus, GPU 3 spare.
- Always launch in the background so closing a tab cannot kill the run:
  `CUDA_VISIBLE_DEVICES=1 nohup python train_v1.py --hidden 128 --epochs 5 --n-val 1873 --n-holdout 300 --eval-every 12800 --seed 0 --name train_vN > ~/work/train_vN.nohup.out 2>&1 &`
- Do not type in a terminal tab where someone else's job is printing. Ctrl+C kills it.
- Tell the group before using a leaderboard submission (5 per day for the whole team).
- Commit the code behind every result, by file name (never `git add .`). Keep seeds fixed. The top 8 must reproduce their result for prize verification.
- No weights, caches or zips in git. No passwords or tokens in any file.

## Runs so far

All scores are weighted Pearson on valid `is_scored` rows. v1 and v2 picked their best checkpoint on a 300-sequence valid subset. v3 and v4 picked on all 1,873 sequences, so compare across the two groups with care.

| Run | Setup | Best valid | Final valid | Notes |
|---|---|---|---|---|
| baseline | starter GRU | 0.617052 (full valid) | | **leaderboard 0.59353** (drop 0.0235). Our v2 dropped 0.0275, so the test set is harder for everyone; lead over baseline is 0.0585 on the leaderboard vs 0.0627 locally |
| v1 | GRU hidden 256, 3 epochs | 0.6745 | 0.6740 | too slow on CPU: about 203 us/row here (about 133 min projected) |
| v2 | GRU hidden 128, 3 epochs | 0.6770 | | int8 build: 0.679703 on full valid (official scorer), **leaderboard 0.65200**, about 59 us/row here, platform run time 20m32s |
| v3 | hidden 128, 8 epochs | 0.6783 (step 64,000) | 0.6704 | overfits after about epoch 5; held-out train sequences also fall |
| v4 | hidden 128, 5 epochs | 0.6788 | 0.6778 | within noise of v2; not worth submitting alone |
| v5 | hidden 128, 5 epochs, seed 1 (checkpoint train_v5_h128_e5_seed1_best.pt) | 0.6801 | 0.6791 | about +0.001 over v4 (seed 0): seed noise is about 0.001 |
| v6 | hidden 128, 5 epochs, seed 2 | 0.6793 | 0.6788 | all rows >= 99: 0.5064; seeds 0/1/2 give best 0.6788 / 0.6801 / 0.6793, spread about 0.0013 |
| ens v4+v5 (int8) | two-model int8 ensemble, one ONNX graph | 0.6822 masked (fp32, valid) | | **leaderboard 0.65315** (v2 alone 0.65200, +0.00115 vs +0.0025 on valid); rank 543 -> 486; projected 39.7 min on the platform |
| ens (valid, fp32) | average raw outputs of v2+v4+v5 | 0.6831 masked | | v2+v5 0.6825 (+0.0024), v4+v5 0.6822, v2+v4 0.6818; all-rows gain about +0.004 to +0.006 (ensemble_check_results.txt) |

## Open questions / next runs

- [x] Seed noise: v4/v5/v6 (seeds 0/1/2) best 0.6788 / 0.6801 / 0.6793. Spread about 0.0013, so the recipe is at a ceiling near 0.680 and single-run differences below 0.002 mean nothing.
- [x] Four-model check (ensemble_check2.py): all four masked 0.6835 (+0.0034), best triple v2+v5+v6 0.6833, pairs 0.6818 to 0.6825 (v2+v5 0.6825, v5+v6 0.6825, v4+v5 0.6822). Triples and the quad do not fit the time limit.
- [x] Two-model int8 ensemble v4+v5 in one ONNX graph (export_ens.py, solution_ens.py): int8 loss 0.00008 on the first 100 valid sequences (0.687698 int8 vs 0.687778 fp32), 114.3 us/row vs 60.7 single (1.88x, one graph saved almost nothing), projected platform time 39.7 min. Zip 341 KB: submission_ens_v4v5_int8.zip.
- [x] Submitted the v4+v5 ensemble: leaderboard 0.65315 (+0.00115 over 0.65200). Valid gain was +0.0025, so the test set rewards ensembling about half as much. Both runs scored on the same test set, so the difference is not sampling noise.
- [ ] Next direction: look at the top of the leaderboard to see how far the leaders are, then pick a new idea (train on train+valid, input or loss changes, different architecture). More seeds and bigger hidden sizes are not promising (hidden 256 was no better than 128).
- [x] Leaderboard calibration: baseline scored 0.59353 on the leaderboard (local 0.617052). Shift is shared, not specific to our model.
- [x] Score spread (spread_v2.py, spread_v2_results.txt): SD of the score over random valid subsets is 0.0268 at 5% of sequences, 0.0189 at 10%, 0.0113 at 25%, 0.0062 at 50% (about 0.27/sqrt(k) for k sequences). Per-sequence median 0.689, 10th percentile 0.341, 90th percentile 0.852. Top 10% of sequences hold 27.5% of the weight.
- [x] Mask robustness (mask_check.py, mask_check_results.txt): on is_scored rows v2 = 0.6797, v4 = 0.6788. On all rows with step >= 99 v2 = 0.5097, v4 = 0.5096. On unscored rows 0.4784 for both. The valid mask (about 9.5% of rows, even across steps) picks easier moments (mean |target| 0.69 inside vs 0.60 outside). Not tuning to the mask: v2 and v4 agree within 0.001 on all rows. The leaderboard drop (0.028, baseline 0.0235) is small next to this 0.20 gap, so the hidden test mask is probably similar but slightly harder. Weakest step range is 5000-9999 (all rows 0.501).
- [ ] Final-model options: train on train + valid with a fixed epoch count, checkpoint averaging, two-model ensemble (platform ran our int8 build in 20.5 of 60 minutes).
