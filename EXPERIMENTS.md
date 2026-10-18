# Experiment log (Team Varanasi)

One row per run. Add yours before you launch. Never reuse a `--name`.

## Team rules (shared `user_9` login, shared GPUs)

- One GPU each, and check `nvidia-smi` first. Suggested: GPU 0 = Lnv, GPU 1 = GGMU, GPU 2 = Tejuus, GPU 3 spare.
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
| baseline | starter GRU | 0.617052 (full valid) | | leaderboard score not measured yet |
| v1 | GRU hidden 256, 3 epochs | 0.6745 | 0.6740 | too slow on CPU: about 203 us/row here (about 133 min projected) |
| v2 | GRU hidden 128, 3 epochs | 0.6770 | | int8 build: 0.679703 on full valid (official scorer), **leaderboard 0.65200**, about 59 us/row here, platform run time 20m32s |
| v3 | hidden 128, 8 epochs | 0.6783 (step 64,000) | 0.6704 | overfits after about epoch 5; held-out train sequences also fall |
| v4 | hidden 128, 5 epochs | 0.6788 | 0.6778 | within noise of v2; not worth submitting alone |

## Open questions / next runs

- [ ] Seed noise: repeat the v4 setup with seed 1 and 2. If seeds differ by about 0.001, v2, v3 and v4 are indistinguishable.
- [ ] Leaderboard calibration: submit the starter `baseline_submission.zip` once, to see how far its leaderboard score sits from 0.617.
- [ ] Score spread: bootstrap random subsets of valid sequences with the v2 model to see how much a partial test sample can move the score.
- [ ] Final-model options: train on train + valid with a fixed epoch count, checkpoint averaging, two-model ensemble (platform ran our int8 build in 20.5 of 60 minutes).
