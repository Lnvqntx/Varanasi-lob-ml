# Varanasi LOB ML

Team **Varanasi**: Alfa Connectome ML Competition (Wunder Fund), limit order book prediction.

> This repo is private on purpose. The competition rules forbid sharing code, models or detailed strategy before the competition ends. Model weights, caches and submission zips are never committed (see `.gitignore`).

**Team:** Lnvqntx, Mukesh, TejasCS.

---

## 1. The competition in plain words

Wunder Fund (a high-frequency trading firm) gives anonymized real market data from two related instruments, `i0` and `i1`. Each row is one moment in time with 112 features: the top-11 bid and ask price/volume levels (the "order book") for both instruments, recent trade price/volume, and 8 extra anonymous features. Rows come in sequences of 20,000 consecutive steps.

**Goal:** predict two hidden indicators, `t0` and `t1`, of the future price movement of `i0`. Instrument `i1` is input only.

**How code is run:** we submit a `.zip` with `solution.py` defining `PredictionModel.predict(data_point)`. The platform feeds the hidden test set one row at a time, in order, per sequence.
- Steps 0-98 are warm-up (return `None`, but update state).
- Steps 99-19,999: return `[t0, t1]`.
- State must be reset when `seq_ix` changes; only current and earlier rows of the same sequence may be used.

**Metric:** Global Weighted Pearson correlation. Targets and predictions are clipped to [-2, 2], weights are |target|, computed separately for `t0` and `t1` over all scored rows pooled together, then averaged. Only rows with the (hidden at test) `is_scored` mask count. Higher is better.

**Limits:** 1 vCPU, 16 GB RAM, no GPU, no internet, 60 minutes for the whole test set (39.4M rows, so about 90 microseconds per row), 20 MB zip, deterministic output. 5 submissions per day per team.

**Dates:** start Sep 11, team deadline Nov 1 (23:59 UTC), last submission Nov 15, winners Dec 1. Prize pool $13,600 (1st $5,000). Top 8 must pass verification (inference code, training code with fixed seeds, 2-8 page report, code review call, internal-use license).

**Data:** train 10,607 sequences (212M rows), valid 1,873 sequences (37.5M rows) plus a separate validation mask. Test: 1,970 sequences (39.4M rows), mask hidden. The provided starter baseline is a stateful GRU scoring **0.617052** on the full validation set.

**Things the FAQ says that matter (read on Oct 18):**
- The supplied validation set and mask are "one useful validation setup based on a selected subset of moments". The test mask is hidden and may select different moments, so predictions must be reliable at all required moments.
- A strong score on the visible validation subset does not guarantee a strong test score; there is a "private final test".
- Only the provided train and valid data may be used for training; open-source pre-trained models are allowed if they contain no external market data.

---

## 2. Compute setup (our university GPU machine)

Our university provided a shared JupyterLab workstation.

| Resource | Value |
|---|---|
| GPUs | 4 x RTX 2080 Ti, 11 GB each (shared; other users hold 1.6-2.9 GB per card) |
| CPU / RAM | 32 cores / 125 GB (shared) |
| Python | 3.11.6 (same as the competition container) |
| Persistent disk | `~/work` on a 3.6 TB disk, about 935 GB free at start |
| Internet | yes |

Rules we agreed with the machine owner and follow:
- Everything lives in `~/work`. Packages are installed only in our own virtualenv, never in the shared base environment.
- Check `nvidia-smi` before every GPU job. Keep GPU memory small (our jobs use about 1.8 GB).
- Long jobs run with `nohup` from the terminal and log to files (the browser tab can be closed).
- Never store passwords or tokens in files on the shared machine. GitHub pushes use a token cached in memory only (`git config credential.helper 'cache --timeout=1800'`).
- One CPU core (core 31) is used for timing benchmarks via `taskset`, with GPUs hidden (`CUDA_VISIBLE_DEVICES=`), to mimic the competition.

Rules we agreed inside the team (from Oct 18, details in `EXPERIMENTS.md`):
- One GPU each: GPU 0 = Lnvqntx, GPU 1 = Mukesh, GPU 2 = Tejas, GPU 3 spare. Check `nvidia-smi` first.
- Every training run is launched with `nohup`, a unique `--name`, and a log file, and is added to the runs table in `EXPERIMENTS.md`.
- Nobody types in a terminal tab where someone else's job is printing.
- The group is told before a leaderboard submission is used (5 per day for the whole team).
- Code behind every result is committed by file name (never `git add .`), with fixed seeds. No weights, caches, zips, passwords or tokens in git.

---

## 3. Everything we did, in order

### Day 1 (Oct 15, night): understanding
Read all competition pages (task, data, submission guide, rules, timeline, prizes, FAQ). Worked out the key constraint early: the whole test set must run in 60 minutes on one core. At first I estimated about 10 microseconds per row; that was wrong, the correct budget is about 90 microseconds per row (3,600 s / 39.4M rows). Planned: starter pack, baseline, own validation, better model, shrink to fit limits.

### Day 2 (Oct 16): getting a GPU and the data
- Asked the university for a GPU; received a JupyterLab login. The login details were not in the first message and had to be requested again. Do not store them anywhere in the repo.
- Checked the machine with `nvidia-smi`, `df`, `free`, `nproc`, Python version, and an internet test.
- First disk reading looked like only 66 GB free (that was the container root). The persistent `~/work` disk actually has about 935 GB free, a useful correction.
- Asked the machine owner about persistence, background jobs, which GPUs, scratch space (answers: files persist, terminal jobs keep running, use any GPU depending on VRAM needed, no scratch folder).
- Downloaded the starter pack (31.4 GiB, about 33.7 GB) by streaming `curl | tar -xz` in the background with `nohup` (about 11 MB/s, about 50 minutes). It contains `train.parquet` (28 GB), `valid.parquet` (5.3 GB), `valid_mask.parquet`, `utils.py` (scorer and `DataPoint`), and the baseline (`solution.py`, `baseline.onnx`).
- The harmless `tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr...'` messages come from the archive being built on a Mac.

### Day 3 (Oct 17): environment, baseline, model, submission

**Environment**
- Python venv at `~/work/venv`. Installed the starter pack's pinned `numpy==2.2.6`, `onnxruntime==1.23.2`, `pyarrow==19.0.1` (these match the scorer), plus PyTorch 2.14.1 (cu126 build, sees all 4 GPUs), pandas, scipy, tqdm.
- Reproduced the baseline: **0.617052** (t0 0.5879, t1 0.6462), taking about 63 minutes on one thread. This proved our environment, scorer and validation agree with the platform.

**Repo and tooling**
- Created a git repo (first made public by mistake; changed to **private**, which the rules require). Added a `.gitignore` for the venv, data, models, zips, logs, caches.
- Installed Claude Code in `~/work/tools`. It needed Node 22 (machine had 20), installed locally with conda into `~/work/node22`, no admin rights needed. We run it in manual-approval mode and read each command before approving.

**Data exploration (`explore.py`, 20 sequences, read-only)**
- No NaN/inf. Features are already roughly standardized (about +-2.35).
- A fill value of **5.199** (negative in `a*` columns, positive in `dv` columns) marks missing data; we add a flag per such column.
- Price columns barely move within a sequence, so the signal is in their **changes**.
- `t1` is much heavier-tailed than `t0` (excess kurtosis 16.9 vs 4.0). Only 2-2.6% of rows have |t| > 2, and since the metric weights by |target|, those rows dominate.
- `t0` and `t1` are strongly negatively correlated (Pearson -0.64, about -0.81 within a sequence).
- Both are very persistent: autocorrelation 0.96 at lag 1, 0.46 at lag 100, about 0.03 at lag 1000. They stay unchanged about 92% of steps and then jump. So the model needs a memory of a few hundred steps.

**Fast data cache (`make_cache.py`)**
Reading 33 GB of parquet repeatedly is slow, so we converted once into memory-mappable `.npy` files in `~/work/cache/`: 44 price columns as float32 (float16 would round away their tiny differences), the other 68 features as float16, targets float32, plus the valid `is_scored` mask. Total 79.9 GB. Verified exactly against the parquet. Written as `.tmp` files and renamed on completion, about 30 minutes.

**Model v1 (`train_v1.py`)**
- Inputs: 44 prices, their first differences, 68 other features, and 12 flags for the fill-value columns = **168 inputs**, with per-column standardization (without it the tiny price diffs are invisible).
- Network: 2-layer GRU (hidden 256), linear head outputting `[t0, t1]` jointly. Trained on whole 20,000-step sequences in chunks of 1,000 steps with the hidden state carried between chunks, so it learns long memory.
- Loss: weighted MSE on clipped targets/predictions, weights |clipped target| + 0.05, steps >= 99 only. This matches the metric.
- AdamW lr 1e-3 with cosine decay, batch 16 sequences, gradient clipping 1.0, fixed seeds. 3 epochs (38,700 steps), about 44 minutes on one GPU (1.78 GB).
- Validation: official scorer from `utils.py` on a fixed 300-sequence valid subset, plus a 300-sequence train holdout.
- Result: best valid (`is_scored` rows) **0.6745**. Note the same model scores much lower on all steps >= 99 (0.516) because `is_scored` rows are easier; the two numbers are not comparable, so we track both.

**Export and the speed problem (`export_v1*.py`, `check_v1.py`)**
- Exported to ONNX (opset 17, explicit hidden state) and timed on one pinned core. v1 (hidden 256) ran at about 203 us/row, roughly **133 minutes** projected, far over the 60-minute limit. int8 helped (138 us/row) but was still too slow.
- Cause: float32 weights (2.9 MB) do not fit the 1 MB per-core L2 cache.
- Calibration: the starter baseline itself runs at about 91-93 us/row on this machine (about 60 minutes), so the platform's machine is presumably at least as fast. We aimed to be at or below the baseline's speed.

**Model v2: hidden size 128 (`export_v2.py`, `bench_vs_baseline.py`)**
- Retrained with hidden 128. It scored **0.6770** (better than v1), and is much faster.
- Benchmarks on the pinned core (baseline in the same run 93.4 us/row):

| build | us/row | vs baseline | projected for 39.4M rows |
|---|---|---|---|
| starter baseline | 93.4 | 1.00x | 61 min |
| v2 fp32 | 84.9 | 0.91x | 56 min |
| v2 fp32, explicit cell | 102.3 | 1.10x | 67 min |
| **v2 int8** | **59.3** | **0.64x** | **39 min** |

- A first int8 attempt lost 0.0030 of score because folding the input standardization into the weights squashed the tiny price-difference features. Keeping standardization as explicit graph nodes and quantizing only the GRU fixed it: score drop **0.000055** on 300 sequences. Outputs are bitwise identical across runs and match PyTorch to about 3.5e-6 (fp32).

**Full local validation (`run_full_valid.py`)**
The int8 zip scored **0.679703** (t0 0.675291, t1 0.684116) on all 1,873 validation sequences with the official step-by-step scorer, vs the baseline's 0.617052. It took 43.8 minutes on one core (about 70 us/row including the scorer's own loop), about 46 minutes projected for the test set.

**First leaderboard submission (evening of Oct 17, 20:40)**
`submission_v2_int8_std.zip` (177 KB) passed the format check and scored on the platform in 20m32s. **Public leaderboard score: 0.65200.**

### Day 4 (Oct 18): measuring the gap, longer runs, seeds, ensemble, second score

The day was spent on one question (why 0.680 locally but 0.652 on the leaderboard?) and on the cheapest real improvement (averaging models). Each step below was run with Claude Code in manual-approval mode on GPU 0, or by Mukesh on GPU 1, with results logged to `EXPERIMENTS.md`.

**1. Baseline calibration submission (13:53)**
We submitted the official starter `baseline_submission.zip` unchanged, to see how far the baseline's leaderboard score sits from its local 0.617052.
- Baseline leaderboard score: **0.59353**, a drop of 0.0235 from its local score.
- Our v2: 0.679703 local to 0.65200 leaderboard, a drop of 0.0275.
- So the test set is harder for everyone, not only for us. Our lead over the baseline is 0.0585 on the leaderboard against 0.0627 locally. This corrects what I wrote on Oct 17 (that checkpoint selection optimism explained the gap).

**2. Longer training runs v3 and v4 (run by teammates on the shared login)**
Both used hidden 128 and picked checkpoints on all 1,873 valid sequences.

| run | epochs | best valid | final valid | note |
|---|---|---|---|---|
| v3 | 8 | 0.6783 (step 64,000) | 0.6704 | overfits after about epoch 5; held-out train sequences also fall |
| v4 | 5 | 0.6788 | 0.6778 | within noise of v2 |

**3. How much does the score vary across sequences? (`spread_v2.py`, `spread_v2_results.txt`)**
Ran the v2 checkpoint once over all 1,873 valid sequences on GPU 0 and saved per-sequence weighted sufficient statistics for `t0` and `t1`. The merged score is 0.679728, matching the official 0.679703 within 0.00003. Then drew random subsets of whole sequences:

| share of valid | sequences | SD of score | 5th-95th percentile |
|---|---|---|---|
| 5% | 94 | 0.0268 | 0.636-0.722 |
| 10% | 187 | 0.0189 | 0.648-0.709 |
| 25% | 468 | 0.0113 | 0.662-0.698 |
| 50% | 936 | 0.0062 | 0.669-0.690 |

- SD is about 0.27 / sqrt(k) for k sequences. Per-sequence score: median 0.689, 10th percentile 0.341, 90th percentile 0.852. The top 10% of sequences hold 27.5% of the total weight.
- The FAQ says the test set is 1,970 sequences. A 0.028 drop would be roughly 3 SD if the public part used most of them, so chance alone is unlikely. The baseline calibration then showed the shift is shared.

**4. Is our model tuned to the valid mask? (`mask_check.py`, `mask_check_results.txt`)**
Scored v2 and v4 (fp32, GPU 0) on three row sets: masked (`is_scored`, step >= 99), all rows with step >= 99, and unmasked rows.

| model | masked | all rows >= 99 | not masked |
|---|---|---|---|
| v2 | 0.6797 | 0.5097 | 0.4784 |
| v4 | 0.6788 | 0.5096 | 0.4784 |

- The valid mask selects about 9.5% of rows (3,528,273 of 37,274,573 predicted rows), spread evenly across steps 99 to 19,999. Masked rows have larger targets (mean |target| about 0.69 against 0.60 outside), and the metric weights by |target|.
- Not tuning to the mask: both models train on every step >= 99 and never see the mask, and v2 and v4 agree within 0.001 on all rows. Resampling whole sequences 1,000 times gave a masked-minus-all-rows gap of -0.2011 with a 95% range of [-0.2125, -0.1898].
- Weakest step range is 5,000-9,999 (all rows about 0.501); the others sit 0.01-0.02 higher.
- The 0.028 leaderboard drop is small next to this 0.20 gap, so the hidden test mask probably picks similar moments, perhaps slightly harder ones.

**5. Seed noise (Mukesh, GPU 1)**
Repeated the v4 setup (hidden 128, 5 epochs, `--n-val 1873 --n-holdout 300`) with seeds 1 and 2.

| run | seed | best valid | final valid | all rows >= 99 |
|---|---|---|---|---|
| v4 | 0 | 0.6788 | 0.6778 | 0.5096 |
| v5 | 1 | 0.6801 | 0.6791 | 0.5076 at the final step |
| v6 | 2 | 0.6793 | 0.6788 | 0.5064 at the final step |

- Spread across seeds is about 0.0013. v2, v3 (best), v4, v5 and v6 are therefore the same model for practical purposes; the recipe is at a ceiling near 0.680 and differences below about 0.002 mean nothing.
- v5 as a single int8 build scored **0.680082** on full valid (fp32 0.6801), 59.1 us/row in `predict()`, 71.8 us/row overall including parquet reads and the scorer loop (44.8 minutes for 37.46M rows, about 47.1 minutes projected for 39.4M rows on the pinned core). Not submitted alone, because it is only about 0.0004 better than v2.
- Small export slip: `export_v2.py --int8` needs `--cell matmul`; the first attempt stopped on that assertion and was rerun with the flag.

**6. Does averaging models help? (`ensemble_check.py`, `ensemble_check2.py`)**
For every valid batch, ran each model once and averaged the raw outputs of every subset of models (fp32, GPU 0), then clipped. Sanity: v2 0.679728, v5 0.680102, v4 and v6 match their training logs.

| models averaged | masked mean | all rows | gain over best single (masked) |
|---|---|---|---|
| v2+v4+v5+v6 | 0.6835 | 0.5164 | +0.0034 |
| v2+v5+v6 | 0.6833 | 0.5154 | +0.0032 |
| v4+v5+v6 | 0.6831 | 0.5158 | +0.0030 |
| v2+v4+v5 | 0.6831 | 0.5159 | +0.0029 |
| v2+v4+v6 | 0.6828 | 0.5151 | +0.0031 |
| v2+v5 | 0.6825 | 0.5144 | +0.0024 |
| v5+v6 | 0.6825 | 0.5137 | +0.0024 |
| v4+v5 | 0.6822 | 0.5146 | +0.0021 |
| v2+v6 | 0.6821 | 0.5129 | +0.0024 |
| v4+v6 | 0.6819 | 0.5135 | +0.0026 |
| v2+v4 | 0.6818 | 0.5140 | +0.0021 |
| v5 / v2 / v6 / v4 alone | 0.6801 / 0.6797 / 0.6793 / 0.6788 | 0.5096 / 0.5097 / 0.5066 / 0.5096 | |

- Every pair gains at least +0.002 on the masked score and about twice that on all rows, so the gain is not a quirk of the mask.
- Three or four models would not fit the time limit (single model about 21 minutes on the platform, so three about 62 minutes), so we use a pair.
- Pair choice: the highest masked mean was v2+v5 (0.682515). Pairs within 0.0005 of it were compared on all-rows mean, where v4+v5 (0.514578) edged v2+v5 (0.514402). The differences are tiny and within noise; v4+v5 was built.

**7. Two-model int8 ensemble (`export_ens.py`, `solution_ens.py`, `check_ens.py`, `bench_ens.py`)**
- One ONNX graph: shared row preprocessing, each model keeps its own explicit standardization nodes, two GRU stacks and heads, quantized exactly as in `export_v2.py --int8 --cell matmul`. The graph averages the two raw outputs and then clips to [-2, 2]. Zip: `submission_ens_v4v5_int8.zip`, 341,575 bytes.
- Correctness on the first 100 valid sequences (run from the unpacked zip with the official accumulator): int8 ensemble 0.687698, fp32 PyTorch average 0.687778, **int8 loss 0.000080**.
- Timing on pinned core 31 (machine 9.6% busy, nothing heavy running): ensemble **114.3 us/row** against 60.7 for the single v5 int8 build (1.88x), so putting both models in one graph saved almost nothing.
- Projection: 114.3 us x 39.4M rows / 1.89 (the platform-to-core-31 speed factor from the v2 submission: 59 us/row here gave 20m32s there) = **39.7 minutes** on the platform, under the 60-minute limit. The same formula gives 21.1 minutes for single v5, close to the 20m32s we measured for v2, so the factor looks right.

**8. Second leaderboard score (submitted 22:33)**
`submission_ens_v4v5_int8.zip` ("v4+v5 int8 ensemble") scored **0.65315** (+0.00115 over 0.65200). On valid the ensemble gained about +0.0025 over v2; on the test set the same step gives less than half of that. Both runs were scored on the same hidden test set, so the difference is not sampling noise. Rank moved from 543 to **486** of 5,016 teams.

**9. Where we stand against the leaders (leaderboard read at 23:30, Oct 18)**

| rank | score |
|---|---|
| 1 | 0.67243 |
| 2 | 0.67145 |
| 5 | 0.67067 |
| 8 | 0.66992 |
| 10 | 0.66962 |
| 25 | 0.66783 |
| 142 | 0.66063 |
| 225 | about 0.6584 |
| 300 | 0.65655 |
| 409 | 0.65449 |
| 486 (us) | 0.65315 |

- The top 10 are within 0.003 of each other. We are about 0.0168 behind rank 8 and 0.0193 behind rank 1. Around our position, +0.001 is worth roughly 40-50 places.
- Several teams have 100-183 submissions; we have 3. The final ranking probably uses different test data from the public board, so heavy tuning against the public board may not hold. We have not tuned to it.

---

## 4. Results so far

| submission | local (valid) | public leaderboard |
|---|---|---|
| starter `baseline_submission.zip` (calibration) | 0.617052 | 0.59353 |
| `submission_v2_int8_std.zip` (single GRU-128, int8, 177 KB) | 0.679703 (full valid, official scorer) | 0.65200 |
| `submission_ens_v4v5_int8.zip` (two GRU-128, int8, 341 KB) | 0.6822 masked (fp32 average on full valid) | **0.65315** |

- Our rank: about 532 after the first submission (Oct 17), 543-547 during Oct 18, **486 of 5,016** after the ensemble.
- Why the leaderboard (0.65) is below local (0.68): the baseline drops by almost the same amount (0.617 to 0.5935), so the test set is simply harder than valid for everyone. It is not caused by checkpoint selection or by tuning to the valid mask (section 3, steps 1, 3 and 4).
- Gains on valid shrink on the test set: the ensemble gave +0.0025 locally and +0.00115 on the leaderboard.

---

## 5. What went wrong or was misjudged (so we do not repeat it)

- Initial time budget estimate (10 us/row) was wrong; the real one is about 90 us/row.
- Disk space looked tight because we read the wrong filesystem.
- Claude Code required Node 22, not 20.
- Our first model (hidden 256) was too slow for the CPU limit; we caught this by timing before spending a submission.
- First int8 attempt lost score; fixed by keeping standardization explicit.
- Repo was initially public; fixed to private.
- On Oct 17 I explained the 0.652 vs 0.680 gap as checkpoint-selection optimism. The baseline calibration showed it is mostly the test set being harder for everyone. Measure with a calibration submission before explaining a gap.
- We treated single-run differences of 0.001 as meaningful before measuring seed noise (about 0.0013). Always check seed noise before reading anything into small differences.
- The v3 run (8 epochs) overfit: best at step 64,000 (0.6783), final 0.6704. We select the best checkpoint, not the last.
- Instructions I gave for the seed runs guessed the checkpoint file names (`..._fullval_best.pt`); the actual file for v5 was `train_v5_h128_e5_seed1_best.pt`, and Claude Code had to find it. Check `ls ~/work/checkpoints` before naming files.
- `export_v2.py --int8` also needs `--cell matmul`; the reproduce outline had left it out.
- Ensembling two models in one ONNX graph did not cut the per-row cost (114 vs 2 x 60 us). Do not assume a shared graph is cheaper; time it.

---

## 6. Next steps

1. Is the valid mask predictable from the features? Train a small classifier on valid to predict `is_scored` from the order-book rows. If it works, weight or focus training on the rows the test mask probably picks. Cheap, and the biggest idea we have with a chance of a large gain.
2. Train on train + valid together with a fixed 5 epochs (about 19% more data). Cannot be checked locally, only on the leaderboard.
3. Try a loss closer to the metric (a differentiable Pearson loss).
4. Try a different architecture (for example a convolutional front end or LSTM) as a stronger or more diverse ensemble member, within the 60-minute CPU budget (an ensemble of two GRU-128 models already takes about 40 minutes).
5. Keep submissions few and meaningful (3 used so far including the baseline); tell the team before each one.
6. Check the competition's timeline and prizes pages for how the final ranking works and how deep the prize list goes.
7. Keep training code seeded and reproducible (required for prize verification) and prepare the 2-8 page technical report.
8. Confirm the team is registered on the competition site (deadline Nov 1) and add Mukesh and Tejas as repo collaborators.

---

## 7. Repo contents

| File | Purpose |
|---|---|
| `explore.py` | read-only data exploration |
| `make_cache.py` | parquet to memory-mappable float32/float16 cache |
| `train_v1.py` | GRU training (flags `--hidden`, `--epochs`, `--n-val`, `--n-holdout`, `--eval-every`, `--seed`, `--name`, `--smoke`) |
| `export_v1.py`, `export_v1_fast.py`, `export_v2.py`, `export_fast.py` | ONNX export (fp32 and int8) and `solution.py` packaging |
| `solution_v2.py`, `solution_fast.py` | submission `solution.py` templates |
| `check_v1.py` | timing, determinism, ONNX vs PyTorch and scoring check |
| `bench_vs_baseline.py` | benchmark against the starter baseline on one pinned core |
| `run_full_valid.py` | full 1,873-sequence validation with the official scorer |
| `spread_v2.py`, `spread_v2_results.txt` | score spread over random valid subsets (v2) |
| `mask_check.py`, `mask_check_results.txt` | scored vs unscored valid rows, by step range (v2, v4) |
| `ensemble_check.py`, `ensemble_check_results.txt` | model averaging check (v2, v4, v5) |
| `ensemble_check2.py`, `ensemble_check2_results.txt` | model averaging check with v6 (all subsets of four) |
| `export_ens.py`, `solution_ens.py` | two-model int8 ensemble in one ONNX graph and its `solution.py` |
| `check_ens.py`, `bench_ens.py` | correctness check against the fp32 average and timing against the single model |
| `EXPERIMENTS.md` | team rules, runs table, open questions |

## 8. How to reproduce (outline)

```
source ~/work/venv/bin/activate
python make_cache.py                       # builds ~/work/cache (about 80 GB)
CUDA_VISIBLE_DEVICES=0 python train_v1.py --hidden 128 --name train_v2
CUDA_VISIBLE_DEVICES=1 nohup python train_v1.py --hidden 128 --epochs 5 --n-val 1873 --n-holdout 300 --eval-every 12800 --seed 1 --name train_v5_h128_e5_seed1_fullval > ~/work/train_v5.nohup.out 2>&1 &
python export_v2.py --ckpt ~/work/checkpoints/train_v2_best.pt --out submission_v2_int8_std --int8 --cell matmul --zip <path>.zip
python run_full_valid.py <path>.zip /tmp/valid_check      # full validation with the official scorer
```

The v4, v5 and v6 recipe is hidden 128, 5 epochs, `--n-val 1873 --n-holdout 300 --eval-every 12800`, seeds 0, 1 and 2. The ensemble zip is built with `export_ens.py` from the v4 and v5 checkpoints. Check each script's `--help` for exact options.
