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

**Dates:** start Sep 11, team deadline Nov 1 (23:59 UTC), last submission Nov 15, winners Dec 1. Prize pool $13,600, paid to the **top 8 on the private leaderboard** ($5,000 / 2,500 / 1,700 / 1,300 / 1,000 / 800 / 700 / 600). Top 8 must pass verification within 7 days of the preliminary results (inference code, training code with fixed seeds, 2-8 page report, 30-60 minute code-review call, internal-use license).

**Data:** train 10,607 sequences (212M rows), valid 1,873 sequences (37.5M rows) plus a separate validation mask. Test: 1,970 sequences (39.4M rows), mask hidden. The provided starter baseline is a stateful GRU scoring **0.617052** on the full validation set.

**Things the FAQ says that matter (read on Oct 18):**
- The supplied validation set and mask are "one useful validation setup based on a selected subset of moments". The test mask is hidden and may select different moments, so predictions must be reliable at all required moments.
- A strong score on the visible validation subset does not guarantee a strong test score; there is a "private final test".
- Only the provided train and valid data may be used for training; open-source pre-trained models are allowed if they contain no external market data.
- Read on Oct 19 (rules and timeline pages): the final ranking is the **Private Leaderboard**. Near the end the team leader selects some previously submitted and scored solutions, which are re-scored on the Final Dataset. Ties go to the earlier submission. So every candidate for the final choice must already have been submitted and scored before Nov 15.

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
- Never store passwords or tokens in files on the shared machine. GitHub pushes: username and a personal access token are typed at the prompt each time (the in-memory cache expires and the login is shared, so the token is never saved).
- One CPU core (core 31) is used for timing benchmarks via `taskset`, with GPUs hidden (`CUDA_VISIBLE_DEVICES=`), to mimic the competition.

Rules we agreed inside the team (from Oct 18, details in `EXPERIMENTS.md`):
- **One training job at a time for the whole team** (since Oct 19). Training reads about 70 GB from a single hard disk, so two jobs, even on two different GPUs, both fall to about 3 steps/s (section 3, Days 5-7, step 5). Check `nvidia-smi` and tell the group before launching. GPU 0 is the default.
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

### Days 5-7 (Oct 19-21): the mask, weight decay, and where the error sits

By Oct 19 the plain recipe was stuck near 0.680 on valid, so these days went into ideas that change what the model learns, not more seeds. Every run is in `EXPERIMENTS.md`.

**1. Rules we had not read closely (Oct 19)**
- Prizes go to the top 8 on the **private** leaderboard. The final ranking is the Private Leaderboard; the team leader selects some previously submitted and scored solutions for re-scoring on the Final Dataset; ties go to the earlier submission. Every final candidate must therefore be submitted and scored before Nov 15.
- The top 8 deliver code, a report, a review call and a license within 7 days of the preliminary results (section 1).

**2. Is the mask predictable? (`mask_gru.py`)**
- The valid `is_scored` mask picks about 9.47% of rows. A small mask GRU (1 layer, hidden 64, causal features only) predicts it with AUC 0.941 (within a sequence 0.936) on 469 valid sequences it never saw (valid index % 4 == 0). The top decile of predicted probability holds 65.7% of the scored rows, and the mean predicted p is 9.35% against a true 9.19%.
- The train set has no mask, so we predicted p for every train row (`train_maskp.npy`, mean 0.104) and for valid (`valid_maskp.npy`, mean 0.0947). The caches are not in git.

**3. Mask-weighted loss (`train_v2m.py`, runs v7 to v10)**
The loss weight is multiplied by m = alpha + (1 - alpha) * min(p / mean_p, 20). All runs: hidden 128, 5 epochs, `--n-val 1873 --n-holdout 300`.

| run | setup | best / final valid (all 1,873) | 469 held-out | all rows >= 99 (full valid) |
|---|---|---|---|---|
| v4 | plain, seed 0 | 0.6788 / 0.6778 | 0.6845 | 0.5096 |
| v7 | alpha 0.5, seed 0 | 0.6824 / 0.6824 | 0.6892 | 0.5098 |
| v8c | alpha 0.0 (weight proportional to p) | 0.6811 / 0.6806 | | 0.5007 |
| v9 | alpha 0.5, seed 1 | 0.6826 / 0.6820 | 0.6889 | |
| v10 | alpha 0.5, seed 2 | | 0.6863 | |

- alpha 0.5 gives +0.0036 over v4 at the same seed (seed noise is about 0.0013) and +0.0028 on average over three seeds. alpha 0.0 is worse and loses 0.0089 on all rows.
- The mask GRU never saw the 469 held-out sequences, and v7 still gains +0.0047 there, so the gain is not leakage.
- The effect is in the training loss, not in the output. Scaling v7+v9's predictions by the mask probability afterwards (`calib_check.py`) lowered the score for every setting (mildest, (p/mean_p)^0.1: -0.0027), because the metric already ignores unscored rows.

**4. Leaderboard, Oct 19**
`submission_v7_int8.zip` alone scored **0.65367** (platform run 22m21s). The v7+v9 two-model int8 ensemble scored **0.65542**, rank 372 of 5,057. The valid gain of the v7+v9 pair carried over at about 67% (v4+v5 carried 46%).

**5. One training job at a time**
- Two jobs together, even on two different GPUs, each ran at about 3 steps/s instead of about 17. The data (about 64-70 GB) is read from a single 7200 rpm disk (about 82 MB/s, which is about 16 steps/s), and the container's 30 GB memory limit includes the page cache. The GPU alone would do 107-119 steps/s, so training is disk-bound. Heavy validation passes slow a running job the same way. We lost about 10 minutes once by running an ensemble check next to a training run.
- Before Oct 19 the training scripts forced GPU 0 whatever `CUDA_VISIBLE_DEVICES` said (a job started on GPU 1 still ran on GPU 0). Fixed with `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")`.
- `train_multi.py` (Oct 20) trains K models from the same stream of batches in one process, so the disk is read once. K = 3 takes about the same wall time as one run (about 100 minutes). Checked bit for bit against `train_v2m.py` and `train_v5m.py` (K = 1 reproduces their loss lines, identical specs give identical models). Models in one run share the batch order, which matters for ensembles (step 9).

**6. Ideas that did not work (each measured)**

| idea | result | verdict |
|---|---|---|
| scale predictions by mask probability | every setting lowers the score (best -0.0027) | no |
| train on train + valid (`train_v3m.py`, v11 / v12) | 0.6849 / 0.6909 on the 469 against 0.6892 / 0.6889 for v7 / v9; the pair 0.6903 against 0.6915 | no |
| LSTM instead of GRU (`train_v4m.py`, v13) | 0.6827, equal to the GRU; v7+v13 only +0.0008 over v7+v9; residual correlation 0.974 against 0.979 | no |
| GBM correction of the residual on 216 extra rolling features | -0.0016 out of fold, no signal | no |
| auxiliary heads predicting shifted targets at +50, +200, -50, -200 steps (`train_v5m.py`, v14) | 0.6802 against 0.6824 | no |
| alpha 0.75 (v16b) | 0.6807 | worse |
| 3 GRU layers (v16c) | 0.6815, and 22% slower | no |
| dropout alone, weight decay unchanged (v17b) | 0.6821 | no |
| EMA of the weights (v20a) | 0.6857, equal to its raw weights (its early +0.0093 vanishes as the learning rate decays to zero) | no |
| learning rate 2e-3 (v20b) / hidden 192 (v20c) | 0.6841 / 0.6845 | no |
| three-model ensemble | 0.6929 on the 469 but about 48-51 minutes projected on the platform | too risky |

**7. Weight decay was the one knob that worked**
The first weight-decay run (v16a, wd 0.05 with dropout 0.1) was ahead of v7 from mid-run on. Replicates and a sweep followed (AdamW; the old default was 0.01). Dropout turned out to do nothing; weight decay does all of it.

| run | wd | dropout | data seed | final valid | all rows >= 99 |
|---|---|---|---|---|---|
| v7 / v9 / v13 (baselines) | 0.01 | 0 | 0 or 1 | 0.6824 / 0.6820 / 0.6827 | about 0.510 |
| v16a | 0.05 | 0.1 | 0 | 0.6842 | 0.5137 |
| v17a | 0.05 | 0.2 | 0 | 0.6846 | 0.5128 |
| v17b | 0.01 | 0.1 | 0 | 0.6821 | 0.5086 |
| v17c (replicate of v16a, init seed 1) | 0.05 | 0.1 | 0 | 0.6852 | 0.5140 |
| v18a | 0.1 | 0 | 0 | 0.6848 | 0.5139 |
| v18b | 0.3 | 0 | 0 | 0.6817 | 0.5085 |
| **v18c** | **0.05** | 0 | 0 | **0.6857** | 0.5138 |
| v19a | 0.1 | 0 | 1 | 0.6856 | 0.5144 |
| v19b | 0.3 | 0 | 1 | 0.6822 | 0.5083 |
| v19c | 0.05 | 0 | 1 | 0.6849 | 0.5134 |

- Six runs with wd 0.05-0.1 land at 0.6842-0.6857 (about +0.0027 over the baselines) on both data seeds. wd 0.3 is too strong and loses the gain. The best single model (v18c, 0.6857) already equals the v7+v9 pair (0.6850) and takes 21 instead of 34 minutes on the platform.

**8. Where the error sits (analysis, no training)**
- **Large targets:** the 11% of rows at the +-2 clip hold about 32% of the weight and 57% of the squared error. Rows with y > 0 hold 34% of the weight and 54% of the error. Quiet, tight-spread rows score only 0.43 and hold 64.5% of the error. The worst 10% of sequences score 0.20 (the rest 0.73). Step range and t0 against t1 do not matter.
- **Targets are integer multiples of a per-sequence unit** (`target_structure.py`): every sequence has one of four units (0.125433, 0.241330, 0.468536, 0.656868; 398 / 347 / 431 / 697 valid sequences). t0 + t1 behaves like a cost (-1 unit on 47% of rows, -2 on 29%) and t0 - t1 like a direction. Autocorrelation is 0.976 at lag 1, 0.52 at lag 100 and 0.35 at lag 200, so the horizon is long-tailed (a few hundred steps). The anonymised price columns explain only about 4% of either target, so extra targets cannot be derived from them.
- **The coarse-unit group (0.6569) is the weak one** (`unit_analysis.py`): it holds 47.7% of the metric weight and 66% of the weighted squared error and scores about 0.43 against 0.80 / 0.77 / 0.73 for the finer units. Per-unit rescaling gains only +0.0001, so it is not a calibration problem.
- **The unit is easy to recognise** (`unit_probe.py`): from the first 20 steps a classifier gets 99.99% in cross-validation on train and 100% on all 1,873 valid sequences; from the first 99 steps 99.96% / 99.95%. The signal is in the level of a few volume-like and price columns. The generalist models therefore almost certainly know the unit already; an auxiliary unit label was not worth a run.
- **In-group reference scores** on the 697 coarse valid sequences (`unit_group_eval.py`): v18c 0.4334, v19a 0.4332, v20a 0.4334, the pair v18c+v19a 0.4347 (all rows >= 99: about 0.269). Every generalist lands within 0.0003 of the others.

**9. Weight-decay ensembles and the third leaderboard score**
- The wd models are near-duplicates: residual correlations 0.989-0.993, even across data seeds, against 0.979 for v7-v9. So pairing them gains only about +0.0008 over the better member, and the gain of a pair comes mostly from its better single models.
- Best cross-seed pair, `ensemble_check9.py`: **v18c + v19a**, 0.6867 on all 1,873 (+0.0018 over v7+v9) and 0.6938 on the 469 (+0.0022). Chosen as the best of 10 pairs, so slightly optimistic.
- int8 build `submission_ens_v18cv19a_int8.zip` (338,570 bytes): int8 loss -0.000017 on the first 100 valid sequences, 114.0 us/row, projected 39.6 minutes on the platform (the v4+v5 pair was projected at 39.7 and took 34m16s).
- **Leaderboard 0.65684, rank 302 of 5,111** (+0.00142 over v7+v9). 79% of the valid gain carried over, better than before. This is our current best and fallback final submission.
- A bug caught by our own sanity check: `ensemble_check9.py` stopped because it compared the 469-sequence scores with full-valid reference values. The check was right and the scores were fine; the reference lookup was wrong. Lesson: fix the check, never loosen its tolerance.

**10. Leaderboard on Oct 20 (read at 11:26 IST)**

| rank | score |
|---|---|
| 1 | 0.67244 |
| 8 | 0.67017 |
| 25 | 0.66783 |
| 50 | 0.66511 |
| 150 | 0.66049 |
| 302 (us, after step 9) | 0.65684 |

The top 10 are within 0.0023 of each other. Near our score +0.001 is worth about 40 places. To reach the prize line we need about +0.013.

**11. Coarse-unit specialist test (v21, running at the time of writing)**
`train_multi.py --unit-filter 0.6569` trains only on the coarse-unit train sequences (4,432 of 10,307, 43.0%) and evaluates on the 697 coarse valid sequences. Three models (wd 0.05, wd 0.1, wd 0.05 with lr 5e-4), 44,320 steps each, expected to finish about 02:20 IST on Oct 21. The question: does a specialist beat 0.4334 (the generalist in-group score) by about 0.01? If yes, a router (the unit can be read from the first 20-99 steps, which are never scored) could send each sequence to its specialist at no extra inference cost. If no, we stop. The result goes in `EXPERIMENTS.md`.

---

## 4. Results so far

| submission | local (valid) | public leaderboard |
|---|---|---|
| starter `baseline_submission.zip` (calibration) | 0.617052 | 0.59353 |
| `submission_v2_int8_std.zip` (single GRU-128, int8, 177 KB) | 0.679703 (full valid, official scorer) | 0.65200 |
| `submission_ens_v4v5_int8.zip` (two GRU-128, int8, 341 KB) | 0.6822 masked (fp32 average on full valid) | 0.65315 |
| `submission_v7_int8.zip` (single GRU-128, mask-weighted loss) | 0.6824 | 0.65367 |
| `submission_ens_v7v9_int8.zip` (two mask-weighted GRU-128) | 0.6850 (fp32 average, full valid) | 0.65542 |
| `submission_ens_v18cv19a_int8.zip` (two GRU-128 with weight decay 0.05 / 0.1, data seeds 0 and 1, 339 KB) | 0.6867 (fp32 average, full valid) | **0.65684** (current best and fallback final) |

- Our rank: about 532 after the first submission (Oct 17), 543-547 during Oct 18, 486 of 5,016 after the first ensemble, 372 of 5,057 after the mask-weighted pair (Oct 19), **302 of 5,111** after the weight-decay pair (Oct 20).
- Why the leaderboard (0.65) is below local (0.68): the baseline drops by almost the same amount (0.617 to 0.5935), so the test set is simply harder than valid for everyone. It is not caused by checkpoint selection or by tuning to the valid mask (section 3, steps 1, 3 and 4).
- Gains on valid shrink on the test set: the first ensemble gave +0.0025 locally and +0.00115 on the leaderboard (46%); v7+v9 kept about 67%; the weight-decay pair kept 79% (+0.0018 locally, +0.00142 on the leaderboard).
- Only submitted and scored solutions can be picked as the final choice (section 1), so each real candidate is submitted once, after the team is told.

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
- Running two training jobs at once (even on two GPUs) made both about 5 times slower, because the data sits on one hard disk. I first said both would finish by 10 PM, which was wrong. Later I again asked for an ensemble check to run next to a training job and it cost about 10 minutes. Anything that reads the data waits until training is done.
- The training scripts forced GPU 0 whatever `CUDA_VISIBLE_DEVICES` said, which is why one teammate's "GPU 1" run slowed ours down. Check `nvidia-smi` memory on the GPU you meant after launching.
- I first thought three models might fit in 60 minutes (it is about 48-51), then judged it too risky; a timeout scores zero. Keep two models unless a platform-measured time is under about 45 minutes.
- Gains selected as "best of N" are slightly optimistic (the weight-decay pair was the best of 10). The leaderboard check showed 79% carried over, but plan for 50-70%.
- A single run on 469 sequences cannot resolve differences below about 0.002; v16a looked behind v7 on the 469 and ahead on the other 1,404 sequences, and only the replicate (v17c) settled it. Replicate before believing a gain below 0.003.
- A sanity check that fails is information. `ensemble_check9.py` stopped on a reference-value mismatch; the models were fine and the check's lookup was wrong. Fix the check, never raise its tolerance.
- Shell detail: in a waiter script `echo "$(date) ... $?"` resets `$?` before it is printed; store `$?` in a variable first.
- `ps -eo args` truncates long commands and `pgrep -f` also matches the wrapper shell; find the python PID once with `ps -ww -eo pid,comm,args` and wait with `kill -0 PID`.
- v9 was run with a trainer that saved only the best checkpoint, so it has no `_final.pt`; later checks use its best (step 51,200).
- A `git push` was rejected once after a commit made from another tab; `git pull --rebase` and then `git push` fixed it. Never force-push.

---

## 6. Next steps

Done since Oct 18 (kept for the record): the mask is predictable and mask weighting works; train + valid, LSTM, auxiliary targets and the other ideas in section 3 (Days 5-7, step 6) did not help; the rules and prize pages were read (section 1).

1. **Coarse-unit specialist** (v21, running): does a model trained only on the 0.6569-unit sequences beat the generalists' 0.4334 in that group by about 0.01? If yes, build the router (unit recognised from the first 20-99 steps, at no extra inference cost) and test it on the full valid set before spending a submission. If no, drop it.
2. Keep the weight-decay recipe (wd 0.05-0.1, no dropout, mask weighting alpha 0.5) and train more members on **different data seeds** to build the best two-model pair (members on one data seed are near-duplicates, residual correlation 0.99). A third model needs a platform-measured time under about 45 minutes before it is submitted.
3. Any single-model idea is tested with `train_multi.py` (three variants per run, about 100 minutes), compared with v18c (0.6857) and replicated before it is believed.
4. Submit only real improvements (at least +0.0015 over the current pair on all valid and not lower on the 469), one at a time, after telling the team. Used so far: 6 submissions including the baseline calibration.
5. Keep training code seeded and reproducible (required for prize verification). Prepare the 2-8 page technical report and the clean training/inference package well before Nov 15.
6. Check that the team is registered on the competition site (merge deadline Nov 1) and that Mukesh and Tejas are collaborators on this repo.
7. Back up the checkpoints and submission zips of the final candidates off the university machine.

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
| `mask_gru.py` | small GRU that predicts the valid `is_scored` mask; makes `train_maskp.npy` / `valid_maskp.npy` |
| `train_v2m.py` | GRU training with the mask-weighted loss (`--mask-p`, `--alpha`) |
| `train_v3m.py` | same, optionally also trained on valid (`--add-valid`); no gain |
| `train_v4m.py` | adds `--arch lstm`; no gain |
| `train_v5m.py`, `strip_aux.py`, `aux_checks.py` | auxiliary shifted-target heads (`--aux`, `--aux-weight`), converter to a normal 2-output checkpoint, alignment checks; no gain |
| `train_multi.py` | trains K models from one stream of batches (per-model seed, alpha, layers, dropout, weight decay, lr, hidden size, EMA, auxiliary heads, and `--unit-filter` for specialists) |
| `export_v4.py`, `solution_v4.py`, `lstm_export_check.py` | LSTM export and its checks |
| `calib_check.py`, `calib_check_results.txt` | scaling by mask probability (no gain) |
| `ensemble_check3.py` to `ensemble_check9.py` (+ `_results.txt`) | ensemble and pair checks on the 469 held-out and all 1,873 valid sequences |
| `io_bench.py` | disk read speed test (why training is disk-bound) |
| `error_analysis.py`, `probe_features.py`, `residual_probe.py` | where the loss sits; GBM residual probe (no signal) |
| `target_structure.py` | targets are integer multiples of a per-sequence unit; autocorrelation; relation to prices |
| `unit_analysis.py`, `unit_probe.py`, `unit_labels.py`, `unit_group_eval.py` | per-unit score breakdown, unit recognition from the first steps, unit labels, in-group scores of the generalists |
| `EXPERIMENTS.md` | team rules, runs table, open questions, lessons |

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

**Current best recipe (weight decay, Oct 20).** Hidden 128, 2 layers, no dropout, 5 epochs, mask weighting alpha 0.5 (needs `~/work/cache/train_maskp.npy` from `mask_gru.py`), AdamW lr 1e-3 with weight decay 0.05 (or 0.1), global seed and data seed 0 for one member and 1 for the other. The two members of the submitted pair came from `train_multi.py` runs (v18c with weight decay 0.05 and data seed 0, v19a with weight decay 0.1 and data seed 1). The pair is exported with `export_ens.py` and `solution_ens.py` from the two `_final.pt` checkpoints, always with `--int8 --cell matmul`.

```
CUDA_VISIBLE_DEVICES=0 nohup python train_multi.py --hidden 128 --epochs 5 --n-val 1873 --n-holdout 300 --eval-every 12800 --seed 0 --data-seed 0 --alpha 0.5 --mask-p ~/work/cache/train_maskp.npy --models "name1:0:gru::0:wd=0.05,name2:0:gru::0:wd=0.1" > ~/work/train_vN.nohup.out 2>&1 &
```

The exact `--models` spec syntax is in the docstring of `train_multi.py`. Run one training job at a time, and do not read the data (ensemble checks, benchmarks on valid) while it runs.
