# Varanasi LOB ML

**Causal limit order book forecasting · GRU ensembles · INT8 ONNX inference**

Research and inference code for **Team Varanasi** in the **Wunder Fund Alfa Connectome ML Competition**. The task is to predict two future price-movement indicators from anonymized order book and trade data, under a single-CPU streaming inference budget.

**Team:** Lnvqntx · Mukesh · TejasCS  
**Last updated:** November 20, 2026

[Results](#4-final-selections-and-results) · [Research history](#3-everything-we-did-in-order) · [Reproduction](REPRODUCE.md) · [Experiments](EXPERIMENTS.md) · [Code guide](#7-repo-contents)

---

## Project at a glance

| Item | Recorded outcome |
|---|---|
| **Final placement** | **412th** |
| **Final private score** | **0.66582** |
| Best public score | **0.65684**, deployed INT8 ensemble: **v18c + v19a** |
| Public rank snapshot | **495 / 5,840**, observed November 16; **not the final placement** |
| Local validation | **0.6867397** weighted Pearson for the FP32 pair on 1,873 sequences |
| Platform inference | **36m07s · 624 MB** for the best public submission |
| Final selections | **Three submissions confirmed and locked** on November 16 |
| Competition status | **Complete — final results recorded November 20, 2026** |

Team Varanasi finished **412th** with a **final private score of 0.66582**, an improvement of **83 places** from public rank 495. The final leaderboard identifies the best submission as `1VH*`, matching the selected v18c + v19a entry (`1VH23TAR`). [Official final leaderboard](https://wundernn.io/connectome/finals).

Local FP32 validation, deployed INT8 public evaluation and private evaluation use different datasets or execution paths. Their scores are reported separately.

### Approach

- **Causal features:** 168 inputs built from the supplied 112 features, including price changes and missing-value flags.
- **Sequence models:** two-layer GRUs with 128 hidden units; the best submitted pair combines models trained with different seeds and weight decay.
- **Streaming deployment:** explicit input standardization, INT8 ONNX inference, persistent recurrent state within each sequence and resets between sequences.
- **Measured iteration:** feature ablations, loss and architecture comparisons, ensemble audits, deterministic checks and CPU timing. Validation was reused throughout development; small gains are not independent evidence of generalization.

### Final selections

| Selected ensemble | Public score | Status |
|---|---:|---|
| **v18c + v19a** | **0.65684** | Confirmed |
| v18c + v24a | 0.65596 | Confirmed |
| v7 + v9 | 0.65542 | Confirmed |

The final score is the **highest private score** among these three. Selection was confirmed at approximately **17:21 IST / 11:51 UTC on November 16** and cannot be changed. Exact artifact names and evaluation caveats appear in [Final selections and results](#4-final-selections-and-results).

### Start here

| Document | Purpose |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | Recorded reconstruction commands, seeds, artifact provenance and verification limits |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed experiment ledger and historical run instructions |
| [Final ensemble audit](FINAL_ENSEMBLE_AUDIT_20261114.md) | Fixed-checkpoint comparisons, uncertainty estimates and exported-triple runtime measurements |
| [Bounded residual experiment](BOUNDED_RESIDUAL_FINAL_20261114.md) | Final 32.1-minute attempt; neither residual pair improved the incumbent |
| [v34 streaming prototype](V34_STREAM_OPT_20261109.md) | Circular-cache optimization; smoke benchmarks only, no full training or platform certification |
| [Technical report](REPORT.md) | Method, final results, deployment measurements and reproducibility limits |

**Artifacts and reproducibility.** Competition data, model weights, generated caches and submission archives are not included in Git. Reproduction requires those external artifacts. A 308 MB project archive containing source code, checkpoints and selected submission directories was downloaded and passed SHA-256 and archive-readability checks on November 16. Raw data, generated caches and installed environments were outside that backup.

<details>
<summary><strong>Contents</strong></summary>

1. [Competition and evaluation](#1-the-competition-in-plain-words)
2. [Compute setup](#2-compute-setup-our-university-gpu-machine)
3. [Dated research history](#3-everything-we-did-in-order)
4. [Final selections and results](#4-final-selections-and-results)
5. [Lessons and corrections](#5-what-went-wrong-or-was-misjudged-so-we-do-not-repeat-it)
6. [Closeout tasks](#6-closeout-tasks)
7. [Repository contents](#7-repo-contents)
8. [Reproduction and verification limits](#8-reproduction-summary-and-verification-limits)

</details>

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

**Originally published dates:** start Sep 11, team deadline Nov 1 (23:59 UTC), last submission Nov 15, winners scheduled for Dec 1. The separately confirmed submission and final-selection deadlines are recorded below. Prize pool $13,600, paid to the **top 8 on the private leaderboard** ($5,000 / 2,500 / 1,700 / 1,300 / 1,000 / 800 / 700 / 600). Top 8 must pass verification within 7 days of the preliminary results (inference code, training code with fixed seeds, 2-8 page report, 30-60 minute code-review call, internal-use license).

**Data:** train 10,607 sequences (212M rows), valid 1,873 sequences (37.5M rows) plus a separate validation mask. Test: 1,970 sequences (39.4M rows), mask hidden. The provided starter baseline is a stateful GRU scoring **0.617052** on the full validation set.

**Things the FAQ says that matter (read on Oct 18):**
- The supplied validation set and mask are "one useful validation setup based on a selected subset of moments". The test mask is hidden and may select different moments, so predictions must be reliable at all required moments.
- A strong score on the visible validation subset does not guarantee a strong test score; there is a "private final test".
- Only the provided train and valid data may be used for training; open-source pre-trained models are allowed if they contain no external market data.
- Read on Oct 19 (rules and timeline pages): the final ranking is the **Private Leaderboard**. Near the end the team leader selects some previously submitted and scored solutions, which are re-scored on the Final Dataset. Ties go to the earlier submission. Every candidate for the final choice therefore had to be submitted and scored by the new-submission cutoff; the confirmed date/time is below.
- Read on Oct 26 (starter-pack docs: data_overview.md, rules.md, faq.md, timeline.md):
  - Scored rows are `need_prediction AND is_scored`. The test mask is hidden and applied by the platform after inference, and the docs give no rule for how rows are chosen (only "a selected subset of moments"). So we predict every required row as if it were scored.
  - Those documents did not specify the number of final selections. The later email and the three confirmed selections are recorded below.
  - The October documents listed end-of-day deadlines at 23:59 UTC / 05:29 IST the next day, including the Nov 1 team merge and Nov 15 new-submission cutoff. The later final-selection deadline is separate; see the November 16 confirmation below.
  - The bid/ask split is by column index only: p/v 0-10 are bid, 11-21 ask. The index is an identifier, not a depth level.

---

### Final selection and deadlines (confirmed November 16)

The competition deadlines and selection policy were:

- New submissions closed **Sunday, November 15, 2026 at 23:59 UTC / Monday, November 16 at 05:29 IST**.
- The organiser's email gives a separate final-selection deadline: **Tuesday, November 17 at 12:00 UTC / 17:30 IST**. Submitted selections cannot be changed.
- Our three selections were successfully submitted and locked on **Monday, November 16 at approximately 17:21 IST / 11:51 UTC**. No further selection action is required.
- The final score is the highest private score among these three. If a participant makes no selection, the organiser automatically selects their three best public-scoring submissions; that fallback is not needed for our confirmed selection.
- The organiser states that submissions which avoided a timeout on the public dataset will also avoid a timeout on the final dataset. This statement concerns already accepted submissions; it does not certify unsubmitted experimental exports.

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
- **One training job at a time for the whole team** (since Oct 19). Training reads about 70 GB from a single hard disk, so two jobs, even on two different GPUs, both fall to about 3 steps/s (section 3, Oct 19, item 8). Check `nvidia-smi` and tell the group before launching. GPU 0 is the default.
- Every training run is launched with `nohup`, a unique `--name`, and a log file, and is added to the runs table in `EXPERIMENTS.md`.
- Nobody types in a terminal tab where someone else's job is printing.
- The group is told before a leaderboard submission is used (5 per day for the whole team).
- Code behind every result is committed by file name (never `git add .`), with fixed seeds. No weights, caches, zips, passwords or tokens in git.

---

## 3. Everything we did, in order

The complete dated record is preserved below, including unsuccessful experiments, corrections and historical leaderboard snapshots.

<details>
<summary><strong>Expand the full research history — October 15 to November 16, 2026</strong></summary>

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
- Both the unchanged baseline and our model scored lower publicly, consistent with a difference between validation and test periods. Our lead over the baseline is 0.0585 publicly against 0.0627 locally. This corrects the Oct 17 explanation based only on checkpoint-selection optimism; it does not exclude selection effects.

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

- Spread across these seeds is about 0.0013. The tested plain-GRU variants clustered near 0.680; small single-run differences were not sufficient evidence of a reliable improvement.
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

### Daily log, Oct 19 to Nov 16

**How to read it**
- **Dates and times are UTC** (the machine clock). They come from:
  - the timestamps in the training logs,
  - the modification times of the logs and result files in `~/work` (`ls -l --time-style=long-iso`),
  - the commit dates (`git log --date=short`).
  - IST is UTC + 5:30, so work done in the UTC evening falls on the next IST day.
- **What the numbers mean:**
  - "scored" is the global weighted Pearson on valid `is_scored` rows with step >= 99, over all 1,873 valid sequences.
  - "all rows" is the same metric on every row with step >= 99.
  - "the 469" are validation sequences with array index % 4 == 0, historically held out of mask-model fitting. They were repeatedly used for diagnostics and model decisions; they are **not untouched validation**. The full validation set was also reused.
  - "Gap" is the difference to the control named in the line. From the evening of Oct 20 the control is **v18c, 0.6857 scored / 0.5138 all rows**. Seed noise is about 0.0013.
- **Where things are:** the full rows are in `EXPERIMENTS.md`, and the summary tables are under "Reference tables" below.

**Record convention:** retain dated results and the decisions made at the time. New entries cover documented activity only; gaps do not establish that work occurred. UTC timestamps take precedence over run labels. Older launch reports may still say "running" or "not launched"; their completed outcomes are linked in the later entries below.

### Oct 19 (Mon)
1. **Rules and prizes read** (rules and timeline pages).
   - Prizes go to the top 8 on the **private** leaderboard.
   - The team leader selects some previously submitted and scored solutions, and these are re-scored on the Final Dataset. Ties go to the earlier submission.
   - The top 8 must deliver code, a 2-8 page report, a review call and a license within 7 days.
   - Decision: every final candidate must be submitted and scored before Nov 15.
2. **Mask diagnostic** (`mask_predict.py`, 04:24).
   - 9.47% of the valid rows are scored, ranging from 0.5% to 60% per sequence (median 7.8%).
   - The share is flat across steps 99 to 19,999.
   - Scored rows come in blocks: 89% of them sit in runs of 2 or more, against 26% if the mask were random. The lag-1 autocorrelation is 0.72.
   - LightGBM on 173 causal features predicts the mask with AUC 0.887 on 200 unseen sequences; the step index alone gives 0.49.
   - Decision: the mask is predictable, so build a mask model.
3. **Mask GRU** (`mask_gru.py`, 04:43).
   - 1 layer, hidden 64, fitted on 1,404 valid sequences.
   - On the 469: AUC 0.941 (0.936 within a sequence), and the top decile holds 65.7% of the scored rows.
   - Wrote `train_maskp.npy` (mean p 0.104) and `valid_maskp.npy`.
4. **v7**, mask-weighted loss with alpha 0.5 (`train_v2m.py`, 04:57-08:39).
   - Final 0.6824 / 0.5098, against v4 (same seed, plain loss) 0.6788 / 0.5096: **+0.0036**.
   - On the 469: 0.6892 against 0.6845 (+0.0047), so the gain is not leakage from the mask fit.
   - It took 3h42m because it shared GPU 0 with v8b (item 5).
5. **v8, v8b and v8c**, alpha 0.0.
   - v8 crashed at its first eval (05:29, `ZeroDivisionError`, because `--eval-every` was 0).
   - v8b was launched with `CUDA_VISIBLE_DEVICES=1`, but the script forced GPU 0, so it shared a GPU with v7. Stopped at step 15,100.
   - Fix: the training scripts now respect `CUDA_VISIBLE_DEVICES` (`os.environ.setdefault`).
   - v8c (07:26-09:45, GPU 1): 0.6811 best / 0.6806 final (+0.0023 over v4's best 0.6788), but all rows fell to 0.5007 (-0.0089).
   - Decision: alpha 0.5.
6. **v7 int8 submitted** (zip 14:42, 172,818 bytes).
   - int8 loss 0.000175 on 100 sequences; full valid 0.682361 with the official scorer.
   - Leaderboard **0.65367**, rank 470 of 5,051, platform run 22m21s. About 60% of the valid gain carried over.
7. **v9 and v10**, replicates with seeds 1 and 2.
   - v9 (14:40-17:12): 0.6826 best / 0.6820 final, +0.0029 over the plain seed-1 run v5 (final).
   - v10 (15:09-18:35, GPU 1): 0.6816 / 0.6815, +0.0027 over v6 (final).
   - The three mask-weighted seeds average 0.6822 against 0.6794 for the plain recipe (+0.0028).
8. **Disk bottleneck.**
   - v9 and v10 together ran at about 3 steps/s each instead of about 17.
   - `io_bench.py` measured a single reader at 82 MB/s, which caps training at about 16 steps/s. Parallel readers are worse (mmap with 2-16 threads: 36-41 MB/s).
   - Decision: **one training job at a time for the whole team**, and nothing else reads the data while one runs.
   - Mistake: I had said both jobs would finish by 10 PM.
9. **ensemble_check3 and ensemble_check4** (17:37 and 18:44).
   - v7+v9 scores 0.6850 scored and 0.6915 on the 469 (+0.0023 over the better member).
   - v7+v10 0.6846, v9+v10 0.6848, triple 0.6856.
   - **v7+v9 int8 submitted** (zip 17:27, 336,664 bytes): int8 loss -0.000022, 115.1 us/row. Leaderboard **0.65542**, rank 372 of 5,057, about 67% of the valid gain kept.
10. **Calibration test** (`calib_check.py`, 18:59).
    - Scaling the predictions by the mask probability lowers every setting; the mildest costs -0.0027.
    - Decision: mask weighting works through training, not the output. Not used.
11. **v11 and v12 launched** (19:07 and 20:39, chained). These train on train + valid; results on Oct 20.

### Oct 20 (Tue)
1. **v11 and v12**, train + valid (results in ensemble_check5, 05:50).
   - On the 469: v11 0.6849 against v7 0.6892 (-0.0043), v12 0.6909 against v9 0.6889 (+0.0020).
   - The pair scores 0.6903 against v7+v9 0.6915.
   - Decision: no gain on scored rows; dropped.
2. **v13**, an LSTM (`train_v4m.py`, 06:09-07:51).
   - 0.6827 / 0.5080 against v7 0.6824 / 0.5098.
   - ensemble_check6 (08:00): v7+v13 0.6923 on the 469 (+0.0008 over v7+v9). The residual correlation is 0.974 against 0.979 for v7-v9.
   - Decision: below the +0.002 bar, so no mixed build. The int8 LSTM runs at 59.2 us/row.
3. **Error analysis** (`error_analysis.py`, 08:13).
   - Rows at the +-2 clip: 11% of rows, 32% of the weight, 57% of the squared error.
   - y > 0: 34% of the weight, 54% of the error.
   - Tight-spread rows: 45% of rows, score 0.429, 64.5% of the error.
   - The worst 10% of sequences score 0.20.
4. **Residual probe** (`residual_probe.py`, 08:21): LightGBM on 216 extra causal features gives -0.0016 out of fold. Closed.
5. **Target structure** (`target_structure.py`, 08:32).
   - The targets are integer multiples of one of four per-sequence units.
   - t0 + t1 behaves like a cost and t0 - t1 like a direction.
   - The horizon is long-tailed: autocorrelation 0.52 at lag 100.
   - The prices explain only about 4% of either target.
   - Decision: try shifted targets as auxiliary labels (v14).
6. **v14**, auxiliary heads for targets shifted by ±50 and ±200 steps (08:53-10:18).
   - 0.6802 / 0.5085 against v7 0.6824 (-0.0022). No gain.
   - Its replicate v15 was cancelled before it started.
7. **`train_multi.py`** (commit 09:09): K models trained from one batch stream, so the disk is read once.
   - With K = 1 it reproduces the loss lines and weights of `train_v2m.py` and `train_v5m.py` bit for bit.
8. **v16**, the first K = 3 run (10:22-11:43).
   - v16a, dropout 0.1 + wd 0.05: 0.6842 / 0.5137 (+0.0018 over v7), but 0.6875 on the 469 against v7's 0.6892.
   - v16b, alpha 0.75: 0.6807 (-0.0017).
   - v16c, 3 layers: 0.6815 (-0.0009) and 22% slower.
   - ensemble_check7 followed at 12:02.
9. **Unit analysis** (`unit_analysis.py`, 12:26) **and unit probe** (`unit_probe.py`, 13:35).
   - The coarse unit (0.6569) holds 48% of the metric weight and 66% of the error, and scores 0.43 against 0.73-0.80 for the finer units.
   - Rescaling each unit separately gains only +0.0001.
   - From the first 20 steps the unit is recognised with 99.99% accuracy (cross-validated on train) and 100% on valid.
   - Decision: the model very likely knows the unit already, so an auxiliary unit label is not worth a run. Revisited on Nov 5 as v30.
10. **v17** (11:46-13:28; slowed by the ensemble_check7 and unit_probe reads).
    - v17a, dropout 0.2: 0.6846 / 0.5128.
    - v17b, dropout alone at wd 0.01: 0.6821 / 0.5086.
    - v17c, a replicate of v16a with init seed 1: 0.6852 / 0.5140.
    - Verdict: **weight decay 0.05 helps**. The three wd 0.05 runs (v16a, v17a, v17c) average 0.6847, +0.0023 over the mean of v7, v9 and v13; dropout did not show a consistent benefit in these comparisons.
11. **ensemble_check8** (13:51).
    - The wd models that share data seed 0 are near-duplicates: residual correlation 0.990-0.992.
    - Best pair v17a+v17c 0.6859, only +0.0010 over v7+v9, so the build rule was not met.
    - Decision: pairs need members with different data seeds, hence v19.
12. **v18** (13:51-15:15).
    - v18a, wd 0.1: 0.6848 / 0.5139.
    - v18b, wd 0.3: 0.6817 / 0.5085.
    - **v18c, wd 0.05 without dropout: 0.6857 / 0.5138**, +0.0033 over v7 with the same init and batches. **It is the control from here on.**
13. **v19**, data seed 1 (15:16-16:36).
    - v19a, wd 0.1: 0.6856 / 0.5144.
    - v19b, wd 0.3: 0.6822 / 0.5083.
    - v19c, wd 0.05: 0.6849 / 0.5134.
14. **ensemble_check9** (17:10).
    - **v18c+v19a** is the best of 10 cross-seed pairs: 0.6867 scored and 0.6938 on the 469, +0.0018 / +0.0022 over v7+v9.
    - Mistake: the first run stopped on its own sanity check. The reference lookup was wrong; the scores were fine. I fixed the lookup and the rerun passed all 60 checks. Lesson: fix the check, never loosen its tolerance.
15. **v18c+v19a int8 submitted** (zip 17:12, 338,570 bytes).
    - int8 loss -0.000017, 114.0 us/row, projected 39.6 min on the platform.
    - Leaderboard **0.65684**, rank 302 of 5,111 (logged 19:32), +0.00142 over v7+v9. About 79% of the valid gain carried over.
    - **It has been our best submission since.**
    - A three-model ensemble (0.6929 on the 469 in ensemble_check9) was judged too risky at about 48-51 minutes projected.
16. **Correction:** three all-rows numbers I had been given did not match the logs: v17a 0.5128 (given 0.5131), v17b 0.5086 (0.5082), v18b 0.5085 (0.5093). The log values were used.
17. **v20** (17:52-19:21).
    - v20a, EMA of the weights: 0.6857 / 0.5139, a gap of 0.0000. It led by +0.0093 at step 12,800, but the lead was gone by the end of the cosine schedule.
    - v20b, lr 2e-3: 0.6841 / 0.5133 (-0.0016).
    - v20c, hidden 192: 0.6845 / 0.5150 (-0.0012) and about 6x slower to train.
18. **In-group reference** (`unit_group_eval.py`, 19:53): on the 697 coarse-unit valid sequences, v18c scores 0.4334 and v18c+v19a 0.4347.
19. **v21**, coarse-unit specialists (`--unit-filter 0.6569`, 19:59-20:37 UTC, which is 01:29-02:08 IST on Oct 21).
    - In-group finals 0.4059 / 0.4218 / 0.4146 (wd 0.05 / wd 0.1 / lr 5e-4), against 0.4334 for the generalist: -0.012 to -0.028.
    - All three peaked at step 26,400 and then overfit.
    - Decision: the other units' data helps the coarse group, so the router idea was dropped. The results were committed on Oct 21.

### Oct 21 (Wed)
1. **v21 result logged** (commit 08:58). Added the `noise` option and queued the v22 → v24 chain.
2. **Leaderboard read at 11:26 IST** (date approximate). The earlier README dated this read Oct 20, but at 11:26 IST on Oct 20 the pair had not been built yet, so Oct 21 is the likely date.
   - Rank 1 0.67244, rank 8 0.67017, rank 25 0.66783, rank 50 0.66511, rank 150 0.66049; us 302 at 0.65684.
   - About +0.013 was needed for the prize line.
3. **v22** (08:49-10:15).
   - Input noise 0.05: 0.6847 / 0.5137 (-0.0010).
   - Input noise 0.15: 0.6843 / 0.5132 (-0.0014).
   - lr 5e-4: 0.6839 / 0.5125 (-0.0018).
4. **v23**, 7 epochs (10:16-12:14).
   - wd 0.05: 0.6839 / 0.5120 (-0.0018).
   - wd 0.07: 0.6849 / 0.5139 (-0.0008).
   - wd 0.1: 0.6852 / 0.5145 (-0.0005).
5. **v24**, a third data seed (12:14-13:35).
   - wd 0.05: 0.6856 / 0.5135 (-0.0001).
   - wd 0.07: 0.6849 / 0.5132 (-0.0008).
   - wd 0.1: 0.6851 / 0.5140 (-0.0006).
   - Across data seeds 0, 1 and 2 the recipe lands at 0.6848-0.6857.
6. **ensemble_check10** (14:49).
   - The best of 55 pairs is v18c+v24a: 0.6869 scored and 0.6938 on the 469, only +0.0002 / +0.0001 over v18c+v19a.
   - The residual correlations are 0.989-0.996.
   - Build rule not met, so nothing was built that day.
   - Decision: the wd recipe is saturated.

### Oct 22 (Thu)
No logged work (no commits, runs or result files dated this day).

### Oct 23 (Fri)
1. **Mukesh's pull request #2 merged** (a team member's name in the README).
2. **Teacher predictions** (`make_teacher.py`, 515 s): the submitted pair's predictions on every train row.
   - Caveat: the teacher has seen most of the train rows. On rows both members trained on it sits about +0.03 above its half-seen holdout scores.
3. **v25**, K = 6 (09:37-11:06). Distillation runs:
   - kd 0.3: 0.6843 / 0.5146 (-0.0014).
   - kd 0.5: 0.6852 / 0.5151 (-0.0005).
   - kd 1.0: 0.6858 / 0.5158 (**+0.0001**, the best of any run since v18c).
   - kd 0.5 with wd 0.1: 0.6857 / 0.5155 (0.0000).
   
   Row-weight runs:
   - abs(y)^0.5: 0.6820 / 0.5098 (-0.0037).
   - abs(y)^1.5: 0.6834 / 0.5129 (-0.0023).
   
   Distillation learns faster but finishes level; the metric's own abs(y) weighting is the right one.
4. **v18c+v24a int8 built on request** (zip 12:35, 335,138 bytes), even though ensemble_check10's rule was not met.
   - int8 loss +0.000033 on the first 100 valid sequences.
   - 112.9 us/row against 114.5 for v18c+v19a in the same bench, projected 39.2 min.
   - Submitted **October 23**, as recorded in the submission history; public score **0.65596**. This corrects the earlier approximate submission date.
5. **v26**, K = 6 (12:55-14:17).
   - Training chunk 2,000: 0.6829 / 0.5115 (-0.0028).
   - Training chunk 4,000: 0.6819 / 0.5091 (-0.0038).
   - Feature dropout 0.1: 0.6835 / 0.5129 (-0.0022).
   - Feature dropout 0.2: 0.6826 / 0.5113 (-0.0031).
   - Classification head, weight 0.5: 0.6844 / 0.5141 (-0.0013).
   - Classification head, weight 1.0: 0.6845 / 0.5145 (-0.0012).
6. **Mistake** (date approximate, Oct 23-25): an `EXPERIMENTS.md` row written with `|clipped y|` broke the table, because the pipes count as columns. It now reads "abs(clipped y)".

### Oct 24 (Sat)
No logged work.

### Oct 25 (Sun)
1. **Feature feasibility** (`feature_feasibility.py`, 16:55).
   - The 112 columns are quantile-transformed, and the price index is not a depth level.
   - So mid, spread, imbalance and microprice cannot be rebuilt exactly. The proxy-feature check changed the held-out linear-model score by -0.02 to +0.005, without a consistent gain.
   - No new-feature run.
2. **Output post-filters** (`output_filter_check.py`, 17:04): smoothing and a learned causal filter lose in every fold (-0.0003 to -0.035). Closed.
3. **v27**, input-group ablation, K = 6 (17:29-18:50). Each run sets one input group to zero.
   - Volumes: -0.0285.
   - All of i1: -0.0094.
   - Price levels: -0.0039.
   - Anonymous columns: -0.0035.
   - Trade columns: -0.0029.
   - i1 prices: -0.0015.
   - Every group carries signal, so there is nothing to remove.

### Oct 26 (Mon)
1. **v26 and v27 results committed** (04:02).
2. **v28**, extra volume-change inputs, K = 6 (04:07-05:28).
   - xvd=all: 0.6840 / 0.5131 (-0.0017).
   - xvd=i1: 0.6844 / 0.5134 (-0.0013).
   - xagg: 0.6848 / 0.5141 (-0.0009).
   - Both: 0.6837 / 0.5125 (-0.0020).
   - Init-seed-1 replicates: 0.6840 (-0.0017) and 0.6835 (-0.0022).
   - None helps; the GRU builds the changes itself.
3. **Starter-pack docs read** (data_overview, rules, faq, timeline).
   - Scored rows are `need_prediction AND is_scored`.
   - The number of final selections is not specified.
   - The submission cutoff was recorded as 23:59 UTC / 05:29 IST the next day. The later final-selection email specifies a separate deadline; see the November 16 entry.
   - Bid and ask are split by column index.

### Oct 27 (Tue)
No logged work.

### Oct 28 (Wed)
No logged work.

### Oct 29 (Thu)
No logged work.

### Oct 30 (Fri)
No logged work.

### Oct 31 (Sat)
1. **v27 and v28 results and the Discord findings committed** (07:04).
   - The organisers allow ensembles, distillation and any modelling within the limits.
   - The split is by time: train, valid, public test, a small gap, then private test.
   - The data spans several markets and regimes.
   - Neither the mask rule nor the number of final selections is stated.
2. **`REPRODUCE.md`** (07:09) **and `REPORT_DRAFT.md`** (07:11), committed at 07:19.
3. **Mask GRU2** (`mask_gru2.py`, 07:21-07:31): 2 layers, hidden 128.
   - On the 469: AUC 0.9495 against 0.9414, and the top decile holds 67.6% of the scored rows (was 65.7%).
   - It is less well calibrated: mean p 11.08% against a true 9.47%.
4. **v29**, v18c retrained with the GRU2 probabilities (07:40-09:01).
   - alpha 0.5: 0.6846 / 0.5137 (-0.0011).
   - alpha 0.35: 0.6846 / 0.5121 (-0.0011).
   - alpha 0.65: 0.6849 / 0.5140 (-0.0008).
   - Verdict: a sharper mask model changes nothing; the search is over.

### Nov 1 (Sun)
No logged work. The team-merge deadline (23:59 UTC) passed.

### Nov 2 (Mon)
1. **v29 finals logged**, and the README summary for Oct 21 to Nov 2 written (commit 14:34).
2. **Leaderboard: rank 398 of 5,660** at 0.65684, down from 302 of 5,111 on Oct 20; we had no better submission since, and other teams moved past us.
3. **Discord remarks**: what was said that day is not recorded in our files. The Oct 31 facts stand.
4. **v18c+v24a public score recorded: 0.65596**. The score was known by this Nov 2 entry; the submission history records **October 23** as the submission date.
   - That is -0.00088 below v18c+v19a and +0.00054 above v7+v9, so it is our second-best scored submission and a fallback candidate.
   - On valid it had been +0.0002 / +0.0001 ahead of v18c+v19a. Valid differences that small say nothing about the leaderboard.
   - 7 leaderboard submissions used so far.
5. **ensemble_check11** (14:41, 333 s, run alone).
   - Best triple v18c+v24a+v13: 0.6873 scored / 0.6941 on the 469, +0.0006 / +0.0003 over v18c+v19a.
   - Best GRU-only triple v18c+v19a+v24a: 0.6872 / 0.6941.
   - Build rule not met, so nothing was built.
   - v13 has no `_final.pt`; its `_best.pt` is its final step and was used.
   - 168 sanity checks pass.

### Nov 3 (Tue)
No logged work.

### Nov 4 (Wed)
No logged work.

### Nov 5 (Thu)
1. **Causal unit predictions** (`unit_pred.py`, 09:46-09:49, CPU).
   - LightGBM on the first 20 steps.
   - Train predictions are 5-fold out of fold, so no sequence is predicted by a model that saw its label: 0.99991 accurate (one wrong, sequence 3018).
   - Valid: 1.00000.
2. **`train_multi.py` options `unitemb` and `unitpred`**: a learned embedding of the predicted unit, "unknown" before step 20.
   - Smoke tests on GPU 1: with K = 1 the defaults reproduce `train_v2m.py`'s lines exactly (6 of 6). The three v30 models ran end to end.
3. **v30**, the regime (unit) embedding, K = 3 (09:52-11:13, 80 min).
   - **v30a**, embedding size 8: 0.6847 / 0.5142 (-0.0010 / +0.0004).
   - **v30b**, size 16: 0.6853 / 0.5146 (-0.0004 / +0.0008).
   - **v30c**, size 8 with init seed 1: 0.6848 / 0.5143 (-0.0009 / +0.0005).
   - **No measured gain.** The result is consistent with the unit probe, but does not directly establish what the GRU represents internally.
   - A per-unit (coarse-group) evaluation was not run.
4. **Correction:** I first wrote that v30's training and holdout sequences all got the right unit. The run log shows 0.99990 on the training sequences (sequence 3018 is among them) and 1.00000 on the holdout. Fixed in `EXPERIMENTS.md`.
5. **Leaderboard read: rank 419 of 5,711 at 0.65684.**
   - Rank 1 0.67586, rank 8 (musya) 0.67257, rank 22 0.66950, rank 100 0.66426, rank 200 0.66122, rank 300 about 0.6586.
   - The prize line (rank 8) is +0.0157 away.
6. **README rewritten** as this daily log.
7. **v31, unit-normalised outputs**, K = 6 (11:55-13:21 UTC, 86 min; commit 13:36). New `train_multi.py` option `ynorm`.
   - Why: every target is an integer multiple of its sequence's unit (0.1254 / 0.2413 / 0.4685 / 0.6569), and the coarse unit 0.6569 holds 48% of the metric weight and 66% of the error (unit analysis, Oct 20). The idea was to let the GRU predict in unit counts, with one shared output scale for all four units.
   - How: the prediction is clip(o * u, -2, 2), where u is the unit of the causally predicted class (`unit_pred.py`; u = 1 before step 20). No targets are used for u.
     - Mode a keeps the loss unchanged: weighted MSE on o * u.
     - Mode b fits o to y / u, with weights |y / u| + 0.05.
     - Smoke tests on GPU 1: the defaults reproduce `train_v2m.py`'s lines (6 of 6), and mode a with u fixed to 1 reproduces the default loss bit for bit.
   - Finals (scored / all rows; v18c 0.6857 / 0.5138):
     - **v31a / v31b**, mode a, init seed 0 / 1: 0.5881 / 0.4522 and 0.5873 / 0.4516 (-0.0976 / -0.0984 scored).
     - **v31c / v31d**, mode b, init seed 0 / 1: 0.6086 / 0.4521 and 0.6084 / 0.4506 (-0.0772 / -0.0773).
     - **v31e**, mode a plus the 8-wide unit embedding: 0.6822 / 0.5094 (-0.0035).
     - **v31f**, mode b plus the embedding: 0.6805 / 0.5041 (-0.0052).
     - All six are below the control.
   - **In-group scores** (`unit_group_eval2.py`, 13:22-13:24, run alone; 22 sanity checks pass; `unit_group_eval2_results.txt`).
     - Coarse unit 0.6569, scored rows: v18c 0.4334; v31a 0.4311, v31b 0.4312, v31c 0.4051, v31d 0.4056, v31e 0.4320, v31f 0.4224.
     - No v31 model beats v18c in any unit group on scored rows.
     - Within each group, mode a is only 0.002-0.006 below v18c, yet 0.098 below overall. So most of the loss comes from how the four groups combine in the pooled metric, not from the fit inside each group.
   - **Conclusion:** one output scale shared across units, in unit counts, does not help. Our reading, not tested directly: the regimes differ in their tick structure, not just in size.
   - Not submittable as is: the exporters do not multiply by u.
8. **Research and Discord review** (evening IST; the time is not logged).
   - What we read:
     - the public Discord chat;
     - LOB papers (DeepLOB / TLOB-style models, and a paper on crypto LOB inputs);
     - contest write-ups and repos (Jane Street, Optiver, Ubiquant, DRW, and earlier Wunder challenges).
   - What others report:
     - the same plateau, about +0.001 per idea;
     - distillation does not help;
     - a public-versus-local gap of 0.015 to 0.027 is normal. Ours is about 0.03 (v18c + v19a: 0.6867 on valid, 0.65684 public).
     - Nobody discloses a single big idea.
   - **Incomplete:** the Kaggle write-up pages could not be read.
   - Ideas we had already tested:
     - distillation (v25, at best +0.0001);
     - ensembles (ensemble_check10 / 11);
     - EMA (v20a);
     - input noise and feature dropout (v22, v26c / d);
     - longer training and backprop (v23, v26a / b);
     - order-flow inputs (v28);
     - classification heads (v26e / f);
     - regime (unit) inputs (v30, v31).
   - The one new idea was training on the metric itself (a Pearson loss), tested as v32.
9. **v32, Pearson loss**, K = 3 (14:51-16:15 UTC, 84 min; commit 26482d9 at 16:18). New `train_multi.py` option `loss`.
   - pearson = 1 - mean(r_t0, r_t1), the weighted Pearson over all rows >= 99 of the batch's chunk, pooled, with weights |y| * m. Hand check: it equals the metric's own accumulator on the same rows to 1e-12.
   - mix = wMSE / wMSE(zero prediction) + (1 - r).
   - Finals (scored / all rows; v18c 0.6857 / 0.5138):
     - **v32a**, pearson: 0.6852 / 0.5139 (-0.0005 / +0.0001).
     - **v32b**, mix: 0.6853 / 0.5143 (-0.0004 / +0.0005).
     - **v32c**, pearson with alpha 0.0: 0.6842 / 0.5039 (-0.0015 / -0.0100). This is the same pattern as v8c, wMSE with alpha 0.0, which lost 0.0089 on all rows against v4.
   - Output scale: a* = Σ w y o / Σ w o² on valid scored rows is 0.94-1.00 for all three (v18c 0.99). Scaling changes no score by more than 0.0001 (`pearson_scale_eval.py`, 16:15-16:17, run alone; 8 sanity checks pass).
   - **No gain.** wMSE with these weights already tracks the metric.
10. **Modelling paused at this point.** Seven experiments in a row (v26 to v32) gave no gain over v18c. Work resumed on November 8; the subsequent tests and final decisions are recorded below.

### Nov 8 (Sun)

1. **v33 preparation and controlled training.** [V33.md](V33.md) records the hypothesis and checks: unchanged 168-input preprocessing; A, the original GRU; B, a learned 32-dimensional tanh projection; C, that projection plus deviations from causal exponential averages with half-lives 8, 64 and 512 rows. B/C feed a two-layer GRU-128. Averages and hidden state carry across 1,000-row chunks, detach at training boundaries and update throughout warm-up.
2. **Matched full comparison:** shared batches, seed-0 split/order/statistics, mask weighting alpha 0.5, weight decay 0.05 and five epochs, **64,500 updates per model**. The requested run label is `v33_20261109_abc`, but actual training ran **November 8, 19:08:53–20:54:51 UTC** (**November 9, 00:38:53–02:24:51 IST**), elapsed **1h45m58s**. A reproduced historical v18c weights bitwise. All best checkpoints were final.

| v33 configuration | Final FP32 scored WP | All-required WP | Train-holdout WP |
|---|---:|---:|---:|
| A: original GRU | 0.6857267 | 0.5138374 | 0.4524342 |
| B: projection only | 0.6833059 | 0.5112214 | 0.4495886 |
| C: multiscale context | 0.6846189 | 0.5132822 | 0.4522493 |

C improved on B by **0.0013130**, but remained **0.0011078 below A**. [V33_RUN_20261109.md](V33_RUN_20261109.md) is the historical launch record; the completed results are recorded in the [November 9 v33 closeout](EXPERIMENTS.md#v33-investigation-closed--november-9-2026), not a currently running job.

### Nov 9 (Mon)

1. **v33 ensemble complementarity.** One shared pass over all 1,873 validation sequences, each model's own preprocessing and raw equal-weight averages before clipping, reproduced the incumbent at **0.6867397281**.

| Fixed pair | FP32 scored WP | Difference from incumbent | Paired 95% interval |
|---|---:|---:|---|
| C + v19a | 0.6866800 | -0.0000597 | [-0.0005957, +0.0004258] |
| C + v18c | 0.6868471 | +0.0001074 | [-0.0004362, +0.0006303] |
| B + v19a | 0.6858353 | -0.0009045 | [-0.0017080, -0.0002612] |

   Neither C pair established a meaningful scored-WP improvement. These percentile intervals use 2,000 paired whole-sequence resamples, seed 20261109; they do not include training randomness or prior model selection. The reused 469-sequence subset is diagnostic. Per-sequence statistics, bootstrap draws and logs remain in `v33_results/ensemble_20261109_0536/`. A later shell status of 1 came from `ps` finding the already-finished evaluator, not an evaluation failure.

2. **v34 architecture prepared, never fully trained.** Existing preprocessing → 168-to-64 projection/ReLU → eight residual depthwise temporal convolutions, kernel 3, dilations 1–128, with pointwise mixing → two-output head. The model has **46,274 parameters**, a **511-row receptive field** and **127.5 KiB** of per-sequence history. Causality, warm-up, resets, gradients and chunk/stream equivalence passed. Only a **100-update real-cache preflight** was run; no full v34 accuracy result exists. See [V34_REPORT_20261109.md](V34_REPORT_20261109.md).
3. **v34 streaming optimized without changing the architecture.** The first callback benchmark measured **286.19 µs/row FP32**, **292.29 INT8** and **152.34** for the deployed pair. Profiling found that the graph already used three temporal taps, but performed unnecessary history resets/shifts and indexed operations. Circular histories and prebound IOBinding reduced state writes to 2 KiB/row, with one ORT call per row. In a separate five-repeat two-way benchmark, optimized FP32 v34 measured **122.46 µs/row** versus **134.84** for the deployed pair (**0.908×**), passing the 1.2× screen. Output error versus batched inference was at most **4.77e-7**. Different benchmark interleavings have different absolute times; these are measured local callbacks, not platform-runtime guarantees. The tests used smoke weights; v34 was neither fully trained nor submitted. See [V34_STREAM_OPT_20261109.md](V34_STREAM_OPT_20261109.md).

### Nov 14 (Sat)

1. **Final fixed-checkpoint audit.** The exact saved v18c, v19a, v24a and v13 LSTM checkpoints were evaluated in one shared full-validation pass. The incumbent reproduced **0.6867397281**; all audited metrics matched `ensemble_check11` within **2.87e-12**.

| Fixed blend | FP32 scored WP | All-required WP | Difference from incumbent | Paired 95% interval |
|---|---:|---:|---:|---|
| v18c + v19a | 0.6867397 | 0.5164550 | 0 | reference |
| v18c + v24a + v13 | 0.6873498 | 0.5176482 | +0.0006100 | [-0.0000344, +0.0013085] |
| v18c + v19a + v13 | 0.6871564 | 0.5176119 | +0.0004166 | [-0.0000916, +0.0009430] |

   Both candidate intervals include zero (2,000 paired sequence resamples, seed 20261114), and neither gain met the existing +0.0015 build rule. The strongest triple's actual exported callback measured **329.33 µs/row FP32 / 213.99 INT8**, versus **128.79** for the deployed incumbent; INT8 resident memory was **52.4 MiB**. Its local full-test estimate was **140.5 minutes**, or **180.7** with 25% plus five-minute margin. Even the historical 1.89× platform-speed scenario gave **97.9 minutes with margin**; that scenario was not a platform measurement. Audit-only graphs were built, but **the triples were not submitted**, and full-validation INT8 score retention was not established. The incumbent was retained. See [FINAL_ENSEMBLE_AUDIT_20261114.md](FINAL_ENSEMBLE_AUDIT_20261114.md).

2. **Final bounded residual-head experiment.** The saved v18c GRU and original output head were frozen. A was untouched; B added a zero-initialized linear residual on hidden state; C added a 128→32→2 ReLU residual with zero output initialization. A 100-update training-only pilot set one fixed budget: **25,800 updates per head, two epochs**, with shared frozen-backbone computation, identical masks/loss/batches and no validation-target training. The backbone remained bitwise unchanged.

| Model or pair | FP32 scored WP | All-required WP | Difference from incumbent |
|---|---:|---:|---:|
| A | 0.6857267071 | 0.5138373995 | -0.0010130211 |
| B | 0.6853658549 | 0.5130313474 | -0.0013738732 |
| C | 0.6854536465 | 0.5130385663 | -0.0012860816 |
| A + v19a | 0.6867397281 | 0.5164549890 | 0 |
| B + v19a | 0.6867321719 | 0.5163323561 | -0.0000075562 |
| C + v19a | 0.6866978683 | 0.5162292958 | -0.0000418598 |

   The attempt completed in **32.1 minutes**, from **19:52:18 to 20:24:23 UTC**, including the interrupted initial inspection; training took **23m36s**, full validation **91.62s**. **Neither residual-head pair improved the incumbent. No residual candidate ZIP was created.** The accuracy gate failed, so export/runtime certification was not pursued; the prepared exporter helper was not executed. This one-seed bounded result does not establish a general limit on residual models. The attempt ended with the incumbent retained. See [BOUNDED_RESIDUAL_FINAL_20261114.md](BOUNDED_RESIDUAL_FINAL_20261114.md).

### Nov 16 (Mon)

**Final selection completed.** The three submissions in [Final selections and results](#4-final-selections-and-results) were selected and locked at approximately **17:21 IST / 11:51 UTC** on November 16. Public rank at the time was **495/5,840**. Private evaluation was outstanding at this stage; the final result is recorded in the November 20 entry below.

The incumbent's recorded platform runtime/memory is **36m07s / 624 MB**, distinct from local estimates. The separate submission and selection deadlines, automatic-selection rule and organiser timeout assurance are recorded in [Final selection and deadlines](#final-selection-and-deadlines-confirmed-november-16). Our selection is complete and cannot be changed; no re-upload or further experiment is required. The repository was private during this stage of the competition.

### Nov 20 (Fri)

**Final result: 412th, private score 0.66582.** The final leaderboard records an 83-place improvement from public rank 495. Its best-submission identifier, `1VH*`, matches the selected v18c + v19a submission (`1VH23TAR`). The other selected submissions’ private scores are not recorded here. Competition modelling and selection are complete.

### Reference tables

Historical summary tables through November 5; the later v33/v34, audit and residual-head results are in the dated entries above. Unless a row says otherwise, the v20–v32 generalist runs used v18c's settings: hidden 128, 2 layers, no dropout, alpha 0.5, wd 0.05, lr 1e-3, 5 epochs, data seed 0, init seed 0. Those runs share v18c's batches, so the gap comes from the change alone. The full rows, with every evaluation step, are in `EXPERIMENTS.md`.

**Mask-weighted loss** (Oct 19, `train_v2m.py`). The loss weight is multiplied by m = alpha + (1 - alpha) * min(p / mean_p, 20). All runs: hidden 128, 5 epochs, `--n-val 1873 --n-holdout 300`.

| run | setup | best / final valid (all 1,873) | 469 held-out | all rows >= 99 (full valid) |
|---|---|---|---|---|
| v4 | plain, seed 0 | 0.6788 / 0.6778 | 0.6845 | 0.5096 |
| v7 | alpha 0.5, seed 0 | 0.6824 / 0.6824 | 0.6892 | 0.5098 |
| v8c | alpha 0.0 (weight proportional to p) | 0.6811 / 0.6806 | | 0.5007 |
| v9 | alpha 0.5, seed 1 | 0.6826 / 0.6820 | 0.6889 | |
| v10 | alpha 0.5, seed 2 | 0.6816 / 0.6815 | 0.6863 | |

**Ideas that did not work, Oct 19-20** (each measured)

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

**Weight decay** (Oct 20, AdamW; the old default was 0.01)

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
| v24a / v24b / v24c | 0.05 / 0.07 / 0.1 | 0 | 2 | 0.6856 / 0.6849 / 0.6851 | 0.5135 / 0.5132 / 0.5140 |

**Training-recipe tests (v20, v22, v23)**

| run | tested | final (scored / all rows) | vs v18c (scored) |
|---|---|---|---|
| v20a | EMA of the weights, decay 0.9995 | 0.6857 / 0.5139 | 0.0000 (+0.0093 early, gone by the end of the cosine schedule) |
| v20b | peak lr 2e-3 | 0.6841 / 0.5133 | -0.0016 |
| v20c | hidden 192 | 0.6845 / 0.5150 | -0.0012, and about 6x slower to train |
| v22a / v22b | Gaussian input noise 0.05 / 0.15 | 0.6847 / 0.5137, 0.6843 / 0.5132 | -0.0010 / -0.0014 |
| v22c | peak lr 5e-4 | 0.6839 / 0.5125 | -0.0018 |
| v23a / v23b / v23c | 7 epochs, wd 0.05 / 0.07 / 0.1 | 0.6839 / 0.5120, 0.6849 / 0.5139, 0.6852 / 0.5145 | -0.0018 / -0.0008 / -0.0005 (v23c is +0.0004 over v18a, 5 epochs at wd 0.1) |

**Distillation and row weights (v25)**

| run | tested | final (scored / all rows) | vs v18c (scored) |
|---|---|---|---|
| v25f / v25a / v25b | distillation weight kd 0.3 / 0.5 / 1.0 | 0.6843 / 0.5146, 0.6852 / 0.5151, 0.6858 / 0.5158 | -0.0014 / -0.0005 / +0.0001 |
| v25c | kd 0.5 with wd 0.1 | 0.6857 / 0.5155 | 0.0000 |
| v25d / v25e | row weight abs(y)^0.5 / abs(y)^1.5 instead of abs(y) | 0.6820 / 0.5098, 0.6834 / 0.5129 | -0.0037 / -0.0023 |

**Backprop length, feature dropout, classification head (v26)**

| run | tested | final (scored / all rows) | vs v18c (scored) |
|---|---|---|---|
| v26a / v26b | training chunk 2,000 / 4,000 steps (longer backprop, fewer optimizer steps) | 0.6829 / 0.5115, 0.6819 / 0.5091 | -0.0028 / -0.0038 |
| v26c / v26d | input-column dropout 0.1 / 0.2 | 0.6835 / 0.5129, 0.6826 / 0.5113 | -0.0022 / -0.0031 |
| v26e / v26f | auxiliary 21-bin classification head, weight 0.5 / 1.0 | 0.6844 / 0.5141, 0.6845 / 0.5145 | -0.0013 / -0.0012 |

**Which inputs matter (v27, one input group set to zero)**

| run | group dropped | final (scored / all rows) | vs v18c (scored) |
|---|---|---|---|
| v27c | the 44 volume-like columns | 0.6572 / 0.4876 | **-0.0285** |
| v27a | the whole second instrument i1 (78 inputs) | 0.6763 / 0.5040 | -0.0094 |
| v27f | absolute price levels (their changes kept) | 0.6818 / 0.5105 | -0.0039 |
| v27e | the 8 anonymous columns and their flags | 0.6822 / 0.5109 | -0.0035 |
| v27d | trade columns and their flags | 0.6828 / 0.5107 | -0.0029 |
| v27b | i1 prices and their changes only | 0.6842 / 0.5121 | -0.0015 |

**Explicit volume changes as extra inputs (v28)**

| run | added inputs | final (scored / all rows) | vs v18c (scored) |
|---|---|---|---|
| v28a / v28e | one-step changes of all 44 volume columns (init seed 0 / 1) | 0.6840 / 0.5131, 0.6840 / 0.5133 | -0.0017 / -0.0017 |
| v28b | changes of the 22 i1 volumes | 0.6844 / 0.5134 | -0.0013 |
| v28c | 6 order-flow aggregates (bid, ask and bid minus ask per instrument) | 0.6848 / 0.5141 | -0.0009 |
| v28d / v28f | both (init seed 0 / 1) | 0.6837 / 0.5125, 0.6835 / 0.5117 | -0.0020 / -0.0022 |

**A sharper mask model (mask GRU2, v29).** Only the mask probabilities differ from v18c.

| run | alpha | final (scored / all rows) | vs v18c (scored / all rows) |
|---|---|---|---|
| v29b | 0.35 | 0.6846 / 0.5121 | -0.0011 / -0.0017 |
| v29a | 0.5 | 0.6846 / 0.5137 | -0.0011 / -0.0001 |
| v29c | 0.65 | 0.6849 / 0.5140 | -0.0008 / +0.0002 |

**Unit (regime) embedding (v30).** The predicted unit class is embedded and appended to the input. The class is "unknown" before step 20. The input layer is larger (176 or 184 inputs), so the initial weights are not v18c's.

| run | embedding | final (scored / all rows) | vs v18c (scored / all rows) |
|---|---|---|---|
| v30a | 8, init seed 0 | 0.6847 / 0.5142 | -0.0010 / +0.0004 |
| v30b | 16, init seed 0 | 0.6853 / 0.5146 | -0.0004 / +0.0008 |
| v30c | 8, init seed 1 | 0.6848 / 0.5143 | -0.0009 / +0.0005 |

**Historical public-submission results** (7 submissions, including the baseline calibration)

The local column retains the original results: the starter/v2 scores came from official-scoring checks; later ensemble values are FP32 averages, distinct from their deployed INT8 public scores.

| date | submission | local valid | leaderboard | rank | share of the valid gain kept |
|---|---|---|---|---|---|
| Oct 17 | `submission_v2_int8_std.zip` | 0.679703 | 0.65200 | about 532 | |
| Oct 18 | starter baseline (calibration) | 0.617052 | 0.59353 | | |
| Oct 18 | `submission_ens_v4v5_int8.zip` | 0.6822 | 0.65315 | 486 of 5,016 | about 46% |
| Oct 19 | `submission_v7_int8.zip` | 0.6824 | 0.65367 | 470 of 5,051 | about 60% |
| Oct 19 | `submission_ens_v7v9_int8.zip` | 0.6850 | 0.65542 | 372 of 5,057 | about 67% |
| Oct 20 | `submission_ens_v18cv19a_int8.zip` | 0.6867 | **0.65684** | **302 of 5,111** | about 79% |
| Oct 23 | `submission_ens_v18cv24a_int8.zip` | 0.6869 | 0.65596 | not recorded | below v18c+v19a despite +0.0002 on valid |

Observed public rank at 0.65684: 398/5,660 (Nov 2), 419/5,711 (Nov 5), and **495/5,840 (Nov 16)**. These are historical public standings; the final private placement is **412th**.

**Leaderboard snapshots**

| rank | Oct 18 (23:30) | Oct 21 (11:26 IST, date approximate) | Nov 5 |
|---|---|---|---|
| 1 | 0.67243 | 0.67244 | 0.67586 |
| 8 | 0.66992 | 0.67017 | 0.67257 (musya) |
| 22 | | | 0.66950 |
| 25 | 0.66783 | 0.66783 | |
| 50 | | 0.66511 | |
| 100 | | | 0.66426 |
| 150 | | 0.66049 | |
| 200 | | | 0.66122 |
| 300 | 0.65655 | | about 0.6586 |
| us | 486 at 0.65315 | 302 at 0.65684 | 419 at 0.65684 |

---

</details>

## 4. Final selections and results

All three rows below are **selected, submitted and locked**. Public scores are for deployed **INT8** submissions; local scores are **FP32** raw-prediction averages on all 1,873 validation sequences, not full-validation INT8 measurements. Earlier submissions remain in the historical table above.

| Selected submission | Submitted | FP32 scored-row validation WP | INT8 public score |
|---|---|---:|---:|
| `submission_ens_v18cv19a_int8.zip` | Oct 20 | 0.6867397 | **0.65684** |
| `submission_ens_v18cv24a_int8.zip` | Oct 23 | 0.6869 | **0.65596** |
| `submission_ens_v7v9_int8.zip` | Oct 19 | 0.6850 | **0.65542** |

Final selection was confirmed on **November 16 at approximately 17:21 IST / 11:51 UTC**. The final result, recorded November 20, is **412th with a private score of 0.66582**. The best submission was v18c + v19a, matching the final leaderboard’s `1VH*` identifier to submission `1VH23TAR`. Its public evaluation completed in **36m07s with 624 MB memory**; these are public-platform measurements, not separately measured private-evaluation timings. Individual private scores for the other two selections are not available in this record.

The unchanged-baseline calibration also showed a local-to-public drop (0.617052 to 0.59353), supporting a difference between validation and test periods. It does not rule out model-selection effects. Small validation gains did not reliably transfer: v18c+v24a led v18c+v19a locally by about 0.0002 but scored 0.00088 lower publicly. Later v33, triple and residual-head checks did not justify replacing the incumbent. The repeatedly reused 469-sequence subset is diagnostic, not independent confirmation.

---

## 5. What went wrong or was misjudged (so we do not repeat it)

- Initial time budget estimate (10 us/row) was wrong; the real one is about 90 us/row.
- Disk space looked tight because we read the wrong filesystem.
- Claude Code required Node 22, not 20.
- Our first model (hidden 256) was too slow for the CPU limit; we caught this by timing before spending a submission.
- First int8 attempt lost score; fixed by keeping standardization explicit.
- Repo was initially public; fixed to private.
- On Oct 17 I explained the 0.652 vs 0.680 gap as checkpoint-selection optimism. The baseline also scored lower publicly, supporting a difference between evaluation periods without excluding model-selection effects. Measure with a calibration submission before explaining a gap.
- We treated single-run differences of 0.001 as meaningful before measuring seed noise (about 0.0013). Always check seed noise before reading anything into small differences.
- The v3 run (8 epochs) overfit: best at step 64,000 (0.6783), final 0.6704. We select the best checkpoint, not the last.
- Instructions I gave for the seed runs guessed the checkpoint file names (`..._fullval_best.pt`); the actual file for v5 was `train_v5_h128_e5_seed1_best.pt`, and Claude Code had to find it. Check `ls ~/work/checkpoints` before naming files.
- `export_v2.py --int8` also needs `--cell matmul`; the reproduce outline had left it out.
- Ensembling two models in one ONNX graph did not cut the per-row cost (114 vs 2 x 60 us). Do not assume a shared graph is cheaper; time it.
- Running two training jobs at once (even on two GPUs) made both about 5 times slower, because the data sits on one hard disk. I first said both would finish by 10 PM, which was wrong. Later I again asked for an ensemble check to run next to a training job and it cost about 10 minutes. Anything that reads the data waits until training is done.
- The training scripts forced GPU 0 whatever `CUDA_VISIBLE_DEVICES` said, which is why one teammate's "GPU 1" run slowed ours down. Check `nvidia-smi` memory on the GPU you meant after launching.
- Early three-model runtime projections were about 48–51 minutes and were judged too risky; they were estimates, not platform measurements. The November 14 mixed-triple export was substantially slower in a matched callback benchmark. The selected incumbent later had a recorded platform runtime of 36m07s.
- Gains selected as "best of N" are slightly optimistic (the weight-decay pair was the best of 10). The leaderboard check showed 79% carried over, but plan for 50-70%.
- The 469-sequence comparisons were sensitive to sequence composition: v16a looked behind v7 on that subset and ahead on the other 1,404 sequences. Later paired sequence bootstraps quantify sampling uncertainty, but neither those intervals nor the reused subset account for training randomness or prior selection.
- A sanity check that fails is information. `ensemble_check9.py` stopped on a reference-value mismatch; the models were fine and the check's lookup was wrong. Fix the check, never raise its tolerance.
- Shell detail: in a waiter script `echo "$(date) ... $?"` resets `$?` before it is printed; store `$?` in a variable first.
- `ps -eo args` truncates long commands and `pgrep -f` also matches the wrapper shell; find the python PID once with `ps -ww -eo pid,comm,args` and wait with `kill -0 PID`.
- A valid lead of +0.0002 means nothing on the leaderboard: v18c+v24a (valid 0.6869) scored 0.65596 against 0.65684 for v18c+v19a (valid 0.6867). Keep the +0.0015 build rule.
- v9 was run with a trainer that saved only the best checkpoint, so it has no `_final.pt`; later checks use its best (step 51,200).
- A `git push` was rejected once after a commit made from another tab; `git pull --rebase` and then `git push` fixed it. Never force-push.

---

## 6. Closeout tasks

- **Competition complete:** final rank **412th**, private score **0.66582**; all three final selections were confirmed on November 16.
- **Documentation complete:** the [technical report](REPORT.md), dated research history and experiment ledger record the final result and the unsuccessful final experiments.
- **Artifact backup verified:** the 308 MB archive passed SHA-256 and archive-readability checks after download. It includes the repository, checkpoints and selected submission directories, but not raw data, generated caches, installed environments or every loose workstation file.
- **Reproducibility limits retained:** [REPRODUCE.md](REPRODUCE.md) records the reconstruction procedure and the checks actually performed. A fresh end-to-end reconstruction and full-validation INT8 check of the final pair are not claimed.

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
| `mask_predict.py`, `mask_predict_results.txt` | valid mask diagnostic: block structure, LightGBM predictability from causal features |
| `mask_gru.py`, `mask_gru_results.txt`, `mask_gru_heldout_scores.txt` | small GRU that predicts the valid `is_scored` mask; makes `train_maskp.npy` / `valid_maskp.npy`; main-model scores by mask-fit split |
| `mask_gru2.py` | larger mask GRU (2 layers, hidden 128; AUC 0.9495); makes `train_maskp2.npy` / `valid_maskp2.npy` (v29, no gain) |
| `train_v2m.py` | GRU training with the mask-weighted loss (`--mask-p`, `--alpha`) |
| `train_v3m.py` | same, optionally also trained on valid (`--add-valid`); no gain |
| `train_v4m.py` | adds `--arch lstm`; no gain |
| `train_v5m.py`, `strip_aux.py`, `aux_checks.py` | auxiliary shifted-target heads (`--aux`, `--aux-weight`), converter to a normal 2-output checkpoint, alignment checks; no gain |
| `train_multi.py` | trains K models from one stream of batches (per-model seed, alpha, layers, dropout, weight decay, lr, hidden size, EMA, auxiliary heads, input noise, distillation `kd`, row-weight power `wpow`, training chunk, feature dropout `fdrop`, classification head `cls`, input-group ablation `drop`, extra volume inputs `xvd` / `xagg`, unit embedding `unitemb` / `unitpred`, and `--unit-filter` for specialists) |
| `make_teacher.py`, `teacher_results.txt` | teacher predictions of the submitted pair on all train rows (for distillation, v25) |
| `export_v4.py`, `solution_v4.py`, `lstm_export_check.py` | LSTM export and its checks |
| `calib_check.py`, `calib_check_results.txt` | scaling by mask probability (no gain) |
| `ensemble_check3.py` to `ensemble_check11.py` (+ `_results.txt`) | ensemble, pair and (check11) triple checks on the 469 held-out and all 1,873 valid sequences |
| `output_filter_check.py`, `output_filter_check_results.txt` | causal smoothing and learned filters of the predictions (all lose) |
| `feature_feasibility.py`, `feature_feasibility_results.txt` | what the anonymised columns are (quantile-transformed) and whether book features can be rebuilt (no) |
| `io_bench.py` | disk read speed test (why training is disk-bound) |
| `error_analysis.py`, `probe_features.py`, `residual_probe.py` | where the loss sits; GBM residual probe (no signal) |
| `target_structure.py` | targets are integer multiples of a per-sequence unit; autocorrelation; relation to prices |
| `unit_analysis.py`, `unit_probe.py`, `unit_labels.py`, `unit_group_eval.py` | per-unit score breakdown, unit recognition from the first steps, unit labels, in-group scores of the generalists |
| `unit_pred.py`, `unit_pred_results.txt` | causal unit class per sequence from the first 20 steps (out of fold on train) for `unitpred=1` (v30) |
| [EXPERIMENTS.md](EXPERIMENTS.md) | preserved experiment record, historical launch notes and November 16 closeout |
| `REPRODUCE.md` | exact pipeline for `submission_ens_v18cv19a_int8.zip`: commands, seeds, expected outputs and run times |
| [REPORT.md](REPORT.md) | completed technical report, final results and reproducibility limits |
| [V33.md](V33.md), [V33_RUN_20261109.md](V33_RUN_20261109.md) | historical v33 preparation/launch records; completed Nov 8 UTC, outcome in the Nov 9 experiment closeout |
| `v33_model.py`, `train_v33.py`, `check_v33.py`, `smoke_v33.py`, `eval_v33_pairs.py` | matched A/B/C implementation, checks and fixed-pair evaluation; evidence in `v33_results/` |
| [V34_REPORT_20261109.md](V34_REPORT_20261109.md), [V34_STREAM_OPT_20261109.md](V34_STREAM_OPT_20261109.md) | TCN smoke checks and same-architecture streaming optimization; never fully trained |
| `v34_model.py`, `train_v34.py`, `check_v34.py`, `v34_stream.py`, `bench_v34.py` | prepared causal TCN and bounded preflight/streaming checks; evidence in `v34_results/` |
| `v34_stream_ring.py`, `profile_v34_stream.py`, `bench_v34_ring.py`, `confirm_v34_ring.py` | circular-cache execution prototype, profiling and repeated matched callback measurements |
| [FINAL_ENSEMBLE_AUDIT_20261114.md](FINAL_ENSEMBLE_AUDIT_20261114.md) | final fixed-checkpoint comparisons, bootstrap intervals and exported-triple runtime assessment |
| `final_ensemble_audit.py`, `audit_mixed_stream.py`, `bench_final_audit.py`, `audit_stream_memory.py` | audit-only evaluation/export/timing tools; evidence in `audit_results/`; triples not submitted |
| [BOUNDED_RESIDUAL_FINAL_20261114.md](BOUNDED_RESIDUAL_FINAL_20261114.md) | completed 32.1-minute frozen-v18c residual-head attempt; incumbent retained |
| `bounded_residual_20261114.py`, `bounded_residual_export_check.py` | bounded head-only training/evaluation and prepared-but-unused exporter; evidence in `bounded_results/`; no candidate ZIP |

## 8. Reproduction summary and verification limits

[REPRODUCE.md](REPRODUCE.md) is the ordered reconstruction record for `submission_ens_v18cv19a_int8.zip`, with actual commands, filenames, seeds, expected outputs and verification limits. Its October 31 statements about what had not been rerun are historical; the later v33 control subsequently reproduced v18c's saved weights bitwise. No reproduction command was executed for this documentation closeout.

1. `make_cache.py` builds the supplied parquet cache. `mask_gru.py` creates the training mask-probability cache, using the recorded preprocessing-statistics provenance. Neither step should be silently substituted with a different cache or statistics sample.
2. The submitted members came from `train_multi.py`: **v18c, data/init seed 0, wd 0.05**, and **v19a, data/init seed 1, wd 0.1**. Both use GRU-128 × 2, no dropout, five epochs / 64,500 updates, batch 16, chunk 1,000, lr 0.001 cosine, alpha 0.5, n-val 1,873, n-holdout 300 and n-stats 64. Initialization seeds belong in `--models`; the script has `--data-seed`, **not `--seed`**. Use the recorded per-run commands in REPRODUCE.md rather than an illustrative mixed-seed command.
3. The incumbent export uses `/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt` and `/home/jovyan/work/checkpoints/train_v19a_wd10_ds1_final.pt`. `export_ens.py` accepts `--ckpt-a`, `--ckpt-b`, `--out`, optional `--zip` and `--fp32`. **INT8 is its default; it does not accept `--int8` or `--cell matmul`.** Those options belong to other exporters. Standardization remains explicit in the graph, and `solution_ens.py` carries state between rows and resets it between sequences.
4. `check_ens.py`, `bench_ens.py` and `run_full_valid.py` have distinct correctness, callback-timing and official-scoring roles. Their recorded checks must not be conflated: **0.6867397 is the full-validation FP32 pair score**, while **0.65684 is the submitted INT8 public score**. REPRODUCE.md records the smaller INT8 comparison, byte-identical re-exported model/solution/statistics files and the full-pipeline/full-INT8 checks that were not performed. Its approximately 5.5-hour reconstruction estimate is not evidence of a fresh end-to-end rerun.

Earlier unsupported or obsolete example commands have been removed. Preserve source versions, checkpoint hashes and existing artifacts; competition selection is complete and no rerun or re-upload is required.

