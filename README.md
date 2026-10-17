# Varanasi LOB ML

Team **Varanasi**: Alfa Connectome ML Competition (Wunder Fund), limit order book prediction.

> This repo is private on purpose. The competition rules forbid sharing code, models or detailed strategy before the competition ends. Model weights, caches and submission zips are never committed (see `.gitignore`).

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

**Data:** train 10,607 sequences (212M rows), valid 1,873 sequences (37.5M rows) plus a separate validation mask. Test: 1,970 sequences, mask hidden. The provided starter baseline is a stateful GRU scoring **0.617052** on the full validation set.

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

---

## 4. Result so far

First leaderboard submission: `submission_v2_int8_std.zip` (177 KB).

- Passed format check and scoring (scoring took 20m32s; the earlier wait was queue time).
- **Public leaderboard score: 0.65200.**
- Starter baseline: 0.617. Leaders seen on Oct 15: about 0.672 (top 10 within 0.004).
- Our team's displayed position after this submission: about 532 of 4,985 (the number shown in the site sidebar, which appears to be rank).

Why leaderboard (0.652) is lower than local (0.680): the public leaderboard scores a hidden sample with its own mask, and our checkpoint was chosen on validation, so some optimism is expected. We should measure how much our local score varies across subsets before trusting small differences.

---

## 5. What went wrong or was misjudged (so we do not repeat it)

- Initial time budget estimate (10 us/row) was wrong; the real one is about 90 us/row.
- Disk space looked tight because we read the wrong filesystem.
- Claude Code required Node 22, not 20.
- Our first model (hidden 256) was too slow for the CPU limit; we caught this by timing before spending a submission.
- First int8 attempt lost score; fixed by keeping standardization explicit.
- Repo was initially public; fixed to private.

---

## 6. Next steps

1. Measure score variance across different valid subsets and use all 1,873 sequences for checkpoint selection.
2. Train v2 longer with a lower final learning rate (8-10 epochs).
3. Use the int8 time headroom (39 of 60 minutes): try hidden about 160-192.
4. Average the last few checkpoints or ensemble small models, if time allows.
5. Try extra inputs such as multi-lag price differences, and loss variants.
6. Run experiments on free GPUs in parallel, one change at a time, each logged to its own file.
7. Keep training code seeded and reproducible (required for prize verification), and prepare the 2-8 page technical report.
8. Confirm the team is registered on the competition site (deadline Nov 1) and add teammates as repo collaborators.

---

## 7. Repo contents

| File | Purpose |
|---|---|
| `explore.py` | read-only data exploration |
| `make_cache.py` | parquet to memory-mappable float32/float16 cache |
| `train_v1.py` | GRU training (supports `--hidden` and `--name`) |
| `export_v1.py`, `export_v1_fast.py`, `export_v2.py`, `export_fast.py` | ONNX export (fp32 and int8) and `solution.py` packaging |
| `solution_v2.py`, `solution_fast.py` | submission `solution.py` templates |
| `check_v1.py` | timing, determinism, ONNX vs PyTorch and scoring check |
| `bench_vs_baseline.py` | benchmark against the starter baseline on one pinned core |
| `run_full_valid.py` | full 1,873-sequence validation with the official scorer |

## 8. How to reproduce (outline)

```
source ~/work/venv/bin/activate
python make_cache.py                       # builds ~/work/cache (about 80 GB)
CUDA_VISIBLE_DEVICES=0 python train_v1.py --hidden 128 --name train_v2
python export_v2.py --ckpt ~/work/checkpoints/train_v2_best.pt --out submission_v2_int8_std --int8 --zip <path>.zip
python run_full_valid.py                   # full validation with the official scorer
```

Check each script's `--help` for exact options.
