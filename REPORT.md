# Technical report: Team Varanasi, Alfa Connectome ML Competition

Final project report · November 20, 2026. **Final rank: 412th · Private score: 0.66582.** Experiment evidence is recorded in [EXPERIMENTS.md](EXPERIMENTS.md); reconstruction details and limitations are in [REPRODUCE.md](REPRODUCE.md).

## 1. Summary

We predict two hidden indicators of the future price movement of instrument `i0` from limit order book data, one row at a time, on one CPU core.

**Final model.** Our final model is an average of two small recurrent networks: 2-layer GRUs with 128 hidden units, about 214,000 parameters each. Both are trained on whole 20,000-step sequences with a weighted squared-error loss aligned with the scoring emphasis. Three things carried the score:
- a training loss weighted by a learned estimate of which rows are scored;
- stronger weight decay;
- averaging two models trained on different data orders.

**Deployment.** The pair is exported as one INT8 ONNX graph. The selected submission completed public-platform inference in **36m07s**, using **624 MB**. Local pinned-core callback timings vary with benchmark conditions and are not platform-runtime guarantees.

**Scores.**
- Local validation: 0.6867 on all 1,873 valid sequences, against 0.6171 for the starter baseline.
- Public leaderboard: 0.65684 (rank 302 of 5,111 on Oct 20), against 0.59353 for the baseline.

**Final result:** **412th**, private score **0.66582**. The three selected submissions were v18c + v19a, v18c + v24a and v7 + v9. The final leaderboard’s best-submission prefix `1VH*` matches v18c + v19a (`1VH23TAR`).

## 2. Task and constraints

**Data.** Each row is one moment for two instruments, `i0` and `i1`: 112 anonymised features.
- For each instrument: 22 price-like and 22 volume-like order-book columns (11 bid, 11 ask), 4 trade-price and 4 trade-volume columns.
- 8 further anonymous columns.

Rows come in independent sequences of 20,000 steps. Steps 0-98 are warm-up; steps 99-19,999 need a prediction of `t0` and `t1`. A prediction may use only the current and earlier rows of the same sequence, and the state is reset between sequences.

**Metric.** Global weighted Pearson correlation. Targets and predictions are clipped to [-2, 2], each row is weighted by |target|, all scored rows of all sequences are pooled, the score is computed per target, and the two scores are averaged. Only rows with `need_prediction AND is_scored` count. The mask is given for valid and hidden for test, and its rule is not stated, either in the docs or by the organisers on Discord.

**Limits.**
- 1 vCPU, 16 GB RAM, no GPU, no internet.
- 60 minutes for the whole test set: 1,970 sequences, 39.4M rows, so about 90 us per row.
- A 20 MB zip and deterministic output.
- 5 submissions per day.

The organisers confirmed on Discord that ensembles, distillation and any modelling are allowed within these limits.

**Split.** According to the organisers (Discord), the split is by time: train, valid, public test, a small gap, then private test, all contiguous. The private part is slightly later than valid, and the data spans several markets and regimes.

**Data sizes.** Train has 10,607 sequences (212M rows) and valid 1,873 sequences (37.5M rows).

## 3. What we learned about the data

- **Features.** The observed feature distributions are consistent with per-column normal-quantile transformation; the exact anonymization transform is not established: pooled mean 0, sd 1, values stopping near +-2.33, and a fill value of 5.1992 marking missing data in `a0`, `a5`, `a6`, `a7` and the trade-volume columns.
  - Because of this transform, mid price, spread, imbalance and microprice cannot be rebuilt exactly. The proxies we built were 87-100% linear in the inputs we already use and added nothing.
  - Price columns barely move within a sequence, so the information is in their changes.
- **Targets.** They are very persistent: autocorrelation 0.96 at lag 1, 0.46 at lag 100, about 0.03 at lag 1,000. They stay unchanged on about 92% of steps and then jump.
  - Within each sequence they are exact integer multiples of one of four units (0.1254, 0.2413, 0.4685, 0.6569), like tick counts.
  - `t0` and `t1` are negatively correlated (-0.64). `t1` is much heavier-tailed (excess kurtosis 16.9 against 4.0 for `t0`).
- **Where the score is lost.**
  - The coarsest unit (0.6569) holds 47.7% of the metric weight and 66% of the weighted squared error, and scores only about 0.43, against 0.73-0.80 for the finer units.
  - Rows at the +-2 clip are 11% of rows but 32% of the weight and 57% of the squared error.
  - The unit itself is easy to recognise from the first 20 warm-up steps (99.99% accuracy). Rescaling per unit gains only +0.0001, so the network already knows the unit.
- **The scoring mask.**
  - The valid mask selects about 9.5% of rows, evenly across steps. Scored rows are easier (larger targets, mean |target| 0.69 against 0.60): the same model scores about 0.68 on scored rows and about 0.51 on all rows ≥ 99.
  - A small GRU predicts the mask from causal inputs with AUC 0.941 on 469 valid sequences it never saw. We use this predicted probability in the training loss (section 4).
- **Which inputs matter** (each group set to zero in training and evaluation, v27; the control is 0.6857):

| inputs removed | columns | valid (masked) | change |
|---|---|---|---|
| volume-like columns of both instruments | 44 | 0.6572 | -0.0285 |
| all of `i1` | 78 | 0.6763 | -0.0094 |
| absolute price levels (changes kept) | 44 | 0.6818 | -0.0039 |
| the 8 anonymous columns and their flags | 12 | 0.6822 | -0.0035 |
| trade price and volume columns | 24 | 0.6828 | -0.0029 |
| `i1` prices and their changes | 44 | 0.6842 | -0.0015 |

Every group carries signal, so nothing is removed. The order-book volumes matter most. About 0.008 of `i1`'s value comes from its non-price columns.

## 4. Method

**Inputs (168 per row).**
- The 44 price columns and their one-step changes (0 at step 0).
- The 68 other features.
- 12 binary flags marking the fill value.

Continuous inputs use recorded per-column training statistics. The main training configuration uses `n-stats=64`; preprocessing and mask-cache statistics provenance is detailed in [REPRODUCE.md](REPRODUCE.md). The standardisation matters: without it the tiny price changes are invisible to the network.

**Network.** A 2-layer GRU (hidden 128) with a linear head that outputs `[t0, t1]` jointly. At test time it runs one row at a time and carries its hidden state, exactly as in training.

**Training.**
- Whole 20,000-step sequences, batch 16, truncated backpropagation in 1,000-step chunks with the hidden state carried (detached) between chunks.
- AdamW, peak learning rate 1e-3 with cosine decay to 0, gradient clipping 1.0, 5 epochs (64,500 steps).
- Fixed seeds and deterministic GPU kernels.

**Loss.** Weighted squared error on clipped targets and predictions, steps ≥ 99, with row weight (|y| + 0.05) x m.
- m = 0.5 + 0.5 x min(p / mean p, 20).
- p is the mask GRU's predicted probability that the row is scored.

This mask weighting changes what the model learns. Scaling the outputs by p instead lowers the score, because the metric already ignores unscored rows.

**Regularisation.** AdamW weight decay 0.05 or 0.1 (the default 0.01 was too weak). No dropout.

**Ensemble.** The average of the raw outputs of two models with different data seeds (different train/holdout split, batch order and initialisation), then clipped to [-2, 2]:
- v18c: weight decay 0.05, data seed 0;
- v19a: weight decay 0.1, data seed 1.

**Deployment.** One ONNX graph holds the shared row preprocessing, each model's standardisation as explicit nodes, two GRU stacks as explicit cells with fused matrix products, and the averaging and clip. Every matrix product is dynamically quantised to int8. The graph runs once per row through ONNX Runtime with preallocated input/output buffers.

## 5. How we validated

- **Validation sets.** Model selection uses all 1,873 valid sequences: the masked score is primary, the all-rows-≥-99 score is secondary.
  - The 469 valid sequences with index % 4 == 0 were never seen by the mask GRU. They provide a diagnostic for mask-model fitting, but were repeatedly reused for model selection and are not an untouched holdout.
  - A 300-sequence train holdout is also tracked.
- **Noise level.**
  - Seed noise of one run is about 0.0013, and differences below about 0.002 are not trusted without a replicate.
  - The score's standard deviation over random subsets of k valid sequences is about 0.27 / sqrt(k).
- **Local vs public.** Each submitted model is first scored on the full valid set with the official step-by-step scorer or the PyTorch average, then on the public leaderboard.
  - Local and public scores differ by a nearly constant offset: the baseline dropped by 0.0235 and our first model by 0.0275, consistent with a difference between evaluation periods; this does not exclude model-selection effects.
  - Only part of each valid gain carried over to the leaderboard: 46% for the first pair, 67% for the second, 79% for the third.
- **Build rule.** A new pair is built only if it gains at least +0.0015 on all 1,873 valid sequences and is not worse on the 469.

## 6. What worked and what did not

**Worked** (each with a controlled comparison: same data and initialisation unless stated):

| change | result | gain |
|---|---|---|
| hidden 128 instead of 256 | 0.6770 vs 0.6745, and 2.4x faster on CPU | +0.0025 |
| mask-weighted loss, alpha 0.5 (v7 vs v4) | 0.6824 vs 0.6788; on the 469 never seen by the mask GRU 0.6892 vs 0.6845; mean of 3 seeds +0.0028 | +0.0036 |
| weight decay 0.05 instead of 0.01 (v18c vs v7) | 0.6857 vs 0.6824; six runs with wd 0.05-0.1 on two data seeds land at 0.6842-0.6857 | +0.0033 |
| averaging two models | first pair +0.0024 over its better member (v2+v5); v7+v9 +0.0023; weight-decay pair v18c+v19a 0.6867, about +0.0010 over its better member | +0.001 to +0.0025 |

**Did not work** (control v18c 0.6857 / all rows 0.5138 unless noted):

| idea | result |
|---|---|
| mask weight alpha 0 / alpha 0.75 | 0.6811 (all rows -0.0089) / 0.6807 (control v7 0.6824) |
| train on train + valid (v11, v12) | mean -0.0012 on the 469 vs v7 / v9, seeds disagree by 0.006 |
| LSTM instead of GRU (v13) | 0.6827 vs GRU 0.6824, pairs no better |
| auxiliary heads for shifted targets (v14) | 0.6802 vs 0.6824 |
| 3 GRU layers (v16c) / hidden 192 (v20c) | 0.6815 (vs v7 0.6824) / 0.6845 |
| dropout (v16-v18) | no effect once weight decay is used; dropout alone 0.6821 vs 0.6824 |
| EMA of the weights (v20a) | 0.6857, equal to its raw weights at the end |
| peak lr 2e-3 / 5e-4 | 0.6841 / 0.6839 |
| 7 epochs instead of 5 (wd 0.05 / 0.07 / 0.1) | 0.6839 / 0.6849 / 0.6852 |
| Gaussian input noise 0.05 / 0.15 | 0.6847 / 0.6843 |
| input-column dropout 0.1 / 0.2 | 0.6835 / 0.6826 |
| longer backpropagation (2,000 / 4,000-step chunks) | 0.6829 / 0.6819 |
| auxiliary classification head (weight 0.5 / 1.0) | 0.6844 / 0.6845 |
| distillation from the submitted pair (kd 0.3 / 0.5 / 1.0) | 0.6843 / 0.6852 / 0.6858; learns much faster but finishes level |
| other row-weight powers (abs(y)^0.5 / ^1.5) | 0.6820 / 0.6834 |
| explicit volume changes / order-flow aggregates as extra inputs (v28) | 0.6835-0.6848 for all six variants |
| specialist for the coarse unit (v21) | 0.4059-0.4218 in-group vs 0.4334 for the generalist |
| third data seed for the pair | best pair 0.6869, +0.0002 over v18c+v19a |
| gradient-boosted correction of the residual (216 rolling features) | -0.0016 out of fold |
| scaling outputs by the mask probability | -0.0027 at best |
| causal smoothing or learned filters on the outputs | -0.0003 to -0.0350 |
| hand-built order-book features (mid, spread, imbalance proxies) | no gain in a linear check; no run |
| three-model ensemble | Final matched audit: best triple 0.6873498, gain +0.0006100 with a 95% interval crossing zero; INT8 callback 213.99 µs/row versus incumbent 128.79 µs/row; not submitted |

**Reading of the results.**
- The recipe has been at a ceiling near 0.686 for a single model since Oct 20. Width, depth, optimiser settings, regularisers and extra targets do not move it, and models trained this way are very similar (residual correlation 0.989-0.996, even across data seeds), so ensembling them gains little.
- The remaining error sits in the coarse-unit sequences and the large-target rows. These experiments did not establish whether that error is irreducible.

## 7. Speed engineering

**Inference** (on one pinned core of our machine; the starter baseline ran at about 93 us/row there):

| build | us/row | projected for 39.4M rows |
|---|---|---|
| GRU hidden 256, fp32 | about 203 | about 133 min |
| GRU hidden 256, int8 | 138 | too slow |
| GRU hidden 128, fp32 | 84.9 | 56 min |
| GRU hidden 128, int8 | 59.3 | 39 min (platform: 20m32s) |
| two-model int8 ensemble in one graph | 114.0 | 74.9 min local extrapolation; selected pair measured 36m07s on the public platform |

- The hidden-256 model was too slow because its float32 weights (2.9 MB) do not fit the 1 MB L2 cache of one core.
- The first int8 attempt lost 0.0030 of score, because folding the standardisation into the weights squashed the tiny price-change inputs. Keeping the standardisation as explicit graph nodes and quantising only the matrix products cut the loss to 0.000055. For the submitted pair the int8 loss is -0.000017 on the first 100 valid sequences.
- Putting both models into one graph saved almost nothing (114 vs 2 x 60 us/row).
- Early submissions suggested a platform speed advantage of about 1.89× relative to local timing. This historical ratio is not a guarantee for different exports or benchmark conditions.

**Training.**
- The cache: the parquet files are converted once to memory-mappable arrays (80 GB). Prices stay float32, because float16 would round away their changes; the other features are float16.
- Training is disk-bound. One hard disk feeds about 16-17 training steps per second, while the GPU could do 107-119. Two jobs at once each fall to about 3 steps/s.
- So `train_multi.py` trains up to six models from one stream of batches: each batch is read once and used by all models, each with its own seed, settings, optimiser and checkpoints. Three models take about the wall time of one (81-85 minutes for the v18 and v19 runs), and later runs trained six at once.
- It was checked bit for bit against the single-model trainers.

## 8. Leaderboard history

| date | submission | local valid | public leaderboard | rank |
|---|---|---|---|---|
| Oct 17 | single GRU-128, int8 | 0.679703 (full valid, official scorer) | 0.65200 | about 532 |
| Oct 18 | starter baseline (calibration) | 0.617052 | 0.59353 | |
| Oct 18 | pair v4+v5 (plain recipe) | 0.6822 | 0.65315 | 486 of 5,016 |
| Oct 19 | single v7 (mask-weighted loss) | 0.682361 (official scorer) | 0.65367 | 470 of 5,051 |
| Oct 19 | pair v7+v9 (mask-weighted) | 0.6850 | 0.65542 | 372 of 5,057 |
| Oct 20 | pair v18c+v19a (mask-weighted + weight decay) | 0.6867 | **0.65684** | **302 of 5,111** |
| Oct 23 | pair v18c+v24a | 0.6869 (FP32) | 0.65596 | not recorded |

On Oct 20 the top 10 were within 0.0023 of each other (rank 1 0.67244, rank 8 0.67017). The prize line was about +0.013 above our score.


**Final selections, November 16:** v18c+v19a, v18c+v24a and v7+v9. **Final result, November 20:** rank **412**, private score **0.66582**. The public-rank reference was 495; the final leaderboard records an 83-place improvement. The final score is the maximum across selected submissions, not an average of their private scores.

The selected v18c+v19a pair's exact FP32 validation score is **0.6867397**, distinct from the deployed INT8 public score of **0.65684** and private score of **0.66582**.

**Final experiments.** v33's multiscale projection did not establish a clear pair improvement. v34 remained an untrained streaming prototype. The fixed-checkpoint triple audit found uncertain gains and substantial runtime risk. A 32.1-minute frozen-backbone residual-head experiment completed 25,800 updates per head; B+v19a scored 0.6867321719 and C+v19a 0.6866978683, below the reproduced incumbent's 0.6867397281. No residual candidate ZIP was created.

## 9. Lessons

- **Measure before explaining.** We first blamed the local-to-leaderboard gap on checkpoint selection; the unchanged baseline also scored lower publicly, supporting a difference between evaluation periods without ruling out selection effects.
- **Know the noise.** Seed noise (about 0.0013) and subset noise (about 0.27 / sqrt(k)) decide which differences are real. Replicate anything below about 0.003, and expect only 50-80% of a valid gain on the leaderboard, less for "best of N" choices.
- **Change what the model learns, not what it outputs.** Mask weighting in the loss helped; every output post-processing we tried hurt.
- **Speed is part of the model.** Time the CPU build before spending a submission. Keep the standardisation explicit when quantising. Do not assume a shared graph is cheaper. A timeout scores zero, so keep a margin (we stay near 40 of 60 minutes).
- **Training was limited by the disk, not the GPU.** One job at a time, and many models per job.
- **Treat failed sanity checks as information.** When a check failed, we fixed the check's lookup and never loosened its tolerance.
- **Reproducibility from day one.** Fixed seeds, deterministic kernels and a log line per run. Code is committed by file name; weights and data stay out of git.

## 10. Reproducibility

`REPRODUCE.md` gives the ordered commands, seeds, expected outputs and run times, from the parquet files to the submitted zip (about 5.5 hours on one RTX 2080 Ti). Re-exporting the two checkpoints gives byte-identical ONNX and solution files. A model trained twice with the same settings repeats its logged evaluations and loss lines exactly.

**Scope and limits.** The repository contains source code and experiment documentation. Model checkpoints, competition data, generated caches and submission archives are external artifacts. Recorded re-export checks do not establish a fresh end-to-end reconstruction; full-validation INT8 scoring of the final pair was not performed. The later v33 control reproduced v18c weights bitwise. The 469-sequence diagnostic set and full validation set were reused during development.

The project-artifact backup passed SHA-256 and archive-readability checks on November 16. Raw data, generated caches and installed environments were not included. This report documents the completed project; it does not claim a prize-verification review or a newly granted software license.

