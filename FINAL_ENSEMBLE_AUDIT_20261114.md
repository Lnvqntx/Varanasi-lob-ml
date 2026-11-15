# Final existing-checkpoint ensemble audit — November 14, 2026

**Recommendation: retain v18c + v19a as the incumbent.** The strongest fixed triple adds only .0006100 scored WP, its paired interval includes zero, and its exported callback is 1.66× slower. It misses the existing +.0015 build rule and has substantial runtime risk. No new training, weight search, v34 run, leaderboard submission or push occurred. All previous artifacts are preserved.

## Full validation: fixed equal-weight blends

| Blend | Scored WP | t0 | t1 | All-required WP | Diagnostic 469 WP | Scored improvement | Paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| v18c+v19a | 0.6867397 | 0.6824939 | 0.6909856 | 0.5164550 | 0.6937848 | +0.0000000 | [+0.0000000, +0.0000000] |
| v18c+v24a+v13 | 0.6873498 | 0.6833031 | 0.6913965 | 0.5176482 | 0.6941013 | +0.0006100 | [-0.0000344, +0.0013085] |
| v18c+v19a+v13 | 0.6871564 | 0.6829906 | 0.6913221 | 0.5176119 | 0.6939772 | +0.0004166 | [-0.0000916, +0.0009430] |

One shared pass across all 1,873 valid sequences (86.2s, GPU0 peak .84GiB). Each model uses its own saved 168-input preprocessing statistics. Hidden states, including both LSTM h/c, are reset per independent sequence and carried through warm-up and 1,000-row chunks. Raw float32 predictions are averaged **before** clipping to [-2,2]. Scored rows are is_scored AND step>=99; all-required rows are step>=99. Official GlobalAccumulator matches merged float64 sufficient statistics to <1e-12. Incumbent .6867397281 is within 2.82e−8 of .6867397. All four standalone models match their checkpoint scores; all seven single/blend metrics reproduce ensemble_check11 saved statistics within 2.87e−12.

Intervals use 2,000 paired whole-sequence bootstrap resamples with replacement, fixed seed 20261114; global WP is recomputed after merging each resample. They do not account for training randomness or prior model selection. The reused 469 sequences (array index % 4 == 0) are diagnostic, not untouched validation. Both candidate intervals include zero.

## Exact saved checkpoints

- v18c, step 64500: `/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt`; SHA-256 `0742d207fd8e4961efea773860b6465e5639193351d53cd4498f7252f449a2db`.
- v19a, step 64500: `/home/jovyan/work/checkpoints/train_v19a_wd10_ds1_final.pt`; SHA-256 `43457713f232780e998039b3c441cd094b76b3e0b2c817fec43b19d66ffd35d7`.
- v24a, step 64500: `/home/jovyan/work/checkpoints/train_v24a_ds2_wd05_final.pt`; SHA-256 `d9e661085ad63f749b37538cc432ffaeaa9e02f2ef8cc3aeca854496735260e6`.
- v13, step 64500: `/home/jovyan/work/checkpoints/train_v13_lstm_maskw_a05_h128_e5_seed0_best.pt`; SHA-256 `a096c5afb7eba123ed199c149af4a4f5ed3006f5e7853141fc023b1fd9c25894`.

Paths/loaders are those in ensemble_check11.py, ensemble_check6.load_lstm (train_v4m.LSTMModel) and ensemble_check7.load_gru (train_v1.GRUModel). Each _final.pt was used exactly; v13 _best.pt is its final step 64,500. Checkpoint hashes were rechecked after the audit.

## Actual exported strongest triple: v18c + v24a + v13

The original export_ens.py supports two GRUs only. export_v4.py supports a standalone LSTM. There was no ready mixed three-model exporter/adapter. audit_mixed_stream.py composes the existing validated explicit cells from export_v4.build(cell="matmul", fold=False), retains separate explicit standardization, averages the three raw heads, then clips once. The unchanged generic solution_v4.py adapter carries all eight GRU/LSTM hidden/cell tensors in preallocated ping-pong buffers and performs one ORT call per row. This produced audit-only fp32/int8 ONNX graphs, not a submission zip.

Full callback timing: pinned core 31, one ORT/BLAS thread, GPU hidden, alternate model order, three repeats of two complete 20,000-row real sequences (120,000 callbacks/model). Includes preprocessing/input checks, sequence-reset/warm-up calls, recurrent updates and returned prediction copy. No warm-up callbacks excluded.

| Export | Mean callback µs/row | Runtime here,39.4M rows | With25% +5min margin | Isolated RSS / address-space high-water |
|---|---:|---:|---:|---:|
| Strongest triple fp32 | 329.33 | 216.3 min | 275.3 min | 55.3 MiB |
| Strongest triple int8 | 213.99 | 140.5 min | 180.7 min | 52.4 MiB |
| Deployed incumbent int8 | 128.79 | 84.6 min | 110.7 min | 51.6 MiB |

Triple fp32 vs batched PyTorch: maximum error 2.5e-06, mean 2.33e-07; int8 vs fp32: max 0.055631, mean 0.005052. Sequence resets and repeated outputs are bitwise deterministic. Callback buffers total 9,000 bytes for the triple. Graph sizes: fp32 3,420,792 bytes, int8 884,482 bytes. Memory was measured in separate no-PyTorch workers after 2,000 fixed-shape callbacks using /proc VmRSS/VmHWM. resource.ru_maxrss inherited the large export parent's peak in the first probe; those values are explicitly superseded by memory_verified.json and are not used here.

The repository's historical 1.89× platform-speed calibration would project the triple int8 to **74.4min callback-only,97.9min with the stated margin**, versus incumbent 44.7/60.9min. That historical factor is only a scenario, not a current platform guarantee. Direct local extrapolation also exceeds 60 min even for the already accepted incumbent, illustrating why this machine cannot certify platform runtime. The triple still has appreciably greater risk under both views. No actual full-test/platform runtime was measured.

Validation table scores are fp32 checkpoint results. The exported int8 triple has **not** been scored across all 1,873 sequences; its quantization loss could erase a small local gain. Export fidelity/timing covered two whole sequences only. Prior broad model selection and the reused subset further limit claims. The incumbent has existing public-score evidence; these triples do not acquire any new leaderboard evidence from this audit.

## Fresh evidence

- `audit_results/final_20261114_1940/`: results.json/txt, sequence_stats.npz, bootstrap.npz, source/checkpoint fingerprints, execution record/log and source snapshot.
- `audit_results/final_stream_20261114_1945/verified_results.json`: final streaming timing, corrected memory, fidelity and runtime scenarios.
- `audit_results/final_stream_20261114_1945/export/{fp32,int8}/`: actual audit-only graph, matching stats indices and generic callback adapter.
- `audit_results/final_stream_20261114_1945/memory_verified.json`: isolated process resident/high-water memory evidence.
- New audit scripts: final_ensemble_audit.py, audit_mixed_stream.py, bench_final_audit.py, audit_stream_memory.py. Existing trainers/exporters and checkpoints remain untouched.
