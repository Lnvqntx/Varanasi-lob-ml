# Final bounded frozen-backbone residual attempt — November 14, 2026

**Completed; decision: INCUMBENT. Candidate ZIP: none. This attempt is stopped; no additional experiment is proposed.**

Original clock start **2026-11-14T19:52:18+00:00**, including the interrupted inspection; completed **2026-11-14T20:24:23.784714+00:00** (32.1 minutes). Original hard deadline 2026-11-14T20:52:18+00:00; training cutoff 2026-11-14T20:37:18+00:00. The clock was not restarted. Training ended 20:20:53 UTC, before the 15-minute evaluation/reporting reserve. Full validation finished 20:22:25 UTC.

## Method and controls

Loaded the exact v18c final checkpoint with its original preprocessing statistics. The GRU and original Linear(128,2) prediction head were frozen/eval throughout. A is untouched. B adds a zero-initialized Linear(128,2) residual. C adds Linear(128,32)→ReLU→Linear(32,2), with its output layer zero-initialized. Heads use an isolated initialization RNG. One frozen-GRU chunk forward is shared by B/C; only current 1000-row hidden tensors and recurrent state are retained, never the full training hidden-state dataset.

Original seed 0 split/order: 10,307 training sequences, 300 train holdout excluded; original stats frozen; existing train_maskp.npy; mean probability .1041544597041231; alpha .5; clipped weighted MSE from train_v2m.loss_fn; AdamW lr .001, wd .05, norm clipping 1; batch 16, chunk 1000. Mask weights and update counts are identical for B/C. No validation targets were opened until after final head checkpoints were saved. No intermediate validation, early stopping, weight search or hyperparameter search.

Disposable 100-update pilot: 19.00 shared updates/s. Restarted both heads/optimizers and the established main data RNG. The single fixed budget was two epochs, **25,800 updates/head**, with cosine decay across that budget, chosen using pilot/time only. Main training **1415.6s (23.59min)** at **18.23 updates/s**, GPU peak **1.038GiB**. No interruption by the deadline was needed.

Checks: zero-output initialization matches A bitwise; gradients finite/nonzero where expected. C first-layer gradient is exactly zero initially because its output layer is zero, then becomes nonzero after one update. Future perturbations leave past predictions exact; whole/chunked/single-row output error ≤5.97e−7; reset/deterministic checks exact. All backbone weights remained bitwise unchanged and backbone gradients were absent. States update through warm-up and carry across chunks; reset only at sequence boundaries.

## Final full-validation results

One shared pass across all 1,873 sequences, using each backbone's own saved statistics. Average raw outputs before clipping; official global weighted Pearson, scored is_scored & step>=99, all-required step>=99. Sufficient statistics agree with GlobalAccumulator to 1e−12. Full pass 91.62s. A and the incumbent reproduce their historical values within 1e−5.

| Model | Scored WP | t0 | t1 | All-required WP | Δ vs incumbent |
|---|---:|---:|---:|---:|---:|
| A | 0.6857267071 | 0.6815190706 | 0.6899343436 | 0.5138373995 | -0.0010130211 |
| B | 0.6853658549 | 0.6811209042 | 0.6896108057 | 0.5130313474 | -0.0013738732 |
| C | 0.6854536465 | 0.6812777500 | 0.6896295431 | 0.5130385663 | -0.0012860816 |
| A+v19a | 0.6867397281 | 0.6824938641 | 0.6909855922 | 0.5164549890 | +0.0000000000 |
| B+v19a | 0.6867321719 | 0.6824759263 | 0.6909884175 | 0.5163323561 | -0.0000075562 |
| C+v19a | 0.6866978683 | 0.6824578125 | 0.6909379241 | 0.5162292958 | -0.0000418598 |

The best corrected pair, B+v19a, is **−.0000075562** below the incumbent; C+v19a is **−.0000418598** below. Standalone B/C are both below A. Small differences are not evidence of generalization; the full validation and existing 469 subset have been repeatedly reused and are not untouched test data. This one seed/two-epoch experiment does not prove that all residual approaches fail; it supplies no winning candidate under the specified attempt.

## Runtime and packaging gate

The unchanged incumbent's prior actual matched callback benchmark is **128.79µs/row**, core 31/thread 1, in audit_results/final_stream_20261114_1945/verified_results.json. That is existing incumbent evidence, not a new measurement or certification of residual candidates. No candidate CPU/platform runtime is claimed. The accuracy gate already fails, so export checks, full quantized-score verification and packaging were not pursued. A prepared exporter helper is saved but was **not executed**. No ZIP was produced. A partial check has not been labeled submission readiness.

## Evidence and preservation

- Fresh code: bounded_residual_20261114.py; prepared-but-unused bounded_residual_export_check.py.
- bounded_results/residual_20261114_1952/: B_final.pt, C_final.pt, checks.json, pilot.json, plan.json, training.json (curve), results.json, sequence_stats.npz, decision.json, source snapshot/fingerprints.
- bounded_results/residual_20261114_1952.run.log and residual_20261114_1952_launch.json: detached nohup execution and original deadline record. The empty .nohup.out is retained from a background-launch attempt that did not start a process; the fresh .run.log is the actual run log.
- No existing checkpoint, submission or cache was overwritten. No other users' jobs were interrupted. No new architectures, v34, leaderboard submission, deletion or push occurred. No further run is scheduled.
