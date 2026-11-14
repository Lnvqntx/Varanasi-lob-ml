# Varanasi final-push research: published financial ML competition solutions
Research date: 15 November 2026, India time. Prepared for Team Varanasi.

## Decision
Do not launch v34 merely because its streaming implementation now runs faster. It is still untrained; its benchmark establishes feasibility, not predictive value. First audit the best already-trained blends. If there is time for one new learning experiment, test a small nonlinear residual prediction head on the existing GRU, preserving its inputs and recurrent backbone.

This is a broad search across roughly thirty competition/event families, not a claim to have exhaustively searched every contest or recovered every winning system. Evidence below distinguishes original high-ranking write-ups, self-reported results, reproductions, and events where no usable winning method was found. Public code was inspected through available repository pages and write-ups; it was not executed. Some Kaggle text was available through search indexing when the direct page did not render. Placement and methods are attributed to the linked sources; no result in another contest predicts our improvement.

## Our actual constraints and evidence
The supplied rules require causal, deterministic processing of every row, including warm-up rows, with state reset between sequences. Only supplied train/valid data may train the model. Test scoring mask is hidden. No GPU at inference: one CPU core, 16 GB, 60 minutes. There are 39.4 million test rows.

The current v18c + v19a pair has full-validation scored WP about 0.6867397 and public score 0.65684. The supplied latest leaderboard screenshot shows 482/5,832, not a final rank. Its recorded platform runtime is 36m07s.

Validation scores cannot be compared numerically with other competitions' scores. Even within our project, the v18c + v24a pair improved validation slightly but scored worse publicly. The 469-sequence diagnostic subset has already been used for selection, so it is not an untouched confirmation set.

v33 C recovered some of the projection-only model's loss but did not beat the original GRU. Its two tested pairs did not demonstrate a clear gain over the incumbent. v34's optimized circular-cache streaming prototype is useful engineering, but no full-trained v34 performance exists.

A repository result worth revisiting is ensemble_check11_results.txt: v18c + v24a + v13 LSTM was approximately 0.6873 on scored validation, roughly +0.0006 over the incumbent; v18c + v19a + v13 was roughly 0.6872. These are rounded, previously selected exploratory results, not new verified gains. A three-model runtime extrapolation from 36m07s is about 54 minutes if cost scales by 1.5; mixed architectures, callback overhead and platform variability make that estimate unreliable. Runtime must be measured before proposing submission.

## Closest precedents: Wunder Fund

### RNN Challenge 2025
The archived contest specification describes next-state prediction, a different feature layout, shorter sequences and a different metric. The archived leaderboard names insuperabilehart first. I did not locate a verifiable original public technical account or repository from that winner.

Sources:
- [Archived specification and results](https://github.com/msrishav-28/wunder-fund-market-state-forecasting/blob/main/website.md)
- [Historical official route](https://wundernn.io/wunder_challenge)

Two public repositories are useful only with correct labels:
- [msrishav-28 reconstruction](https://github.com/msrishav-28/wunder-fund-market-state-forecasting) explicitly presents a portfolio reconstruction. Its CV score is not proof of a winning competition entry or equivalent evaluation.
- [zitrasa 2025 experiments](https://github.com/zitrasa/wunderfund-challenge-2025) contains recurrent/convolutional experiments, but no verified winning placement was established.

### Predictorium 2025–2026
[Official event page](https://wundernn.io/predictorium). This preceding task is closer: recurrent streaming prediction with weighted correlation. Its input/output layout, clipping, test size and runtime allowance still differ. I found no sufficiently verified original first-place technical solution to present as the winner's method.

[Hemanth Reddy's repository](https://github.com/hemanthreddyllm/lob-price-prediction) self-reports 53rd/4,900, not first place. It describes ten GRUs across folds/seeds, engineered EMA features, a convolutional feature mixer and stateful ONNX. Those are plausible ideas, but much of the ensemble/preprocessing territory is already explored here. Its reported approximately 1.6 ms/row inference belongs to the older task: our budget averages about 91 microseconds per test row before overhead. Copying that system would exceed our budget by a large margin.

I excluded detailed public participant solutions for the currently running Connectome competition from strategy extraction because the supplied rules prohibit detailed solution sharing before the contest ends. Historical public solutions and independent technical ideas are the research basis.

## Original high-ranking solutions: what transfers

### Jane Street Market Prediction 2021 — first-place team member
[Original Cats Trading / Yirun Zhang write-up](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)

The supervised autoencoder learned a useful representation jointly with prediction; its representation was combined with original features. Noise, careful folds and multiple prediction targets supported the model. The lesson is representation learning without discarding useful raw information. Our v33 bottleneck replaced the original GRU input and is not a test of every representation-learning approach. A faithful supervised autoencoder would require substantial new training and validation, so it is a later option, not the fastest finish.

### Jane Street Real-Time Market Data Forecasting — eighth place
[Original eighth-place write-up](https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution)
[Author repository](https://github.com/evgeniavolkova/kagglejanestreet)
[Repository solution notes](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md)

This solution used recurrent models with nonlinear dense prediction layers, seed ensembles, temporal statistics and auxiliary outputs. The author reports useful local gains from architectural changes. Online updates contributed substantially, but that mechanism depended on feedback unavailable here. The transferable hypothesis is a nonlinear readout of recurrent state. It is distinct from projecting inputs before a GRU. It is not evidence that a particular head width, dropout or gain will transfer.

I located the competition's first-place leaderboard result but not a sufficiently verified original first-place technical account. This is an eighth-place precedent, not a mislabeled winning recipe.

### Optiver Trading at the Close — first place
[Original hyd write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)

The reported final blend combined CatBoost, GRU and Transformer with weights 0.5/0.3/0.2. Shared engineered features and complementary model errors mattered. Online updates and cross-stock adjustments also helped there. We cannot reproduce unavailable feedback or synchronized stock batches from shuffled independent sequences. The useful lesson is to measure whether models help a blend: a weaker standalone model can still contribute. This supports rechecking our saved LSTM blend, not blindly adding architectures.

### Optiver Realized Volatility — first place
[Original nyanp nearest-neighbor solution](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors)

Nearest-neighbor features across stocks/time buckets and a heterogeneous ensemble were central. Much of the feature gain used dataset structure absent here. Recovered chronology and cross-sectional neighbors are not permission to use arbitrary seq_ix as time or connect independent sequences. Only causal within-sequence analogues could be considered.

[PKSHA fourth-place account](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/pksha-life-is-volatile-tentative-3rd-place-solutio) also used normalization and learned features. Its title says tentative third; the indexed write-up metadata records fourth. Its complete-window representation learning is not directly usable in our row-by-row inference setting.

### DRW Crypto Market Prediction 2025 — first place
[Original first-place write-up](https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st)

A relatively simple MLP was strong, with boosted trees contributing diversity. Feature selection and representation experiments were important; architecture size alone was not the explanation. This makes a compact nonlinear component worth considering. Its mixed-loss idea is not new for our project: v32 already investigated correlation-based loss. The winner's feature selection does not establish that removing our feature groups will help.

### G-Research Crypto Forecasting — first three places confirmed by organiser
[Official competition recap](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)

The organiser identifies MemeLord Capital, Nathaniel Maddux and GABA as the first three teams and reports LightGBM use across the top solutions. Feature engineering was a common theme. This establishes their model family, not that trees universally beat RNNs. Our residual LightGBM experiment already failed; repeating it without a specific new causal feature or a cleaner training design is low priority.

[Tom Forbes' final thirteenth-place write-up](https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/tom-forbes-13th-place-final-1st-place-6-weeks-in-f) is also useful evidence that leading a public board temporarily is not equivalent to winning the final evaluation.

### Ubiquant Market Prediction — first place
[Original K.I.Y write-up](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy)

LightGBM and TabNet, cross-sectional aggregates and choices about recent training history formed the approach. Known time groups and supplemental data supported choices that are unavailable here. Our shuffled sequence identifiers do not reveal recency. The useful principle is to match validation and deployment structure; the exact preprocessing is not portable.

### M6 financial forecasting — first in forecasting
[Winner Dan Weitzenfeld's account](https://danielweitzenfeld.github.io/passtheroc/)
[Competition methods repository](https://github.com/Mcompetitions/M6-methods)

A probabilistic factor model emphasized volatility, uncertainty and restrained expected-return estimates. It won a distributional forecasting task over assets and horizons unlike our two-target weighted correlation problem. Hierarchical shrinkage is conceptually useful, but copying near-zero return predictions would not solve our objective.

## Wider search coverage and evidence gaps
These were searched to avoid limiting the review to famous Kaggle names. They are not all direct comparators, and a winner's repository without a readable methodological account is not enough to recommend its strategy.

| Event/family | Public evidence located | Use for this project |
|---|---|---|
| Wunder RNN 2025 | Archived task/results; reconstructions | Historical comparison; no verified winner method found |
| Wunder Predictorium | Official task; self-reported top-53 repo | Close predecessor; major runtime differences |
| Jane Street 2021 | Original first-place member write-up | Supervised representation plus original features |
| Jane Street Real-Time | Original eighth-place write-up/repo | Nonlinear recurrent readout; feedback limitation |
| Optiver Trading at Close | Original first-place write-up | Measured model complementarity |
| Optiver Realized Volatility | Original first/fourth write-ups | Feature design; cross-sectional methods constrained |
| DRW Crypto | Original first-place write-up | Simple nonlinear models and careful features |
| G-Research Crypto | Organiser confirms top three | Trees/features; avoid repeating failed baseline |
| Ubiquant | Original first-place write-up | Validation structure; chronology unavailable here |
| M6 | Winner account; methods archive | Probabilistic forecasting; different objective |
| MITSUI Commodity | Original third-place write-up | Target-specific stable features; not LOB |
| Hull Tactical | Original fourth-place account | Allocation/risk objective differs |
| Two Sigma Financial Modeling | Official task; historical solution discussions | Winner implementation not verified in readable primary material |
| Two Sigma Using News | Official task/data descriptions | Daily news/market task; no verified winning recipe recovered |
| Winton Stock Market Challenge | Official task; late-entry repository | Late-entry claimed rank is not actual contest placement |
| JPX Kaggle Stock Prediction | Official task/discussions | No sufficiently verified winner method recovered |
| JPX/SIGNATE fundamentals | Author's first-place repository | Different daily/fundamental task; no invented method attribution |
| QRT/ENS Stock Return Prediction | Official task; participant repo | Participant top-15% claim is not a winner |
| QRT DurHack | Sponsor-prize project account | Limited technical/evaluation detail |
| Citadel Data Open | First-place participant repositories | Some winning tasks concern environment rather than markets |
| IMC Prosperity 2 | Original second-place team repository | Execution/inventory simulation, not supervised WP |
| IMC Prosperity 4 | First-place-labeled repository | Placeholder; no usable method verified |
| Ready Trader Go | Organiser winner announcement | No usable original winning code verified |
| Numerai | Official model/neutralization documentation | Era/cross-sectional setting differs; docs are not a winner |
| Crunch DataCrunch #2 | Official anonymized equity task | No verified winning implementation recovered |
| Crunch Mid+One | Official one-step forecasting task | Relevant streaming comparison; no verified winner recipe |
| ADIA Lab Causality | Organiser winning-method case study | Causal graph discovery, not direct return prediction |
| ADIA Structural Break | Official winners and benchmark | Regime detection; no verified winning code recovered |
| BattleFin Big Data Combine | Historical task/participant material | No usable primary top solution verified |
| Algorithmic Trading Challenge | Historical contest references | No usable primary winning method verified |

Additional primary links:
- [MITSUI third place](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/3-rd-place-solution-directional-trends-over-vola)
- [Hull Tactical fourth place](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te)
- [JPX/SIGNATE first-place author repository](https://github.com/masahiro-mochizuki/signate-fundamentals-challange-1st-place)
- [QRT/ENS task](https://challengedata.ens.fr/challenges/23)
- [QRT DurHack project](https://devpost.com/software/qrt-stock-prediction)
- [Citadel SoCal 2018 winning team repository](https://github.com/Jeremy123W/Citadel-SoCal-Datathon-2018-1st-Place-Winners)
- [IMC Prosperity 2 second-place repository](https://github.com/ericcccsliu/imc-prosperity-2)
- [IMC Prosperity 4 repository](https://github.com/Seven-Deuce-Capital/imc-prosperity-4-seven-deuce-capital)
- [Numerai official model documentation](https://docs.numer.ai/numerai-tournament/models)
- [DataCrunch #2](https://docs.crunchdao.com/competitions/competitions/datacrunch-2)
- [Mid+One](https://docs.crunchdao.com/competitions/rallies/mid+one)
- [ADIA causality case study](https://www.crunchdao.com/case-studies/adia-lab-causality)
- [ADIA structural-break event](https://www.adialab.ae/structural-break-challenge)

LinkedIn helped locate some authors and organiser announcements, but an author's repeated placement claim is not independent verification. Reproductions, medal repositories, post-deadline entries and empty first-place repositories were not upgraded into winning evidence.

## Ranked actions

### 1. Existing checkpoints: fastest credible finish
Reproduce the incumbent and the two predetermined LSTM triples in one shared full-validation pass. Use the official clipping/weighting and report each target, scored rows, all required rows and sequence-level diagnostics. Use paired sequence bootstrap intervals, explicitly acknowledging prior selection and training uncertainty. Do not launch a broad blend-weight search against the familiar validation set.

Verify the exact incumbent ZIP and inference settings, then export only the strongest eligible blend. Quantization may change the ranking; compare actual exported predictions, not only PyTorch outputs. Benchmark full callbacks with resets, warm-up, startup, memory and enough consecutive sequences for stable timing. Three models close to 60 minutes need further optimization or exclusion.

The previous +0.0015 improvement gate can be revisited for the user's smaller rank-improvement goal, but +0.0006 remains weak evidence. If a modest exploratory submission is built, say so honestly; retain the established incumbent and check the final-selection rules.

### 2. One new learning hypothesis: nonlinear residual GRU readout
Keep the original 168-feature preprocessing and existing GRU backbone. Start from a saved training checkpoint and add a small nonlinear residual readout to the hidden state:
prediction = existing_linear_head(h) + small_MLP(h).
Initialize the residual's final layer to zero so initial predictions match the checkpoint. First test a frozen-backbone version against a matched head-training control; only then consider full fine-tuning if time and evidence justify it.

This is an inference from the Jane Street precedent, not a published winner recipe for Connectome. It tests a hypothesis not covered by v33 input projection. Keep identical training data, masks, sequence order, budget and validation accounting. Never train on validation targets during this comparison. Check causal streaming equivalence, state reset, deterministic execution and CPU export. Measure throughput before giving a wall-clock estimate.

Avoid storing every training hidden state in RAM. Read actual container memory limits: host free memory is not the container allowance. Shared-disk throughput has already limited parallel jobs in this project.

### 3. v34 remains a secondary option
A trained temporal CNN might add diversity, but implementation readiness and short-stream speed do not establish predictive performance or platform runtime. Full training plus validation/export could consume hours. Do not launch it automatically while the user is trying to finish quickly.

### Deprioritized ideas
- More seeds/models without demonstrating useful marginal improvement per CPU cost.
- Cross-sectional neutralization, time-ID reconstruction or online label updates that require unavailable information.
- Another mask-only optimization that could overfit the visible scored subset.
- Repeating Pearson loss, auxiliary classification, simple input EMA, feature deletion, target scaling or residual trees without a new hypothesis: related variants already failed here.
- Copied financial indicators whose required raw semantics are not established.
- Assuming the public/private gap is a calibration problem or assuming an old contest's “oracle” is our score ceiling.

## Operational finish
Research does not guarantee a climb. Preserve the known working submission, record hashes/settings for any candidate, confirm the official deadline and final-selection mechanism, then select using both evidence and runtime feasibility. No training, repository edits, submission or push was performed during this research task.

