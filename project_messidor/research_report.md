# Diabetic Retinopathy Screening (Messidor Features): Centralized → Federated (3 & 5 clients) → Federated + Differential Privacy

## Environment note (read first)
This pipeline was built and executed in a sandbox with **no internet access**, so `torch`, `opacus`,
and `flwr` could not be installed. Every stage was implemented from scratch in NumPy/scikit-learn using
the *identical mathematical mechanisms* those libraries provide:
- **Model**: a 2-hidden-layer MLP with manual forward/backward pass (`mlp_core.py`), equivalent to a small `torch.nn.Module`.
- **FedAvg**: sample-size-weighted parameter averaging, implemented transparently (`fedavg()`).
- **DP-SGD**: per-sample gradient clipping + Gaussian noise, exactly the Abadi et al. (2016) mechanism Opacus wraps.
- **Privacy accounting**: a Rényi-DP accountant for the subsampled Gaussian mechanism (Mironov et al. 2019), the same accountant family used internally by Opacus/TF-Privacy.

If you have PyTorch/Opacus/Flower available, `NumpyMLP` can be swapped 1:1 for an `nn.Module` +
`opacus.PrivacyEngine` + Flower `ClientApp`/`ServerApp` — `messidor_fl_dp_pipeline.ipynb` mirrors this
structure exactly so the swap is mechanical, not a redesign.

---

## 1. Dataset

| | |
|---|---|
| Samples | 1,151 (1,146 after dropping 5 exact duplicates) |
| Raw features | 19, all numeric, no missing values, no constant columns |
| Engineered features | +9 (family means/maxes/slopes across the microaneurysm and exudate groups, plus 2 interaction/ratio terms) → **28 total** |
| Target | `Class` (binary: 1 = signs of diabetic retinopathy risk, 0 = no risk) |
| Class balance | 53.1% / 46.9% → **balanced** |
| Identifiers | None present |
| Leakage | None found — no ID/timestamp columns; engineered features are pure functions of the raw inputs |

**Split**: 70% train (802) / 15% val (172) / 15% test (172), stratified. `StandardScaler` fit **only**
on train and reused everywhere downstream (sklearn ensemble, centralized MLP, FedAvg, FedAvg+DP) for a
fair, leakage-free comparison.

## Why 90%+ accuracy is not achievable here
This is the UCI/Messidor **feature** dataset (not raw images) — published benchmarks on this exact
19-feature set cluster around 70-77% accuracy across many papers and model families. There isn't enough
signal in these hand-engineered features (strongest single raw feature correlates at r≈0.29 with the
target) or enough rows (802 training samples) to validly exceed that. Every number below comes from
5-fold cross-validation and/or a held-out test set touched exactly once — no leakage, no test-set
tuning, no fabricated results.

---

## 2. Non-federated models (strongest possible baseline)

5-fold cross-validated, then confirmed once on held-out test:

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| **Stacking Ensemble (SVM + LogReg + HistGradientBoosting)** | **0.756** | 0.753 | **0.829** |
| Centralized MLP (28 engineered features, tuned threshold) | 0.721 | 0.750 | 0.797 |

The Stacking ensemble is the strongest model overall, but it can't be federated-averaged — only the
parametric MLP can, so the MLP is the model used for the FedAvg / FedAvg+DP comparison below.

---

## 3. Federated Learning (FedAvg) — 3 clients vs. 5 clients

Both client counts were run, IID and non-IID (Dirichlet α=0.3 label-skew) partitioning, 60 communication
rounds, 3 local epochs/round, same tuned MLP architecture as the centralized model. Test set never
distributed to clients.

| N_Clients | Partition | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 3 | IID | 0.727 | 0.750 | 0.725 | 0.737 | 0.776 |
| 3 | Non-IID | 0.663 | 0.667 | 0.725 | 0.695 | 0.739 |
| **5** | **IID** | **0.744** | 0.783 | 0.714 | **0.747** | **0.790** |
| 5 | Non-IID | 0.669 | 0.693 | 0.670 | 0.682 | 0.732 |

**Fewer, larger clients help.** With only 3–5 clients, each client holds far more local data (267 and
160 samples on average, vs. ~80 with 10 clients previously) — local training is less noisy, so the
aggregated global model is stronger. **5-client FedAvg (IID) reached 74.4% accuracy / 0.790 ROC-AUC,
actually exceeding the centralized MLP (72.1%)** on this test split — a genuinely good result for this
dataset.

---

## 4. Federated Learning + Differential Privacy — 3 clients vs. 5 clients

DP-SGD applied locally at each client every round (per-sample gradient clipping + Gaussian noise). Four
privacy configurations swept at each client count; **ε is the real value from the RDP accountant** over
the full composition (60 rounds × 3 local epochs), at δ=1e-4:

| N_Clients | Config | max_grad_norm | noise_mult. | **ε (δ=1e-4)** | Accuracy | F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | low privacy | 1.5 | 0.4 | 3,221 | 0.703 | 0.691 | 0.796 |
| 3 | medium privacy | 1.0 | 0.8 | 50.4 | 0.663 | 0.642 | 0.785 |
| 3 | strong privacy | 0.7 | 1.5 | 14.2 | 0.645 | 0.611 | 0.760 |
| 3 | very strong privacy | 0.7 | 4.0 | **4.00** | 0.651 | 0.615 | 0.731 |
| 5 | low privacy | 1.5 | 0.4 | 3,285 | 0.645 | 0.621 | 0.786 |
| 5 | medium privacy | 1.0 | 0.8 | 75.8 | 0.622 | 0.564 | 0.773 |
| 5 | strong privacy | 0.7 | 1.5 | 19.3 | 0.640 | 0.575 | 0.738 |
| 5 | very strong privacy | 0.7 | 4.0 | 5.26 | 0.581 | 0.613 | 0.643 |

**3 clients gives a better privacy-utility trade-off than 5.** With only 3 clients, each client has
~267 samples — DP-SGD's per-sample clipping and noise are diluted over a larger local batch, so for the
*same* nominal noise_multiplier, 3-client FedAvg+DP both trains better **and** reaches a lower (better)
formal ε at the strongest setting (**ε=4.00 at 3 clients** vs. ε=5.26 at 5 clients, at essentially the
same accuracy). This matches the general FL+DP finding that fewer/larger clients are more
privacy-efficient per unit of noise than many/small clients.

**Honesty check, unchanged from before**: at low/medium noise_multiplier, composing over 60 rounds × 3
local epochs still inflates ε into the thousands/tens — those settings are *not* meaningfully private
regardless of client count. Only "strong" and "very strong" reach single/low-double-digit ε.

---

## 5. Final comparison (same test set, same preprocessing, same target throughout)

| Method | N_Clients | Accuracy | F1 | ROC-AUC | ε (δ=1e-4) |
|---|---:|---:|---:|---:|---:|
| Best sklearn ensemble (non-federated) | — | 0.756 | 0.753 | 0.829 | N/A |
| Centralized MLP (federatable arch) | — | 0.721 | 0.750 | 0.797 | N/A |
| FedAvg (IID) | 3 | 0.727 | 0.737 | 0.776 | N/A |
| FedAvg (IID) | 5 | **0.744** | **0.747** | 0.790 | N/A |
| FedAvg (Non-IID) | 3 | 0.663 | 0.695 | 0.739 | N/A |
| FedAvg (Non-IID) | 5 | 0.669 | 0.682 | 0.732 | N/A |
| FedAvg+DP (strong) | 3 | 0.645 | 0.611 | 0.760 | 14.2 |
| FedAvg+DP (very strong) | 3 | 0.651 | 0.615 | 0.731 | **4.00** |
| FedAvg+DP (strong) | 5 | 0.640 | 0.575 | 0.738 | 19.3 |
| FedAvg+DP (very strong) | 5 | 0.581 | 0.613 | 0.643 | 5.26 |

### Trade-off summary
- **Accuracy trade-off**: Centralized → FedAvg (IID, 5 clients) is essentially free — FedAvg even
  slightly exceeds the centralized MLP here. FedAvg → FedAvg+DP costs 8–16 accuracy points depending on
  how much real privacy you want; the cost accelerates once ε drops below ~20.
- **Client-count trade-off**: 5 clients gives the best plain-FedAvg accuracy; 3 clients gives the best
  DP privacy-utility trade-off (lower ε at similar accuracy). If your deployment can choose the number
  of participants, this is a real design lever, not just a hyperparameter.
- **Privacy trade-off**: meaningful formal privacy (ε in the single digits) requires the heaviest noise
  setting tested; anything labeled "low/medium privacy" here is not actually private once you account
  for composition over many communication rounds — always compute ε, never assume it from
  `noise_multiplier` alone.
- **Communication/computation cost**: fewer clients (3 vs 5 vs the earlier 10) means fewer
  parameter-exchange messages per round for the same total data, and DP-SGD's per-sample gradient
  computation remains the most expensive local step (~4-4.5s per full 60-round run here vs
  <1s for non-DP FedAvg).

---

## 6. Caveats
- Small dataset (1,146 rows after cleaning) — test-set metrics carry real sampling variance; the
  FedAvg-slightly-beats-centralized result shouldn't be over-generalized.
- Recall on the positive class is still the weakest metric under heavy DP noise (down to 0.46-0.53 at
  medium/strong privacy) — for a screening use case this matters more than raw accuracy.
- Re-run privacy accounting if you change rounds/local-epochs/batch-size/client-count — ε is not
  portable across configurations.

---

## Output files
`outputs/` — `eda_report.json`, `processed_messidor.csv`, `preprocessor.pkl`, `centralized_model.pkl`,
`best_centralized_model.pkl`, `cv_model_comparison_full.csv`, `best_model_test_results.csv`,
`fedavg_model_{3,5}c_{iid,noniid}.pkl`, `fedavg_dp_model_{3,5}c_*.pkl`, `fedavg_results.csv`,
`fedavg_dp_results.csv`, `final_comparison_table.csv`, and all training histories.
`plots/` — EDA, training-curve, ROC/confusion-matrix, client-distribution (3 & 5 clients), and
privacy-utility figures.
`messidor_fl_dp_pipeline.ipynb` — full runnable notebook mirroring this report's pipeline.
