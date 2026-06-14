# T1DUJxUoM: Causal Plausibility vs. Forecast Accuracy in T1D Digital Twins

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-orange.svg)](https://github.com/)

**Authors:**
- **Ameen Banjar** (University of Jeddah, Saudi Arabia)
- **Ahmed Ibrahim** (University of Jeddah, Saudi Arabia)
- **Simon Harper** (University of Manchester, UK)

---

## 📌 Overview

This repository is the reproducibility companion for the retrained (v3) pipeline behind our paper on **patient-specific, data-driven Type 1 Diabetes digital twins**, evaluated with a strict **60/20/20 chronological train/validation/test split** and a held-out **two-hour autoregressive rollout** (24 steps at 5-minute resolution).

It supersedes the earlier `T1DUJxUoM` v3.0.0 release (Zenodo DOI [10.5281/zenodo.18434352](https://doi.org/10.5281/zenodo.18434352)), which used a different split protocol and reported a different (smaller, biased) set of results. **The numbers, tables, and figures in the current paper are reproduced by the code in this repository, not the earlier release.**

### What changed vs. the earlier release

- Proper **60/20/20 chronological** train/val/test split (the earlier release evaluated on the same split used for checkpoint selection).
- Direct glucose prediction (`pred = model(x)`, trained against the scaled absolute glucose target — not a residual/delta).
- A 7-arm ablation over the physics-guided ingredients (stochastic glucose masking + double-monotonicity loss), run on **13 patients**, evaluated on the held-out **test** segment only.
- An honest, **no-spin re-reporting** of results, including reversals relative to the earlier release (see below).

---

## 🔬 Key Findings (honest summary — see paper for full statistics)

1. **The "RMSE paradox":** on pure multi-horizon rollout RMSE, a naive persistence ("glucose stays flat") baseline beats *every* trained LSTM configuration — including the full physics-guided model — at 30/60/120-minute horizons. The most heavily constrained model (`FullCore`) has the *worst* RMSE of all 7 configurations at every horizon.
2. **Accuracy ≠ causal plausibility.** Despite worse RMSE, the physics-guided model (`FullCore`, stochastic glucose masking + double-monotonicity loss) produces directionally correct, more plausible responses under counterfactual insulin/carbohydrate perturbations than a model with the monotonicity loss removed (`NoMono`):
   - Patient-level insulin-dose monotonicity violations (lower is better, out of 4 dose steps): `FullCore` = 0.385, `Vanilla` = 1.231, `NoMono` = 1.846.
   - Patient-level carbohydrate-dose monotonicity violations (out of 3 dose steps): `FullCore` = 0.462, `Vanilla` = 0.615, `NoMono` = 1.000.
   - In the dangerous-overbolus (+15 U) scenario panel (N=5 patients), the mean 2-hour glucose delta for `FullCore` is in the physiologically correct (negative) direction (-24.0 mg/dL, 95% CI includes 0), while `NoMono`'s mean delta is significantly *positive* (+86.9 mg/dL, 95% CI excludes 0) — the wrong direction for an overbolus.
3. **Bottom line:** `FullCore` is *not* the most accurate model by RMSE — it is the *most causally well-behaved* one. The contribution of the physics-guided training (masking + double-monotonicity loss) is improved counterfactual/causal plausibility, not improved point-forecast accuracy, and the two should be evaluated and reported separately.

---

## 🧩 Model Configurations (7-arm ablation)

All configurations share the same 2-layer LSTM (hidden=128, dropout=0.2) trained with Adam (lr=1e-3, 40 epochs, batch=256, seed=42), `StandardScaler` fit on the TRAIN portion only, and a `SEQ_LEN=72` (6h) input window with `ROLL_STEPS=24` (2h) autoregressive rollout for evaluation.

| Config       | Features                                   | Glucose masking (p_mask, p_apply) | Double-monotonicity loss (λ) |
|--------------|---------------------------------------------|:---:|:---:|
| `FullCore`   | glucose, Bolus_IOB, Basal_Rate, carbs, COB, tod_sin, tod_cos, is_weekend | 0.60, 0.40 | 0.10 |
| `Vanilla`    | same as FullCore | 0, 0 | 0 |
| `NoMask`     | same as FullCore | 0, 0 | 0.10 |
| `NoMono`     | same as FullCore | 0.60, 0.40 | 0 |
| `NoIOB`      | glucose, Raw_Bolus (no IOB), Basal_Rate, carbs, COB, tod_sin, tod_cos, is_weekend | 0.60, 0.40 | 0.10 |
| `NoCOB`      | glucose, Bolus_IOB, Basal_Rate, carbs (no COB), tod_sin, tod_cos, is_weekend | 0.60, 0.40 | 0.10 |
| `Enriched`   | FullCore features + heart_rate, steps, sleep_efficiency (10 patients only) | 0.60, 0.40 | 0.10 |

The double-monotonicity loss penalises, via a perturbed second forward pass, a predicted glucose *increase* after an insulin perturbation (×1.5) and a predicted glucose *decrease* after a carbohydrate perturbation (×1.5):

```
loss = MSE + λ * ( relu(pred(insulin×1.5) - pred(base)) + relu(pred(base) - pred(carbs×1.5)) )
```

---

## 📂 Repository Structure

```text
T1DUJxUoM/
├── configs/            # train_config.yaml (7-arm ablation), feature_schema.yaml, panel_protocol.yaml
├── docs/               # Reproducibility guide
├── scripts/            # Executable scripts (synthetic-data demo, real patient CSVs are private)
│   ├── 01_generate_demo.py  # Generates synthetic data with the real 8-feature schema
│   ├── 02_train_model.py    # Training loop: direct prediction, masking + double-monotonicity loss
│   └── 03_run_panel_demo.py # Counterfactual scenario panel (Observed / Unbolused meal / Overbolus / Meal omitted)
├── t1dujxuom/          # Source code package
├── data_demo/          # Synthetic demo data (real patient CSVs are not redistributed)
├── pretrained_models/  # Demo model artefacts
└── outputs/            # Generated figures and CSV metrics
```

## Real-data pipeline (not redistributed)

The 13-patient real-data pipeline (`Processed_Train_Data_V2/Train_<PID>.csv`, 60/20/20 split,
`train_all.py` for the 7-arm ablation, `phase3_*.py` for rollout RMSE / dose-response / safety-panel
analyses used to generate the paper's tables and figures) operates on identifiable clinical CGM data
and is **not distributed** in this repository. The scripts here reproduce the *same architecture,
loss, masking, and evaluation protocol* on synthetic data with an identical schema, so the pipeline's
mechanics can be verified end-to-end without access to the private dataset.

## Citation

If you use this code, please cite the archived release (DOI to be added on publication) and the
paper. The earlier `T1DUJxUoM` v3.0.0 release (DOI [10.5281/zenodo.18434352](https://doi.org/10.5281/zenodo.18434352))
corresponds to a superseded analysis and should **not** be cited as supporting the results in the
current paper.
