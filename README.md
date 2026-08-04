# T1DUJxUoM: Forecasting Accuracy Is Not Enough — Safety-Oriented Evaluation of Neural Digital Twins in Type 1 Diabetes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-orange.svg)](https://github.com/)
[![tests](https://github.com/ameen-banjar/T1DUJxUoM/actions/workflows/tests.yml/badge.svg)](https://github.com/ameen-banjar/T1DUJxUoM/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21794825.svg)](https://doi.org/10.5281/zenodo.21794825)

**Authors:**
- **Ahmed Ibrahim** (University of Jeddah, Saudi Arabia)
- **Ameen Banjar** (University of Jeddah, Saudi Arabia)
- **Simon Harper** (University of Manchester, UK)
- **Ala Alarood** (University of Jeddah, Saudi Arabia)
- **Eesa Alsolami** (University of Jeddah, Saudi Arabia)

---

## Overview

This repository is the reproducibility companion for our paper evaluating **patient-specific,
physiology-guided LSTM digital twins for Type 1 Diabetes** across **two independent cohorts**
(University of Manchester T1D-UOM, 13 participants; OhioT1DM, 12 participants), **five random
seeds**, and **six neural configurations**, under a strict **60/20/20 chronological
train/validation/test split**.

The paper's central claim is that **forecasting accuracy and treatment-response plausibility are
separable properties**: the most accurate model by one-step RMSE (`Vanilla`) is not the model that
responds most plausibly to a simulated insulin or carbohydrate intervention, and a non-learning
persistence baseline beats every trained neural configuration on multi-horizon rollout RMSE. We
audit both axes explicitly, rather than assuming low forecasting error implies safe intervention
behaviour.

### Corrected in this release (v5.0.0)

An implementation defect was identified and fixed in the training-time monotonicity-penalty
perturbation. The penalty is meant to probe the model's response to a genuine
$1.5\times$ increase of the raw insulin-on-board / carbohydrate-on-board value, but an earlier
version of the code scaled the *standardised* value directly ($z' = 1.5z$), which for below-mean
windows moves the raw feature in the wrong direction (and can even drive on-board insulin negative).
The corrected update applies the equivalent raw-space scaling in standardised space:

```
z' = 1.5*z + (1.5 - 1) * mean / scale
```

`tests/test_perturbation.py` is a regression test guarding this specific defect, run automatically
on every push via `.github/workflows/tests.yml`. All `lambda > 0` models (`FullCore`, `NoMask`,
`NoIOB`, `NoCOB`) were retrained after the fix; `Vanilla` and `NoMono` (`lambda = 0`) were never
affected, since the penalty term is never computed when `lambda = 0`.

---

## Key findings (see paper for full statistics)

1. **The RMSE paradox.** A physiology-free `Vanilla` LSTM has the lowest one-step RMSE in both
   cohorts (Manchester 7.00, Ohio 5.02 mg/dL), and a naive persistence baseline with no learned
   parameters beats *every* trained neural configuration — including the physiology-guided
   `FullCore` — on multi-horizon rollout RMSE at 60 and 120 minutes, even after rollout-aware
   fine-tuning removes the one-step/multi-step training-evaluation mismatch.
2. **Accuracy and plausibility are separable.** On a graded insulin dose-response audit, `FullCore`
   produces the steepest correctly-directed (glucose-lowering) response among the primary
   configurations in both cohorts (dose-response slope: Manchester -5.24, Ohio -3.37 mg/dL per U),
   significantly stronger than the unconstrained `NoMono` configuration (which is wrong-signed in
   both cohorts). Its advantage over the simpler `Vanilla` model is cohort-dependent (significant on
   Ohio, not on Manchester) for reasons the paper does not claim to fully explain.
3. **Which ingredient matters.** An ablation across `NoMask`, `NoIOB`, and `NoCOB` shows the
   directional monotonicity penalty, not the masking augmentation, drives the plausibility effect:
   removing IOB weakens the insulin response far more than removing COB (mechanistically expected,
   since IOB is the feature the insulin audit directly probes), and removing masking alone
   (`NoMask`) produces an even steeper response than `FullCore`.
4. **The trade-off is stable.** Varying the monotonicity weight $\lambda \in \{0, 0.05, 0.10, 0.20\}$
   shows one-step accuracy is flat across all $\lambda > 0$ values, while sign-correctness and
   monotonicity increase with $\lambda$ in both cohorts — the accuracy cost is paid as soon as the
   constraint is switched on, and does not grow further with its strength.

---

## Model configurations (six-arm ablation, both cohorts)

All configurations share a 2-layer LSTM (hidden=128, dropout=0.2) trained with Adam (lr=1e-3, 40
epochs, batch=256), `StandardScaler` fit on the TRAIN portion only, `SEQ_LEN=72` (6h) input window,
and evaluated across five seeds (42-46).

| Config     | Features                                                                 | Masking (p_mask, p_apply) | Monotonicity $\lambda$ |
|------------|---------------------------------------------------------------------------|:---:|:---:|
| `FullCore` | glucose, Bolus_IOB, Basal_Rate, carbs, COB, tod_sin, tod_cos, is_weekend  | 0.60, 0.40 | 0.10 |
| `Vanilla`  | same as FullCore                                                          | 0, 0       | 0    |
| `NoMask`   | same as FullCore                                                          | 0, 0       | 0.10 |
| `NoMono`   | same as FullCore                                                          | 0.60, 0.40 | 0    |
| `NoIOB`    | glucose, Raw_Bolus (no IOB), Basal_Rate, carbs, COB, tod_sin, tod_cos, is_weekend | 0.60, 0.40 | 0.10 |
| `NoCOB`    | glucose, Bolus_IOB, Basal_Rate, carbs (no COB), tod_sin, tod_cos, is_weekend      | 0.60, 0.40 | 0.10 |

The double-monotonicity loss penalises, via a perturbed second forward pass, a predicted glucose
*increase* after an insulin perturbation and a predicted glucose *decrease* after a carbohydrate
perturbation:

```
loss = MSE + lambda * ( relu(pred(insulin_perturbed) - pred(base))
                       + relu(pred(base) - pred(carbs_perturbed)) )
```

where `insulin_perturbed`/`carbs_perturbed` apply the corrected raw-space $1.5\times$ scaling
described above, in standardised-feature space with the additive offset.

---

## Repository structure

```text
T1DUJxUoM/
├── configs/            # train_config.yaml, feature_schema.yaml, panel_protocol.yaml
├── docs/                # Reproducibility guide
├── scripts/
│   ├── 01_generate_demo.py  # Synthetic data, real 8-feature FullCore schema
│   ├── 02_train_model.py    # Training loop: direct prediction, masking + corrected double-monotonicity loss
│   └── 03_run_panel_demo.py # Counterfactual scenario panel (baseline / unbolused meal / overbolus / meal omitted)
├── tests/
│   └── test_perturbation.py # Regression test for the raw-space perturbation fix
├── .github/workflows/tests.yml  # Runs the perturbation test on every push/PR
├── t1dujxuom/           # Source code package
├── data_demo/           # Synthetic demo data (real patient CSVs are not redistributed)
├── pretrained_models/   # Demo model artefacts (regenerated with the corrected code)
└── outputs/             # Generated demo figures and CSV metrics
```

## Real-data pipeline (not redistributed)

The paper's tables and figures are produced from two cohorts' patient-level CGM/pump/nutrition
records (Manchester T1D-UOM, openly licensed CC BY 4.0, Zenodo DOI
[10.5281/zenodo.15806142](https://doi.org/10.5281/zenodo.15806142); OhioT1DM, available under its
own data-use agreement) using the same architecture, loss, masking, evaluation protocol, and
five-seed procedure implemented in `scripts/02_train_model.py`. Manchester results are fully
reproducible from the openly licensed source data; Ohio results require the researcher's own
data-use agreement with the OhioT1DM maintainers. The scripts in this repository reproduce the
identical training/evaluation mechanics on synthetic data with the same schema, so the pipeline can
be verified end-to-end without access to either dataset.

## Citation

If you use this code, please cite the archived release (DOI
[10.5281/zenodo.21794825](https://doi.org/10.5281/zenodo.21794825), v5.0.0) and the paper. The
earlier `T1DUJxUoM` v3.0.0 (DOI [10.5281/zenodo.18434352](https://doi.org/10.5281/zenodo.18434352))
and v4.0.0 (DOI [10.5281/zenodo.20693048](https://doi.org/10.5281/zenodo.20693048)) releases
correspond to superseded analyses (different split protocol, and — for v4.0.0 — the
standardised-space perturbation defect described above) and should **not** be cited as supporting
the results in the current paper.
