# Reproducibility Guide (T1DUJxUoM v5.0.0)

This repository provides a reproducible pipeline for training and evaluating a **six-configuration
ablation** of physiology-guided LSTM digital twins for Type 1 Diabetes, replicated across **two
independent cohorts** (University of Manchester T1D-UOM, OhioT1DM) and **five random seeds**, under
a strict **60/20/20 chronological train/validation/test split**, with **direct glucose prediction**,
**stochastic glucose masking**, and a **corrected, raw-space double-monotonicity loss**.

Due to privacy regulations and dataset licensing, the original clinical CSVs are **not distributed**
by this repository (see "Scope and Data Governance" below for how to obtain each cohort legitimately).
We provide a synthetic data generator that creates data with the same 8-column `FullCore` schema
(`glucose_mgdl, Bolus_IOB, Basal_Rate, meal_carbs_g, COB_g, tod_sin, tod_cos, is_weekend`), enabling
end-to-end verification of the training loop, loss function, and counterfactual scenario panel
without access to either dataset.

## 1. Scope and data governance

- **Public artefacts (this repository):** training/inference code (`scripts/02_train_model.py`), the
  counterfactual scenario panel (`scripts/03_run_panel_demo.py`), configs (`configs/`), the synthetic
  data generator (`scripts/01_generate_demo.py`), and the regression test guarding the perturbation
  fix (`tests/test_perturbation.py`).
- **Obtainable by the researcher, not redistributed here:** the Manchester T1D-UOM dataset is openly
  licensed (CC BY 4.0) and downloadable directly from Zenodo
  (DOI [10.5281/zenodo.15806142](https://doi.org/10.5281/zenodo.15806142)); OhioT1DM requires a
  data-use agreement with its maintainers. Running `scripts/02_train_model.py` against either
  dataset (after adapting the file paths and, for Ohio, the patient ID list) reproduces the paper's
  training protocol exactly.
- **Not distributed at all:** the resulting per-patient, per-seed model checkpoints (390 for
  Manchester + 360 for Ohio) are large, overfit to specific individual patients, and have no direct
  reuse value for a different patient population — the reproducible artefact is the *protocol* that
  derives a new patient-specific model from new data, not the fitted weights themselves.

## 2. Environment setup

### 2.1 Prerequisites
- Python 3.10+
- A virtual environment is recommended.

### 2.2 Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Running the demo pipeline

```bash
python scripts/01_generate_demo.py   # synthetic patient, FullCore 8-feature schema
python scripts/02_train_model.py     # 60/20/20 split, masking + corrected double-monotonicity loss, 40 epochs
python scripts/03_run_panel_demo.py  # counterfactual scenario panel, CSV + plot under outputs/panel_demo/
python -m pytest tests/test_perturbation.py   # regression test for the raw-space perturbation fix
```

## 4. Running the full six-configuration, five-seed protocol on your own data

`scripts/02_train_model.py` trains the `FullCore` configuration by default. To reproduce the paper's
full protocol on your own cohort:

1. Point `DATA_DIR` at a directory of `Train_<PID>.csv` files with the 8-column `FullCore` schema
   (or the 8-column `NoIOB` variant / 7-column `NoCOB` variant for those configurations).
2. Loop `CONFIG_NAME` over `FullCore, Vanilla, NoMask, NoMono, NoIOB, NoCOB` (feature lists and
   `(p_mask, p_apply, lambda)` per Table in the README).
3. Loop the random seed over `{42, 43, 44, 45, 46}` (or your own choice — five seeds is a protocol
   choice for seed-robustness reporting, not a requirement of the method itself).
4. For the dose-response and rollout evaluation used in the paper's Results, perturb the on-board
   insulin (or carbohydrate) channel using the corrected raw-space formula described in the README
   and in `tests/test_perturbation.py`, then run a 24-step (2-hour) autoregressive rollout from each
   held-out test-segment window.

## 5. Mapping to the paper

| Paper artefact | What it evaluates |
|---|---|
| One-step RMSE (Fig. "Adding physiological structure costs one-step accuracy") | `scripts/02_train_model.py` reports VAL/TEST RMSE per config/seed/patient |
| Multi-horizon rollout RMSE, RMSE-paradox figure | Autoregressive rollout of the trained checkpoint against Persistence and Ridge-AR baselines |
| Insulin/carbohydrate dose-response slope, sign-correctness, monotonicity | Graded-dose perturbation + rollout audit; illustrated at single-window scale by `scripts/03_run_panel_demo.py` scenarios B-D |
| $\lambda$-sensitivity table | Same training/evaluation loop repeated at $\lambda \in \{0, 0.05, 0.10, 0.20\}$ |

## 6. Honest reporting note

This pipeline evaluates every reported number on a genuinely held-out TEST segment (the final 20% of
each participant's chronological record), never on the split used for checkpoint selection. A naive
persistence baseline outperforms every trained neural configuration on multi-horizon rollout RMSE at
60/120 minutes in both cohorts — see the paper's Results and Discussion for the full statistics and
the argument that forecasting accuracy and treatment-response plausibility must be audited
separately, not assumed to track one another.
