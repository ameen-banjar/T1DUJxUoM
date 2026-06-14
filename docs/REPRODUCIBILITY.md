# Reproducibility Guide (T1DUJxUoM)

This repository provides a reproducible pipeline for training and evaluating a 7-arm ablation of
LSTM-based digital twins for Type 1 Diabetes, under a **60/20/20 chronological train/validation/test
split**, with **direct glucose prediction**, **stochastic glucose masking**, and a **double-monotonicity
loss**.

Due to privacy regulations, the original clinical datasets (`Train_<PID>.csv`, 13 patients) are **not
distributed**. We provide a synthetic data generator that creates data with the same 8-column FullCore
schema (`glucose_mgdl, Bolus_IOB, Basal_Rate, meal_carbs_g, COB_g, tod_sin, tod_cos, is_weekend`),
enabling end-to-end verification of the training loop, loss function, and counterfactual scenario panel.

## 1. Scope and Data Governance

- **Public artefacts:** training/inference code (`scripts/02_train_model.py`), the counterfactual
  scenario panel (`scripts/03_run_panel_demo.py`), configs (`configs/`), and the synthetic data
  generator (`scripts/01_generate_demo.py`).
- **Private artefacts:** the 13 original patient CSVs and the corresponding `retrain_v3/models/*`
  checkpoints used to produce the paper's tables and figures.

## 2. Environment Setup

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
python scripts/01_generate_demo.py   # synthetic 14-day patient, FullCore schema
python scripts/02_train_model.py     # 60/20/20 split, masking + double-monotonicity loss, 40 epochs
python scripts/03_run_panel_demo.py  # counterfactual scenario panel (A-D), CSV + plot under outputs/panel_demo/
```

## 4. Mapping to the paper

| Paper artefact | Source (real-data pipeline, not redistributed) | Demo equivalent |
|---|---|---|
| Multi-horizon rollout RMSE (fig02-05, `tab:ablation`) | `phase3_rollout_rmse.py`, `phase3_persistence.py`, `train_all.py` | `scripts/02_train_model.py` reports VAL/TEST rmse |
| Counterfactual safety panel (fig06/fig07, overbolus) | `phase3_safety_panel.py` | `scripts/03_run_panel_demo.py` |
| Insulin/carb dose-response (fig08-11, `tab:dose_response`, `tab:patient_monotonicity`) | `phase3_dose_response.py`, `phase3_dose_response_carb.py` | `scripts/03_run_panel_demo.py` scenarios B-D (single-window illustration) |

## 5. Honest reporting note

The earlier `T1DUJxUoM` v3.0.0 release evaluated checkpoints on the same split used for model
selection, which inflates apparent accuracy. This repository's pipeline uses a genuinely held-out TEST
segment (last 20% chronologically) for every reported number. As a result, several findings reverse
relative to the earlier release (see README "Key Findings" and the paper's Results/Discussion/Limitations
sections) — most notably, a naive persistence baseline outperforms every trained configuration on
multi-horizon RMSE at 30/60/120 minutes.
