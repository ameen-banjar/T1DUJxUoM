# T1DUJxUoM: Physics-Guided Digital Twins & In-Silico Panel for T1D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-orange.svg)](https://github.com/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18380842.svg)](https://doi.org/10.5281/zenodo.18380842)

**Authors:**
- **Ameen Banjar** (University of Jeddah, Saudi Arabia)
- **Ahmed Ibrahim** (University of Jeddah, Saudi Arabia)
- **Simon Harper** (University of Manchester, UK)

---

## 📌 Overview
This repository contains the official implementation of the research paper:  
**"A Canonical In-Silico Sensitivity Panel for Verifying Counterfactual Responsiveness in Patient-Specific, Data-Driven Type 1 Diabetes Digital Twins"**

We address a critical failure mode in deep learning for T1D: **Autoregressive Blindness**, where LSTM models memorize glucose history while ignoring therapeutic inputs (insulin/carbs).

To solve this, we introduce a **Physics-Guided Training Strategy** that combines:
1.  **Stochastic Glucose Masking:** Forcing the model to rely on inputs rather than history.
2.  **Counterfactual Data Flooding:** Augmenting training data with physics-constrained samples to enforce causal dynamics (e.g., Insulin $\to$ Glucose Drop).

The models are validated using our **Canonical In-Silico Sensitivity Panel**, which rigorously tests physiological plausibility under counterfactual scenarios (e.g., missed bolus, fasting, overdose).

---

## 🔬 Key Scientific Contributions

### 1. Physics-Guided Data Flooding
Standard LSTM training often learns reverse causality from real-world data (e.g., associating high insulin with high glucose). We implement a **Data Flooding** algorithm that injects synthetic, physics-compliant samples into the training batch to correct these learned biases.

### 2. Stochastic Glucose Masking
To prevent "lazy learning," we randomly mask the glucose history input during training (default `mask_prob=0.90` for resistant patients). This forces the neural network to learn the transfer function of Insulin and Carbohydrates directly.

### 3. Canonical Sensitivity Panel
A standardized validation protocol that subjects the Digital Twin to 4 clinical scenarios:
- **A. Baseline:** Fasting state.
- **B. Meal (No Bolus):** Verifies carb sensitivity.
- **C. Meal (Standard Bolus):** Verifies insulin efficacy.
- **D. Meal (Over Bolus):** Verifies safety margins (hypoglycemia risk).

---

## 📂 Repository Structure

```text
T1DUJxUoM/
├── configs/            # Configuration files (Physics flags, hyperparameters)
├── docs/               # Reproducibility guide & Metric definitions
├── scripts/            # Executable scripts
│   ├── 01_generate_demo.py  # Generates synthetic data for testing
│   ├── 02_train_model.py    # Main training loop (with Physics-Guided logic)
│   └── 03_run_panel_demo.py # Runs the In-Silico Clinical Trial
├── t1dujxuom/          # Source code package (Data loading, LSTM architecture)
├── data_demo/          # Contains synthetic data for verification
├── pretrained_models/  # Contains pre-trained models
└── outputs/            # Generated figures and CSV metrics
