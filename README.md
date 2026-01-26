# T1DUJxUoM: Digital Twin & In-Silico Panel for Type 1 Diabetes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-orange.svg)](https://github.com/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Authors:**
- **Ameen Banjar** (University of Jeddah, Saudi Arabia)
- **Ahmed Ibrahim** (University of Jeddah, Saudi Arabia)
- **Simon Harper** (University of Manchester, UK)

---

## 📌 Overview
This repository contains the official implementation of the research paper:  
**"A Canonical In-Silico Sensitivity Panel for Verifying Counterfactual Responsiveness in Patient-Specific, Data-Driven Type 1 Diabetes Digital Twins"**

We present a robust framework for training LSTM-based Digital Twins using real-world clinical data. Unlike traditional forecasting models, our Digital Twins are validated using a **Canonical In-Silico Sensitivity Panel**, which rigorously tests physiological plausibility under counterfactual scenarios (e.g., missed bolus, fasting, overdose).

## 📂 Repository Structure

```text
T1DUJxUoM/
├── configs/            # Configuration files (Feature order, scaling, hyperparameters)
├── docs/               # Reproducibility guide & Metric definitions
├── scripts/            # Executable scripts (Generate data, Train, Run Panel)
├── t1dujxuom/          # Source code package (Data loading, LSTM architecture)
├── data_demo/          # Contains synthetic data for verification
├── pretrained_models/  # Contains the pre-trained demo model (Patient 2301)
└── outputs/            # Generated figures and CSV metrics