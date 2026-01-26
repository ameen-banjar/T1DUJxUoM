# Reproducibility Guide (T1DUJxUoM)

This repository provides a fully reproducible pipeline for training and evaluating an LSTM-based Digital Twin for Type 1 Diabetes using a **canonical in-silico panel**.

Due to privacy regulations, the original clinical datasets (UoM patient CSVs) are **not distributed**. However, we provide a **Synthetic Data Generator** that creates data with the exact same schema and statistical properties, enabling end-to-end verification of the pipeline.

## 1. Scope and Data Governance

- **Public Artifacts:** Training/Inference code, In-Silico Panel runner, Configs, Synthetic Generator.
- **Private Artifacts:** Original patient CSV files (`Final_Train_2301.csv`, etc.) containing real clinical data.

## 2. Environment Setup

### 2.1 Prerequisites
- Python 3.10+
- A virtual environment is recommended.

### 2.2 Installation
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt