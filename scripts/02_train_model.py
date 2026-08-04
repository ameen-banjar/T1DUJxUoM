"""
Training pipeline matching the real-data retrain_v3 pipeline (train_all.py),
run here on the synthetic demo data (FullCore feature schema, 8 columns).

  - 2-layer LSTM (hidden=128, dropout=0.2), DIRECT glucose prediction
    (pred = model(x), trained against the scaled absolute glucose target)
  - 60/20/20 chronological train/val/test split; StandardScaler fit on TRAIN only
  - stochastic glucose masking (p_mask, p_apply)
  - double-monotonicity loss: lam * (relu(pred(insulin*1.5) - pred(base))
    + relu(pred(base) - pred(carbs*1.5))), base_pred via a separate forward pass
  - Adam lr=1e-3, seed=42
  - best checkpoint = lowest VAL rmse; report TEST rmse at that checkpoint

Trains the "FullCore" configuration by default (set CONFIG_NAME below to any
key in configs/train_config.yaml -> ablation_configs to try another arm).
"""
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
SEQ_LEN = 72
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
PERTURB = 1.5
DEVICE = "cpu"
GLUCOSE_IDX = 0

CONFIG_NAME = "FullCore"
FEATURES = ["glucose_mgdl", "Bolus_IOB", "Basal_Rate", "meal_carbs_g", "COB_g",
            "tod_sin", "tod_cos", "is_weekend"]
INS_IDX, CARB_IDX = 1, 4
P_MASK, P_APPLY, LAM = 0.60, 0.40, 0.10


class CoreLSTM(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_windows(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len, GLUCOSE_IDX])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse_eval(model, X_t, y_mgdl, g_mean, g_scale):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_t).squeeze(-1)
    preds_mgdl = pred_scaled.cpu().numpy() * g_scale + g_mean
    return np.sqrt(np.mean((preds_mgdl - y_mgdl) ** 2))


def train_pipeline():
    set_seed(SEED)
    print(f"Starting training pipeline: config={CONFIG_NAME}")

    csv_path = os.path.join(DATA_DIR, "demo_patient.csv")
    if not os.path.exists(csv_path):
        print("Error: demo_patient.csv not found. Please run '01_generate_demo.py' first.")
        return

    df = pd.read_csv(csv_path)
    data = df[FEATURES].values.astype(np.float64)
    n = len(data)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train_raw = data[:train_end]
    val_raw = data[train_end:val_end]
    test_raw = data[val_end:]

    scaler = StandardScaler().fit(train_raw)
    train_s = scaler.transform(train_raw)
    val_s = scaler.transform(val_raw)
    test_s = scaler.transform(test_raw)

    g_mean, g_scale = scaler.mean_[GLUCOSE_IDX], scaler.scale_[GLUCOSE_IDX]
    # Raw-space perturbation offset: multiplying the RAW feature by PERTURB
    # (x' = PERTURB*x) corresponds, in standardised space, to
    #   z' = PERTURB*z + (PERTURB-1)*mean/scale,
    # not simply z' = PERTURB*z. Using the pure standardised-space scaling
    # moves below-mean windows the WRONG way (can even flip IOB negative).
    # See tests/test_perturbation.py for the regression test.
    ins_off = (PERTURB - 1.0) * scaler.mean_[INS_IDX] / scaler.scale_[INS_IDX]
    carb_off = (PERTURB - 1.0) * scaler.mean_[CARB_IDX] / scaler.scale_[CARB_IDX]

    Xtr, ytr = make_windows(train_s, SEQ_LEN)
    Xval, yval = make_windows(val_s, SEQ_LEN)
    Xtest, ytest = make_windows(test_s, SEQ_LEN)
    print(f"   -> windows: train={len(Xtr)}, val={len(Xval)}, test={len(Xtest)}")

    model = CoreLSTM(n_features=len(FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xval_t = torch.from_numpy(Xval).to(DEVICE)
    yval_mgdl = yval * g_scale + g_mean
    Xtest_t = torch.from_numpy(Xtest).to(DEVICE)
    ytest_mgdl = ytest * g_scale + g_mean

    best_val = float("inf")
    best_test = None
    best_state = None
    n_train = len(Xtr)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = Xtr[idx].copy()
            yb = ytr[idx]

            if P_APPLY > 0 and np.random.rand() < P_APPLY:
                keep = (np.random.rand(xb.shape[0], xb.shape[1]) > P_MASK).astype(np.float32)
                xb[:, :, GLUCOSE_IDX] = xb[:, :, GLUCOSE_IDX] * keep

            xb_t = torch.from_numpy(xb).to(DEVICE)
            yb_t = torch.from_numpy(yb).to(DEVICE)

            pred = model(xb_t).squeeze(-1)
            mse = nn.functional.mse_loss(pred, yb_t)

            if LAM > 0:
                base_pred = model(xb_t).squeeze(-1)

                xb_ins = xb_t.clone()
                xb_ins[:, :, INS_IDX] = xb_ins[:, :, INS_IDX] * PERTURB + ins_off
                ins_pred = model(xb_ins).squeeze(-1)
                l_ins = torch.relu(ins_pred - base_pred).mean()

                xb_carbs = xb_t.clone()
                xb_carbs[:, :, CARB_IDX] = xb_carbs[:, :, CARB_IDX] * PERTURB + carb_off
                carbs_pred = model(xb_carbs).squeeze(-1)
                l_carbs = torch.relu(base_pred - carbs_pred).mean()

                loss = mse + LAM * (l_ins + l_carbs)
            else:
                loss = mse

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_rmse = rmse_eval(model, Xval_t, yval_mgdl, g_mean, g_scale)
        if val_rmse < best_val:
            best_val = val_rmse
            best_test = rmse_eval(model, Xtest_t, ytest_mgdl, g_mean, g_scale)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:2d}/{EPOCHS} | val_rmse={val_rmse:.2f} | best_val={best_val:.2f}")

    print(f"\nBest VAL rmse = {best_val:.2f} mg/dL, TEST rmse at that checkpoint = {best_test:.2f} mg/dL")

    torch.save(best_state, os.path.join(MODEL_DIR, "model_demo.pt"))
    with open(os.path.join(MODEL_DIR, "scaler_demo.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "config": CONFIG_NAME,
        "feature_cols": FEATURES,
        "seq_len": SEQ_LEN,
        "best_val_rmse": float(best_val),
        "best_test_rmse": float(best_test),
    }
    import json
    with open(os.path.join(MODEL_DIR, "meta_demo.json"), "w") as f:
        json.dump(meta, f)

    print("Done. Model ready for the counterfactual scenario panel (03_run_panel_demo.py).")


if __name__ == "__main__":
    train_pipeline()
