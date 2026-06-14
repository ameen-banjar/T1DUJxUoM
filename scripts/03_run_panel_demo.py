"""
In-silico counterfactual scenario panel, matching phase3_safety_panel.py,
run on the demo model trained by 02_train_model.py (FullCore feature schema,
direct glucose prediction, CPU, 24-step / 2h autoregressive rollout).

Scenarios (matching configs/panel_protocol.yaml):
  A: Observed Future (Baseline)        - no perturbation
  B: Unbolused Meal (+60g Carbs)       - carb impulse, COB_g decays tau=120min
  C: Dangerous Overbolus (+15U Insulin) - insulin impulse, Bolus_IOB decays tau=180min
  D: Meal Omitted After Insulin (+10U)  - insulin impulse, Bolus_IOB decays tau=180min
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "panel_demo")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 72
HORIZON_STEPS = 24
GLUCOSE_IDX, IOB_IDX, BASAL_IDX, CARB_IDX, COB_IDX, SIN_IDX, COS_IDX, WKD_IDX = range(8)

IOB_DECAY = np.exp(-5.0 / 180.0)
COB_DECAY = np.exp(-5.0 / 120.0)


class CoreLSTM(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_decay_curve(length, impulse_val, tau_mins):
    decay_factor = np.exp(-5.0 / tau_mins)
    curve = np.zeros(length)
    curve[0] = impulse_val
    for i in range(1, length):
        curve[i] = curve[i - 1] * decay_factor
    return curve


def simulate_scenario(model, scaler, history_raw, future_exog_raw):
    mean, scale = scaler.mean_, scaler.scale_
    hist_scaled = (history_raw - mean) / scale
    fut_scaled = (future_exog_raw - mean) / scale

    curr_seq = torch.from_numpy(hist_scaled.astype(np.float32)).unsqueeze(0)
    fut_tensor = torch.from_numpy(fut_scaled.astype(np.float32)).unsqueeze(0)

    preds_raw = []
    for step in range(HORIZON_STEPS):
        with torch.no_grad():
            pred_scaled = model(curr_seq).squeeze().item()
        pred_raw = pred_scaled * scale[GLUCOSE_IDX] + mean[GLUCOSE_IDX]
        preds_raw.append(pred_raw)

        next_step = fut_tensor[:, step:step + 1, :].clone()
        next_step[:, 0, GLUCOSE_IDX] = pred_scaled
        curr_seq = torch.cat([curr_seq[:, 1:, :], next_step], dim=1)

    return np.array(preds_raw)


def run_panel():
    print("Running counterfactual scenario panel...")

    model_path = os.path.join(MODEL_DIR, "model_demo.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler_demo.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta_demo.json")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("Error: model files not found. Run '02_train_model.py' first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    features = meta["feature_cols"]

    model = CoreLSTM(n_features=len(features))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(os.path.join(DATA_DIR, "demo_patient.csv"))
    data = df[features].values.astype(np.float64)
    n = len(data)
    test_data = data[int(n * 0.80):]

    idx = len(test_data) - SEQ_LEN - HORIZON_STEPS - 1
    hist = test_data[idx:idx + SEQ_LEN]
    fut_obs = test_data[idx + SEQ_LEN:idx + SEQ_LEN + HORIZON_STEPS]
    start_gluc = hist[-1, GLUCOSE_IDX]

    blank_fut = fut_obs.copy()
    blank_fut[:, IOB_IDX] = 0.0
    blank_fut[:, CARB_IDX] = 0.0
    blank_fut[:, COB_IDX] = 0.0
    blank_fut[:, BASAL_IDX] = hist[-1, BASAL_IDX]

    scenarios = {
        "A: Observed Future (Baseline)": fut_obs.copy(),
        "B: Unbolused Meal (+60g Carbs)": blank_fut.copy(),
        "C: Dangerous Overbolus (+15U Insulin)": blank_fut.copy(),
        "D: Meal Omitted After Insulin (+10U)": blank_fut.copy(),
    }
    scenarios["B: Unbolused Meal (+60g Carbs)"][0, CARB_IDX] = 60.0
    scenarios["B: Unbolused Meal (+60g Carbs)"][:, COB_IDX] = build_decay_curve(HORIZON_STEPS, 60.0, 120.0)
    scenarios["C: Dangerous Overbolus (+15U Insulin)"][:, IOB_IDX] = build_decay_curve(HORIZON_STEPS, 15.0, 180.0)
    scenarios["D: Meal Omitted After Insulin (+10U)"][:, IOB_IDX] = build_decay_curve(HORIZON_STEPS, 10.0, 180.0)

    plt.figure(figsize=(10, 6))
    results_data = []
    for name, fut_scenario in scenarios.items():
        preds = simulate_scenario(model, scaler, hist, fut_scenario)
        plt.plot(preds, label=name, linewidth=2)
        results_data.append({
            "Scenario": name,
            "Start_Gluc": round(start_gluc, 1),
            "Delta_2h": round(preds[-1] - start_gluc, 1),
            "Nadir": round(preds.min(), 1),
            "Max": round(preds.max(), 1),
            "Crossed70": bool(preds.min() < 70.0),
            "Crossed180": bool(preds.max() > 180.0),
        })

    plt.title("Digital Twin Counterfactual Response (mg/dL)")
    plt.xlabel("Step (5 min)")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    img_path = os.path.join(OUT_DIR, "panel_demo.png")
    plt.savefig(img_path)
    print(f"   -> Plot saved to: {img_path}")

    csv_path = os.path.join(OUT_DIR, "panel_metrics_demo.csv")
    pd.DataFrame(results_data).to_csv(csv_path, index=False)
    print(f"   -> Metrics saved to: {csv_path}")
    print(pd.DataFrame(results_data).to_string(index=False))


if __name__ == "__main__":
    run_panel()
