import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "panel_demo")
os.makedirs(OUT_DIR, exist_ok=True)

# Model Artifacts
PATIENT_ID = "2301" 
CSV_FILE = "demo_patient.csv"
META_FILE = f"meta_{PATIENT_ID}.json"
SCALER_FILE = f"scaler_x_{PATIENT_ID}.pkl"
MODEL_FILE = f"model_{PATIENT_ID}.pt"

@dataclass
class RunCfg:
    seq_len: int = 72
    step_min: int = 5
    horizon_min: int = 360
    dia_min: int = 300
    peak_min: int = 75
    carb_absorb_min: int = 180
    warmup_min: int = 360
    meal_at_min: int = 30
    clip_glucose: Tuple[float, float] = (40.0, 400.0)

CFG = RunCfg()

# --- HELPERS ---
def load_scaler(path):
    import joblib
    try:
        return joblib.load(path)
    except:
        with open(path, "rb") as f: return pickle.load(f)

def smart_load_glucose(df):
    col = "output_cgm" if "output_cgm" in df else "glucose_mgdl"
    g = pd.to_numeric(df[col], errors="coerce").ffill().astype(float)
    if g.mean() < 30.0: 
        print(f"   [INFO] Detected mmol/L units (Mean: {g.mean():.1f}). Converting to mg/dL...")
        return g * 18.0182
    return g

def _triangle_kernel(length, peak):
    k = np.zeros(length)
    peak = max(1, min(peak, length-2))
    k[:peak+1] = np.linspace(0, 1, peak+1)
    k[peak:] = np.linspace(1, 0, length-peak)
    return k / k.sum() if k.sum() > 0 else k

def compute_iob_cob(ins, carb, cfg):
    dia = int(cfg.dia_min/cfg.step_min)
    peak = int(cfg.peak_min/cfg.step_min)
    dur = int(cfg.carb_absorb_min/cfg.step_min)
    k_iob = _triangle_kernel(dia, peak)
    k_cob = _triangle_kernel(dur, dur//3)
    iob = np.convolve(ins, k_iob, mode="full")[:len(ins)]
    cob = np.convolve(carb, k_cob, mode="full")[:len(carb)]
    return iob, cob

def calc_metrics(g_arr):
    return {
        "Mean_Glucose": np.mean(g_arr),
        "TIR_70_180": np.mean((g_arr >= 70) & (g_arr <= 180)) * 100,
        "TAR_gt180": np.mean(g_arr > 180) * 100,
        "TBR_lt70": np.mean(g_arr < 70) * 100,
        "Peak_Glucose": np.max(g_arr),
        "Lowest_Glucose": np.min(g_arr)
    }

# --- MODEL ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def load_model(path, input_dim):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt: sd = ckpt["state_dict"]
    else: sd = ckpt
    
    w = sd.get("lstm.weight_ih_l0")
    if w is None: w = sd.get("model.lstm.weight_ih_l0")
    if w is None: raise ValueError("Could not find LSTM weights")

    hidden = w.shape[0] // 4
    layers = sum(1 for k in sd.keys() if "weight_ih_l" in k)
    
    print(f"   [Model] Architecture: Input={input_dim}, Hidden={hidden}, Layers={layers}")
    
    model = LSTMRegressor(input_dim, hidden, layers)
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    return model

# --- RUNNER ---
@torch.no_grad()
def run_simulation(model, scaler, meta, df, start, cfg, scenario_fn):
    warmup = int(cfg.warmup_min/cfg.step_min)
    horizon = int(cfg.horizon_min/cfg.step_min)
    chunk = df.iloc[start-warmup : start+horizon].copy().reset_index(drop=True)
    
    i_vals = chunk["input_insulin"].values.astype(float)
    c_vals = chunk["input_meal_carbs"].values.astype(float)
    i_vals, c_vals = scenario_fn(i_vals, c_vals, warmup)
    
    chunk["input_insulin"] = i_vals
    chunk["input_meal_carbs"] = c_vals
    chunk["IOB_U"], chunk["COB_g"] = compute_iob_cob(i_vals, c_vals, cfg)
    
    feats = meta.get("features", meta.get("feature_cols"))
    g_idx = feats.index("glucose_mgdl")
    data = chunk[feats].values.copy()
    
    y_mean = meta.get("y_mean")
    y_std = meta.get("y_std")
    
    preds = []
    for t in range(warmup, len(chunk)):
        seq = data[t-cfg.seq_len : t]
        seq_scaled = scaler.transform(seq)
        x = torch.from_numpy(seq_scaled).float().unsqueeze(0)
        p_scaled = model(x).item()
        
        if y_mean: p_mgdl = p_scaled * y_std + y_mean
        else: 
            d = np.zeros((1, len(feats)))
            d[:, g_idx] = p_scaled
            p_mgdl = scaler.inverse_transform(d)[0, g_idx]
            
        p_mgdl = max(cfg.clip_glucose[0], min(cfg.clip_glucose[1], p_mgdl))
        data[t, g_idx] = p_mgdl
        preds.append(p_mgdl)
        
    return np.array(preds)

def main():
    print("?? Starting In-Silico Panel (Demo Mode)...")
    dpath = os.path.join(DATA_DIR, CSV_FILE)
    if not os.path.exists(dpath):
        print("? Demo data missing."); return

    df = pd.read_csv(dpath)
    df["glucose_mgdl"] = smart_load_glucose(df)
    
    try:
        meta = json.load(open(os.path.join(MODEL_DIR, META_FILE)))
        scaler = load_scaler(os.path.join(MODEL_DIR, SCALER_FILE))
        dim = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else len(meta["features"])
        model = load_model(os.path.join(MODEL_DIR, MODEL_FILE), dim)
    except Exception as e:
        print(f"? Error loading model: {e}"); return

    # --- INSULIN SENSITIVITY ---
    # Force a stronger ratio for demo visualization clarity
    ic = 0.15 
    print(f"Forced Demo I:C Ratio: {ic:.3f}")
    
    # --- SMART WINDOW SEARCH ---
    print("Searching for an eventful window (Meal > 0)...")
    horizon = int(CFG.horizon_min / CFG.step_min)
    warmup = int(CFG.warmup_min / CFG.step_min)
    start_idx = 1000 # Default fallback
    
    for i in range(1000, len(df) - horizon):
        window_carbs = df["input_meal_carbs"].iloc[i : i + horizon].sum()
        if window_carbs > 20:
            start_idx = i
            print(f"? Found Meal Scenario at index {start_idx} (Total Carbs: {window_carbs}g)")
            break
            
    scenarios = {
        "Baseline": lambda i,c,s: (i,c),
        "Fasting": lambda i,c,s: (np.where(np.arange(len(c))>=s, 0, i), np.where(np.arange(len(c))>=s, 0, c)),
        "Meal_Bolus": lambda i,c,s: (i + (np.arange(len(i))==s+6)*50*ic, np.where(np.arange(len(c))==s+6, 50, np.where(np.arange(len(c))>s, 0, c))),
        "Meal_NoBolus": lambda i,c,s: (i, np.where(np.arange(len(c))==s+6, 50, np.where(np.arange(len(c))>s, 0, c))),
        "Meal_OverBolus": lambda i,c,s: (i + (np.arange(len(i))==s+6)*50*ic*1.5, np.where(np.arange(len(c))==s+6, 50, np.where(np.arange(len(c))>s, 0, c))),
    }
    
    time_x = np.arange(0, CFG.horizon_min, CFG.step_min)
    plt.figure(figsize=(10,6))
    colors = ["black", "gray", "green", "red", "blue"]
    styles = ["-", "--", "-", "-.", ":"]
    
    # Store metrics here
    all_metrics = []
    
    for (name, fn), col, sty in zip(scenarios.items(), colors, styles):
        print(f"   -> Simulating: {name}")
        p = run_simulation(model, scaler, meta, df, start_idx, CFG, fn)
        
        # Calculate Metrics
        m = calc_metrics(p)
        m["Scenario"] = name
        all_metrics.append(m)
        
        plt.plot(time_x, p, label=name, color=col, linestyle=sty, linewidth=2)
        
    plt.axhspan(70, 180, color="green", alpha=0.1, label="Target Range")
    plt.title(f"In-Silico Panel (Patient {PATIENT_ID} Demo)")
    plt.xlabel("Time (min)")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    out_img = os.path.join(OUT_DIR, "demo_panel_plot.png")
    out_csv = os.path.join(OUT_DIR, "demo_metrics.csv")
    
    plt.savefig(out_img, dpi=300)
    pd.DataFrame(all_metrics).to_csv(out_csv, index=False)
    
    print(f"\n? Plot saved to: {out_img}")
    print(f"? Metrics saved to: {out_csv}")

if __name__ == "__main__":
    main()