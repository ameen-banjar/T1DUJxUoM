import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Settings
PATIENT_ID = "2301" # Or "NewPatient"
CSV_FILE = "demo_patient.csv" 
EPOCHS = 10 # Short training for demo (use 100+ for real research)
BATCH_SIZE = 64
SEQ_LEN = 72

# --- HELPERS (Feature Engineering) ---
def _triangle_kernel(length, peak):
    k = np.zeros(length)
    peak = max(1, min(peak, length-2))
    k[:peak+1] = np.linspace(0, 1, peak+1)
    k[peak:] = np.linspace(1, 0, length-peak)
    return k / k.sum() if k.sum() > 0 else k

def add_physiological_features(df):
    """Calculates IOB and COB from raw insulin/carb inputs"""
    print("   -> Engineering Physiological Features (IOB/COB)...")
    
    # Constants for decay (approximate)
    dia_min = 300
    peak_min = 75
    carb_absorb_min = 180
    step_min = 5
    
    dia = int(dia_min/step_min)
    peak = int(peak_min/step_min)
    dur = int(carb_absorb_min/step_min)
    
    k_iob = _triangle_kernel(dia, peak)
    k_cob = _triangle_kernel(dur, dur//3)
    
    ins = df["input_insulin"].values
    carb = df["input_meal_carbs"].values
    
    # Convolution
    iob = np.convolve(ins, k_iob, mode="full")[:len(ins)]
    cob = np.convolve(carb, k_cob, mode="full")[:len(carb)]
    
    df["IOB_U"] = iob
    df["COB_g"] = cob
    return df

def smart_load_glucose(df):
    col = "output_cgm" if "output_cgm" in df else "glucose_mgdl"
    g = pd.to_numeric(df[col], errors="coerce").ffill().astype(float)
    if g.mean() < 30.0: 
        print(f"   [INFO] Detected mmol/L units (Mean: {g.mean():.1f}). Converting to mg/dL...")
        return g * 18.0182
    return g

# --- DATASET & MODEL ---
class T1DDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx : idx+self.seq_len], self.y[idx+self.seq_len]

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# --- TRAINER ---
def train_digital_twin():
    print(f"?? Starting Training for Patient {PATIENT_ID}...")
    
    # 1. Load Data
    dpath = os.path.join(DATA_DIR, CSV_FILE)
    if not os.path.exists(dpath):
        print("? Data file not found."); return
        
    df = pd.read_csv(dpath)
    
    # 2. Preprocessing
    df["glucose_mgdl"] = smart_load_glucose(df)
    
    # Calculate IOB/COB if missing (This answers your question!)
    if "IOB_U" not in df.columns or "COB_g" not in df.columns:
        df = add_physiological_features(df)
        
    # Define Features
    features = [
        "glucose_mgdl", "input_insulin", "input_meal_carbs", 
        "IOB_U", "COB_g", 
        "heart_rate", "steps", "sleep_efficiency", 
        "feat_hour_of_day_sin", "feat_hour_of_day_cos", 
        "feat_is_weekend", "heart_rate_WRTbaseline"
    ]
    
    # Check if all columns exist
    for f in features:
        if f not in df.columns:
            print(f"?? Warning: Column {f} missing. Filling with 0.")
            df[f] = 0.0

    # 3. Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(df[features].values)
    
    # Extract Target (Glucose is index 0)
    target = data[:, 0] 
    
    # 4. Prepare Loaders
    dataset = T1DDataset(data, target, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Setup Model
    model = LSTMRegressor(input_dim=len(features), hidden_dim=128, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 6. Training Loop
    print(f"   Training on {len(dataset)} sequences for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 2 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    # 7. Save Artifacts (The Golden Files)
    print("?? Saving Model Artifacts...")
    
    # Save Weights (.pt)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_{PATIENT_ID}.pt"))
    
    # Save Scaler (.pkl)
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_x_{PATIENT_ID}.pkl"))
    
    # Save Meta (.json)
    meta = {
        "patient_id": PATIENT_ID,
        "features": features,
        "y_mean": float(scaler.mean_[0]), # Glucose mean
        "y_std": float(scaler.scale_[0])   # Glucose std
    }
    with open(os.path.join(MODEL_DIR, f"meta_{PATIENT_ID}.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"? Training Complete! Artifacts saved in {MODEL_DIR}")

if __name__ == "__main__":
    train_digital_twin()