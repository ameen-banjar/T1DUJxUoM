import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Load Config
with open(os.path.join(BASE_DIR, "configs", "train_config.yaml"), 'r') as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. THE CORE INNOVATION: Physics-Guided Augmentation
# ---------------------------------------------------------
def augment_data_flooding(X_scaled, features, augment_factor=10):
    """
    Implements the 'Counterfactual Data Flooding' technique.
    It identifies high-insulin/high-carb moments and clones them with 
    enforced physics constraints (Insulin -> Drop, Carbs -> Rise) 
    to override any reverse causality in the raw data.
    """
    print(f"   -> Applying Physics-Guided Data Flooding (Factor={augment_factor})...")
    
    # Identify Column Indices
    try:
        g_idx = features.index("glucose_mgdl")
        ins_idx = features.index("input_insulin")
        carb_idx = features.index("input_meal_carbs")
    except ValueError:
        print("   [!] Warning: Required columns not found. Skipping augmentation.")
        return X_scaled, X_scaled[:, 0]

    # Separate X (Inputs) and y (Target: Next Glucose Step)
    # Note: In this simplified demo, we treat column 0 as target for augmentation logic
    y_scaled = X_scaled[:, g_idx].copy()
    
    # --- A. INSULIN FLOODING (Force Drop) ---
    # Find samples where Insulin is active (> 0.5 scaled implies high activity relative to median)
    # Note: RobustScaler centers at median. Values > 1.0 are typically outliers/high.
    ins_mask = X_scaled[:, ins_idx] > 1.0 
    
    X_aug_ins = np.array([])
    y_aug_ins = np.array([])
    
    if np.sum(ins_mask) > 0:
        X_selected = X_scaled[ins_mask]
        y_selected = y_scaled[ins_mask]
        
        # PHYSICS CONSTRAINT: Force target glucose DOWN
        # We subtract 2.0 standard deviations (in scaled space)
        y_forced = y_selected - 2.0 
        
        # FLOODING: Duplicate these "corrected" samples multiple times
        X_aug_ins = np.tile(X_selected, (augment_factor, 1))
        y_aug_ins = np.tile(y_forced, augment_factor)
        print(f"      + Injected {len(y_aug_ins)} Insulin-Correction samples.")

    # --- B. CARB FLOODING (Force Rise) ---
    carb_mask = X_scaled[:, carb_idx] > 1.0
    
    X_aug_carb = np.array([])
    y_aug_carb = np.array([])
    
    if np.sum(carb_mask) > 0:
        X_selected = X_scaled[carb_mask]
        y_selected = y_scaled[carb_mask]
        
        # PHYSICS CONSTRAINT: Force target glucose UP
        y_forced = y_selected + 2.0
        
        # FLOODING
        X_aug_carb = np.tile(X_selected, (augment_factor, 1))
        y_aug_carb = np.tile(y_forced, augment_factor)
        print(f"      + Injected {len(y_aug_carb)} Carb-Correction samples.")

    # --- COMBINE REAL + SYNTHETIC ---
    list_X = [X_scaled]
    list_y = [y_scaled]
    
    if len(X_aug_ins) > 0:
        list_X.append(X_aug_ins)
        list_y.append(y_aug_ins)
        
    if len(X_aug_carb) > 0:
        list_X.append(X_aug_carb)
        list_y.append(y_aug_carb)
        
    X_final = np.concatenate(list_X, axis=0)
    y_final = np.concatenate(list_y, axis=0)
    
    return X_final, y_final

# ---------------------------------------------------------
# 2. DATASET WITH STOCHASTIC MASKING
# ---------------------------------------------------------
class PhysicsGuidedDataset(Dataset):
    def __init__(self, X, y, seq_len, mask_prob=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.g_idx = 0 # Assuming glucose is first feature

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        x_seq = self.X[i:i+self.seq_len].clone()
        y_val = self.y[i+self.seq_len]
        
        # --- STOCHASTIC MASKING ---
        # Randomly hide glucose history to force model to look at Insulin/Carbs
        if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
            x_seq[:, self.g_idx] = 0.0
            
        return x_seq, y_val

# ---------------------------------------------------------
# 3. MODEL ARCHITECTURE
# ---------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# ---------------------------------------------------------
# 4. MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def train_pipeline():
    print("🚀 Starting Digital Twin Training Pipeline...")
    
    # 1. Load Data
    csv_path = os.path.join(DATA_DIR, "demo_patient.csv")
    if not os.path.exists(csv_path):
        print("Error: demo_patient.csv not found. Run 01_generate_demo.py first.")
        return
    
    df = pd.read_csv(csv_path)
    
    features = [
        "glucose_mgdl", "input_insulin", "input_meal_carbs", 
        "IOB_U", "COB_g", "heart_rate", "steps", 
        "sleep_efficiency", "feat_hour_of_day_sin", 
        "feat_hour_of_day_cos", "feat_is_weekend", "heart_rate_WRTbaseline"
    ]
    
    # 2. Scaling (RobustScaler is best for physiological data)
    print("   -> Scaling Data...")
    scaler = RobustScaler()
    X_raw_scaled = scaler.fit_transform(df[features].values)
    
    # 3. PHYSICS-GUIDED AUGMENTATION CHECK
    use_physics = CONFIG['training'].get('physics_guided', False)
    
    if use_physics:
        aug_factor = CONFIG['training'].get('augmentation_factor', 10)
        mask_p = CONFIG['training'].get('mask_probability', 0.5)
        # Apply Flooding
        X_train, y_train = augment_data_flooding(X_raw_scaled, features, factor=aug_factor)
    else:
        print("   -> Standard Training (No Physics Guidance).")
        X_train = X_raw_scaled
        y_train = X_raw_scaled[:, 0] # Default target
        mask_p = 0.0

    # 4. Create Dataset & Loader
    BATCH_SIZE = CONFIG['training'].get('batch_size', 64)
    SEQ_LEN = 72
    
    train_ds = PhysicsGuidedDataset(X_train, y_train, SEQ_LEN, mask_prob=mask_p)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"   -> Training Set Size: {len(train_ds)} sequences (Masking={mask_p*100}%)")

    # 5. Model Setup
    model = LSTMRegressor(input_dim=len(features))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['training'].get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    # 6. Training Loop
    EPOCHS = CONFIG['training'].get('epochs', 10)
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
            avg_loss = total_loss / len(loader)
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 7. Save Artifacts
    print("💾 Saving Model & Scaler...")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_demo.pt"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_demo.pkl"))
    
    # Save Metadata for the Panel
    meta = {
        "patient_id": "DEMO",
        "trained_glucose_unit": "mg/dL",
        "feature_cols": features,
        "y_mean": float(df["glucose_mgdl"].mean()), # Approx for denormalization
        "y_std": float(df["glucose_mgdl"].std()),
    }
    with open(os.path.join(MODEL_DIR, "meta_demo.json"), "w") as f:
        json.dump(meta, f)

    print("✅ Done! Ready for In-Silico Panel.")

if __name__ == "__main__":
    train_pipeline()
