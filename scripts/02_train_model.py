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
# Handle path logic for both Jupyter and Script execution
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)

# Load Configuration
config_path = os.path.join(BASE_DIR, "configs", "train_config.yaml")
if os.path.exists(config_path):
    with open(config_path, 'r') as f: 
        CONFIG = yaml.safe_load(f)
else:
    # Default config if file missing
    CONFIG = {'training': {'physics_guided': True, 'mask_probability': 0.9, 'augmentation_factor': 10}}

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. PHYSICS-GUIDED AUGMENTATION (Counterfactual Data Flooding)
# ---------------------------------------------------------
def augment_data_flooding(X_scaled, features, augment_factor=10):
    """
    Implements the 'Counterfactual Data Flooding' technique.
    It creates synthetic samples where Insulin guarantees a glucose drop,
    and Carbs guarantee a glucose rise, overriding spurious correlations.
    """
    print(f"   -> Applying Physics-Guided Data Flooding (Factor={augment_factor})...")
    
    g_idx = features.index("glucose_mgdl")
    ins_idx = features.index("input_insulin")
    carb_idx = features.index("input_meal_carbs")

    # Target y is essentially the next glucose value (simplified)
    y_scaled = X_scaled[:, g_idx].copy()
    
    # --- A. INSULIN FLOODING (Force Drop) ---
    # Identify moments of high insulin activity (> 1.0 sigma)
    ins_mask = X_scaled[:, ins_idx] > 1.0 
    X_aug_ins = []
    y_aug_ins = []
    
    if np.sum(ins_mask) > 0:
        X_sel = X_scaled[ins_mask]
        y_sel = y_scaled[ins_mask]
        
        # PHYSICS CONSTRAINT: Force target glucose DOWN by 2.0 std dev
        y_forced = y_sel - 2.0 
        
        # Duplicate these corrected samples to "flood" the batch
        X_aug_ins = np.tile(X_sel, (augment_factor, 1))
        y_aug_ins = np.tile(y_forced, augment_factor)

    # --- B. CARB FLOODING (Force Rise) ---
    carb_mask = X_scaled[:, carb_idx] > 1.0
    X_aug_carb = []
    y_aug_carb = []
    
    if np.sum(carb_mask) > 0:
        X_sel = X_scaled[carb_mask]
        y_sel = y_scaled[carb_mask]
        
        # PHYSICS CONSTRAINT: Force target glucose UP by 2.0 std dev
        y_forced = y_sel + 2.0 
        
        X_aug_carb = np.tile(X_sel, (augment_factor, 1))
        y_aug_carb = np.tile(y_forced, augment_factor)

    # --- C. MERGE DATA ---
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
# 2. PHYSICS-GUIDED DATASET (Stochastic Masking)
# ---------------------------------------------------------
class PhysicsGuidedDataset(Dataset):
    def __init__(self, X, y, seq_len, mask_prob=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.g_idx = 0 # Assuming glucose is the first feature

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        x_seq = self.X[i:i+self.seq_len].clone()
        y_val = self.y[i+self.seq_len]
        
        # --- STOCHASTIC GLUCOSE MASKING ---
        # Randomly zero out the glucose history channel.
        # This prevents the model from relying solely on autoregression.
        if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
            x_seq[:, self.g_idx] = 0.0
            
        return x_seq, y_val

# ---------------------------------------------------------
# 3. LSTM MODEL ARCHITECTURE
# ---------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# ---------------------------------------------------------
# 4. TRAINING PIPELINE
# ---------------------------------------------------------
def train_pipeline():
    print("Starting Physics-Guided Training Pipeline...")
    
    # 1. Load Data
    csv_path = os.path.join(DATA_DIR, "demo_patient.csv")
    if not os.path.exists(csv_path):
        print("Error: demo_patient.csv not found. Please run '01_generate_demo.py' first.")
        return
    
    df = pd.read_csv(csv_path)
    
    features = [
        "glucose_mgdl", "input_insulin", "input_meal_carbs", 
        "IOB_U", "COB_g", "heart_rate", "steps", 
        "sleep_efficiency", "feat_hour_of_day_sin", 
        "feat_hour_of_day_cos", "feat_is_weekend", "heart_rate_WRTbaseline"
    ]
    
    # 2. Scale Data
    scaler = RobustScaler()
    X_raw_scaled = scaler.fit_transform(df[features].values)
    
    # 3. Apply Physics Guidance (Flooding)
    use_physics = CONFIG['training'].get('physics_guided', False)
    
    if use_physics:
        aug_factor = CONFIG['training'].get('augmentation_factor', 10)
        mask_p = CONFIG['training'].get('mask_probability', 0.5)
        # Apply Flooding (FIXED ARGUMENT NAME HERE)
        X_train, y_train = augment_data_flooding(X_raw_scaled, features, augment_factor=aug_factor)
    else:
        print("   -> Standard Training Mode (No Physics Guidance).")
        X_train = X_raw_scaled
        y_train = X_raw_scaled[:, 0]
        mask_p = 0.0

    # 4. Create Dataset & Loader
    ds = PhysicsGuidedDataset(X_train, y_train, seq_len=72, mask_prob=mask_p)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    print(f"   -> Training on {len(ds)} sequences (Masking Probability: {mask_p:.2f})")

    # 5. Initialize Model
    model = LSTMRegressor(input_dim=len(features))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 6. Training Loop
    epochs = 10 # Short training for demo purposes
    model.train()
    
    for ep in range(epochs):
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if (ep+1) % 2 == 0:
            print(f"   Epoch {ep+1}/{epochs} | Loss: {np.mean(losses):.4f}")

    # 7. Save Artifacts
    print("Saving model artifacts...")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_demo.pt"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_demo.pkl"))
    
    # Save Metadata for the Panel Script
    meta = {
        "patient_id": "DEMO",
        "feature_cols": features,
        "y_mean": float(df["glucose_mgdl"].mean()), 
        "y_std": float(df["glucose_mgdl"].std())
    }
    with open(os.path.join(MODEL_DIR, "meta_demo.json"), "w") as f:
        json.dump(meta, f)
        
    print("Done. Model ready for validation.")

if __name__ == "__main__":
    train_pipeline()
