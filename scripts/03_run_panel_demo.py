import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "panel_demo")
os.makedirs(OUT_DIR, exist_ok=True)

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

def run_panel():
    print("Running Canonical In-Silico Sensitivity Panel...")
    
    # 1. Load Model & Artifacts
    try:
        model_path = os.path.join(MODEL_DIR, "model_demo.pt")
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_demo.pkl"))
        meta_path = os.path.join(MODEL_DIR, "meta_demo.json")
        with open(meta_path) as f: 
            meta = json.load(f)
            
        features = meta["feature_cols"]
        model = LSTMRegressor(len(features))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"   -> Model loaded. Features: {len(features)}")
        
    except FileNotFoundError:
        print("Error: Model files not found. Run '02_train_model.py' first.")
        return

    # 2. Setup Baseline
    seq_len = 72
    base_seq = np.zeros((1, seq_len, len(features)))
    feat_map = {k:i for i,k in enumerate(features)}
    base_seq[:, :, feat_map["glucose_mgdl"]] = 120 
    base_seq[:, :, feat_map["heart_rate"]] = 80
    
    # 3. Define Scenarios
    scenarios = {
        "A_Baseline (Fasting)": {"carb": 0, "ins": 0},
        "B_Meal_NoBolus":       {"carb": 60, "ins": 0},
        "C_Meal_StandardBolus": {"carb": 60, "ins": 6},
        "D_Meal_OverBolus":     {"carb": 60, "ins": 12}
    }
    
    plt.figure(figsize=(10, 6))
    
    # قائمة لحفظ النتائج الرقمية
    results_data = [] 

    # 4. Simulation Loop
    for name, params in scenarios.items():
        curr_seq = base_seq.copy()
        preds = []
        
        # Inject Event
        curr_seq[0, -1, feat_map["input_meal_carbs"]] = params["carb"]
        curr_seq[0, -1, feat_map["input_insulin"]] = params["ins"]
        
        # Forecast
        for _ in range(48): # 4 hours
            in_tensor = torch.tensor(scaler.transform(curr_seq[0]), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model(in_tensor).item()
            preds.append(pred_scaled)
            
            # Update State
            new_step = curr_seq[0, -1].copy()
            new_step[feat_map["glucose_mgdl"]] = pred_scaled
            new_step[feat_map["input_insulin"]] = 0
            new_step[feat_map["input_meal_carbs"]] = 0
            
            curr_seq[0, :-1, :] = curr_seq[0, 1:, :]
            curr_seq[0, -1, :] = new_step

        plt.plot(preds, label=name, linewidth=2)
        
        # Save Metrics to List
        results_data.append({
            "Scenario": name,
            "Input_Carbs": params["carb"],
            "Input_Insulin": params["ins"],
            "Min_Glucose_Scaled": min(preds),
            "Max_Glucose_Scaled": max(preds),
            "Final_Glucose_Scaled": preds[-1]
        })

    # 5. Finalize Plot
    plt.title("Digital Twin Response (Scaled Glucose Units)")
    plt.xlabel("Time Steps (5 min)")
    plt.ylabel("Glucose Response (Scaled)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save Image
    img_path = os.path.join(OUT_DIR, "clinical_results_demo.png")
    plt.savefig(img_path)
    print(f"   -> Plot saved to: {img_path}")
    
    # Save CSV
    csv_path = os.path.join(OUT_DIR, "clinical_metrics_demo.csv")
    pd.DataFrame(results_data).to_csv(csv_path, index=False)
    print(f"   -> Metrics saved to: {csv_path}")
    
    print("✅ Panel Complete.")

if __name__ == "__main__":
    run_panel()
