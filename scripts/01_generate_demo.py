import os
import sys
import numpy as np
import pandas as pd

# Setup Paths
# Use try-except to handle Jupyter/Script context differences
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
os.makedirs(DATA_DIR, exist_ok=True)

def simple_kernel(n, p):
    """Creates a simple decay kernel for IOB/COB estimation."""
    k = np.zeros(n)
    k[:p] = np.linspace(0, 1, p)
    k[p:] = np.linspace(1, 0, n-p)
    return k / k.sum() if k.sum() > 0 else k

def generate_demo_data():
    print("Generating physics-ready synthetic demo data...")
    
    # 1. Create Time Index (14 Days)
    # 5-minute intervals
    dates = pd.date_range("2024-01-01", periods=14*288, freq="5min")
    steps = len(dates)
    
    # 2. Simulate Base Glucose (Daily Sine Wave)
    # FIX: Added .values to ensure we work with a mutable numpy array, not a pandas Index
    day_cycle = np.sin(2 * np.pi * dates.hour.values / 24)
    glucose = 140 + (20 * day_cycle) + np.random.normal(0, 5, steps)
    
    # 3. Simulate Meals & Insulin Events
    carbs = np.zeros(steps)
    insulin = np.zeros(steps)
    
    # Randomly inject meals approx every 5 hours
    for i in range(0, steps, 60): 
        if np.random.rand() > 0.3:
            carbs[i] = 50  # 50g Carb Meal
            insulin[i] = 5 # 5U Insulin
            
            # Add physiological response (Rise then Fall)
            if i + 72 < steps:
                # This line caused the error before because 'glucose' was immutable.
                # Now it is a numpy array, so this works:
                glucose[i:i+36] += np.linspace(0, 40, 36) 
                glucose[i+36:i+72] -= np.linspace(0, 40, 36)

    # 4. Construct DataFrame
    df = pd.DataFrame({
        "time": dates,
        "glucose_mgdl": np.clip(glucose, 40, 400),
        "input_insulin": insulin,
        "input_meal_carbs": carbs,
        "heart_rate": np.random.normal(80, 5, steps),
        "steps": np.random.choice([0, 100], p=[0.9, 0.1], size=steps),
        "sleep_efficiency": 0.0,
        "is_train": 1
    })
    
    # 5. Feature Engineering (Must match training requirements)
    df["feat_hour_of_day_sin"] = np.sin(2 * np.pi * dates.hour / 24)
    df["feat_hour_of_day_cos"] = np.cos(2 * np.pi * dates.hour / 24)
    df["feat_is_weekend"] = dates.dayofweek.isin([5,6]).astype(int)
    df["heart_rate_WRTbaseline"] = 0.0

    # Calculate approximate IOB (Insulin On Board) and COB (Carbs On Board)
    df["IOB_U"] = np.convolve(df["input_insulin"], simple_kernel(60, 15), mode='full')[:steps]
    df["COB_g"] = np.convolve(df["input_meal_carbs"], simple_kernel(36, 12), mode='full')[:steps]

    # Save to CSV
    out_path = os.path.join(DATA_DIR, "demo_patient.csv")
    df.to_csv(out_path, index=False)
    print(f"Done. Demo data saved to: {out_path}")

if __name__ == "__main__":
    generate_demo_data()
