import pandas as pd
import numpy as np
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data_demo")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_demo_data():
    print("Generating BALANCED synthetic demo data...")
    
    # Time settings
    n_days = 14
    steps = n_days * 288 # 5 min intervals
    dates = pd.date_range("2024-01-01", periods=steps, freq="5min")
    
    # --- Modification: Simulate a controlled but varied patient ---
    
    # 1. Base Glucose with Daily Cycle (Targeting ~120 mg/dL)
    day_cycle = np.sin(2 * np.pi * dates.hour / 24)
    base_glucose = 130 + (15 * day_cycle) 
    
    # 2. Add Noise (Short term volatility)
    noise = np.random.normal(0, 5, steps)
    
    # 3. Meals & Spikes (Adding Gaussian bumps)
    meal_spikes = np.zeros(steps)
    insulin = np.zeros(steps)
    carbs = np.zeros(steps)
    
    # Random meal simulation
    for i in range(0, steps, 60): # Approx every 5 hours
        if np.random.rand() > 0.3: # 70% chance of meal
            meal_size = np.random.choice([30, 50, 70])
            carbs[i] = meal_size
            insulin[i] = meal_size * 0.1 # Standard bolus
            
            # Simulate meal impact (2-3 hours rise)
            end = min(steps, i + 36)
            t_range = np.linspace(-2, 2, end-i)
            bump = np.exp(-t_range**2) * (meal_size * 0.8) 
            meal_spikes[i:end] += bump

    # 4. Combine: Base + Meals + Noise
    glucose = base_glucose + meal_spikes + noise
    
    # 5. Safety Bounds (To prevent unrealistic values)
    glucose = np.clip(glucose, 60, 300)
    
    # Construct DataFrame
    df = pd.DataFrame({
        "time": dates,
        "datetime_local": dates,
        "output_cgm": glucose, 
        "input_insulin": insulin,
        "input_meal_carbs": carbs,
        "heart_rate": np.random.normal(80, 5, steps),
        "steps": np.random.choice([0, 100], p=[0.9, 0.1], size=steps),
        "sleep_efficiency": 0,
        "feat_hour_of_day_sin": np.sin(2 * np.pi * dates.hour / 24),
        "feat_hour_of_day_cos": np.cos(2 * np.pi * dates.hour / 24),
        "feat_is_weekend": dates.dayofweek.isin([5,6]).astype(int),
        "heart_rate_WRTbaseline": 0,
        "is_train": 1 
    })
    
    # Approx IOB/COB calculations
    df["IOB_U"] = df["input_insulin"].rolling(60, min_periods=1).sum() * 0.9 
    df["COB_g"] = df["input_meal_carbs"].rolling(36, min_periods=1).sum() * 0.8
    
    out_path = os.path.join(OUT_DIR, "demo_patient.csv")
    df.to_csv(out_path, index=False)
    print(f"? Success! Balanced demo data created at: {out_path}")
    
    # Use numpy mean to avoid index error
    print(f"   Mean Glucose: {np.mean(glucose):.1f} mg/dL (Target: ~130-140)")

if __name__ == "__main__":
    generate_demo_data()