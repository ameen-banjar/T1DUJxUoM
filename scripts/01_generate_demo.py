import os
import numpy as np
import pandas as pd

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data_demo")
os.makedirs(DATA_DIR, exist_ok=True)

IOB_DECAY = np.exp(-5.0 / 180.0)
COB_DECAY = np.exp(-5.0 / 120.0)


def generate_demo_data():
    """Synthetic data with the same 8-column FullCore schema used by the
    real-data pipeline (glucose_mgdl, Bolus_IOB, Basal_Rate, meal_carbs_g,
    COB_g, tod_sin, tod_cos, is_weekend), so 02_train_model.py and
    03_run_panel_demo.py exercise the real architecture/loss end-to-end."""
    print("Generating synthetic demo data with the FullCore feature schema...")

    dates = pd.date_range("2024-01-01", periods=14 * 288, freq="5min")
    steps = len(dates)

    day_cycle = np.sin(2 * np.pi * dates.hour.values / 24)
    glucose = 140 + (20 * day_cycle) + np.random.normal(0, 5, steps)

    bolus_iob = np.zeros(steps)
    basal_rate = np.full(steps, 0.8)
    meal_carbs = np.zeros(steps)
    cob = np.zeros(steps)

    for i in range(0, steps, 60):
        if np.random.rand() > 0.3:
            meal_carbs[i] = 50.0
            bolus_iob[i] += 5.0
            if i + 72 < steps:
                glucose[i:i + 36] += np.linspace(0, 40, 36)
                glucose[i + 36:i + 72] -= np.linspace(0, 40, 36)

    for t in range(1, steps):
        bolus_iob[t] = bolus_iob[t - 1] * IOB_DECAY + bolus_iob[t]
        cob[t] = cob[t - 1] * COB_DECAY + meal_carbs[t]

    angle = 2 * np.pi * (dates.hour.values * 60 + dates.minute.values) / (24 * 60)
    tod_sin = np.sin(angle)
    tod_cos = np.cos(angle)
    is_weekend = dates.dayofweek.isin([5, 6]).astype(int)

    df = pd.DataFrame({
        "time": dates,
        "glucose_mgdl": np.clip(glucose, 40, 400),
        "Bolus_IOB": bolus_iob,
        "Basal_Rate": basal_rate,
        "meal_carbs_g": meal_carbs,
        "COB_g": cob,
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
        "is_weekend": is_weekend,
    })

    out_path = os.path.join(DATA_DIR, "demo_patient.csv")
    df.to_csv(out_path, index=False)
    print(f"Done. Demo data saved to: {out_path}")


if __name__ == "__main__":
    generate_demo_data()
