from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# --- Configuration & Constants ---
EOS_API_KEY = "apk.3053e16b25638262973480cd8f2f4a5d4ac9d3e14dc89c31ada444822fa9f146"
ZONES = {
    "Alandi": {"lat": 18.68, "lon": 73.73, "area": 9.95},
    "Amanora": {"lat": 18.52, "lon": 73.93, "area": 4.50},
    "Balewadi": {"lat": 18.57, "lon": 73.78, "area": 6.80},
    "Bhosari": {"lat": 18.63, "lon": 73.81, "area": 13.50},
    "Hadapsar": {"lat": 18.51, "lon": 73.92, "area": 18.20},
    "Hinjawadi": {"lat": 18.58, "lon": 73.74, "area": 15.60},
    "Katraj": {"lat": 18.45, "lon": 73.86, "area": 11.40},
    "Kharadi": {"lat": 18.55, "lon": 73.94, "area": 14.30},
    "Pimpri": {"lat": 18.62, "lon": 73.80, "area": 12.00},
    "Shivajinagar": {"lat": 18.53, "lon": 73.85, "area": 7.30},
    "Wagholi": {"lat": 18.58, "lon": 73.99, "area": 19.40}
}
SQFT_PER_KM2 = 10_763_910.4167
TARGET_TEMP = 25
NEEM_COVERAGE_SQFT = 450
MAX_COOLING_DELTA = 12
SQFT_PER_EXISTING_TREE = 1073
YEAR_GAP = 2  # 2025 to 2027

# --- Initialize AI Models (Global) ---
temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
cool_model = RandomForestRegressor(n_estimators=100, random_state=42)

def train_models():
    n_samples = 1000
    # Temp Model Training
    t_train = pd.DataFrame({
        "current_temp": np.random.uniform(28, 38, n_samples),
        "area_sqft": np.random.uniform(1e6, 2e8, n_samples),
        "year_offset": np.random.randint(0, 5, n_samples)
    })
    t_train["future_temp"] = t_train["current_temp"] + 0.25 * t_train["year_offset"] + 0.000000002 * t_train["area_sqft"]
    temp_model.fit(t_train[["current_temp", "area_sqft", "year_offset"]], t_train["future_temp"])
    
    # Cool Model Training
    c_train = pd.DataFrame({
        "area_sqft": np.random.uniform(1e6, 2e8, n_samples),
        "existing_trees": np.random.uniform(500, 200000, n_samples),
        "added_neem": np.random.uniform(0, 300000, n_samples),
        "temp": np.random.uniform(30, 40, n_samples)
    })
    c_train["cooling"] = (0.00004 * c_train["added_neem"]).clip(0, MAX_COOLING_DELTA)
    cool_model.fit(c_train[["area_sqft", "existing_trees", "added_neem", "temp"]], c_train["cooling"])

train_models()

def get_live_temp(lat, lon):
    url = f"https://api-connect.eos.com/weather/v1/current?lat={lat}&lon={lon}&api_key={EOS_API_KEY}"
    try:
        r = requests.get(url, timeout=5)
        return r.json()["main"]["temp"] if r.status_code == 200 else np.random.uniform(30, 35)
    except: return np.random.uniform(30, 35)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict')
def predict():
    results = []
    for zone, z in ZONES.items():
        curr_temp = get_live_temp(z["lat"], z["lon"])
        area_sqft = z["area"] * SQFT_PER_KM2
        
        f_temp = temp_model.predict([[curr_temp, area_sqft, YEAR_GAP]])[0]
        delta = max(0, f_temp - TARGET_TEMP)
        
        existing = area_sqft / SQFT_PER_EXISTING_TREE
        candidates = np.linspace(0, area_sqft/NEEM_COVERAGE_SQFT, 20)
        cooling_preds = cool_model.predict([[area_sqft, existing, t, f_temp] for t in candidates])
        
        neem_req = candidates[np.argmin(np.abs(cooling_preds - delta))]
        add_neem = max(0, neem_req - existing)

        results.append({
            "zone": zone, "lat": z["lat"], "lon": z["lon"],
            "future_temp": round(f_temp, 2), "delta": round(delta, 2),
            "existing": int(existing), "needed": int(add_neem),
            "total": int(existing + add_neem)
        })
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
