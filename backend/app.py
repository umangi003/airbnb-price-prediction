from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from feature_builder import build_feature_vector

app = FastAPI(title="Airbnb Price Prediction API - XGBoost Only")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_dir = Path("../models")

try:
    model = joblib.load(model_dir / "xgboost" / "xgb_final_pipeline.pkl")
    print("[OK] XGBoost model loaded successfully")
except Exception as e:
    print(f"[ERROR] XGBoost model failed: {e}")

class PredictionInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        feature_vector = build_feature_vector(input_data.data)
        X = pd.DataFrame([feature_vector])
        log_prediction = model.predict(X)[0]
        price_eur = float(np.expm1(log_prediction))
        
        return {
            "model": "XGBoost",
            "log_prediction": float(log_prediction),
            "predicted_price_eur": price_eur,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "predicted_price_eur": 0.0, "status": "error"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
