from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from feature_builder import build_feature_vector, _feature_names

app = FastAPI(title="Airbnb Price Prediction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
models = {}
model_dir = Path("../models")

try:
    knn_data = joblib.load(model_dir / "knn" / "knn_final_pipeline.pkl")
    # Handle KNN model which is saved as dict with pipeline inside
    if isinstance(knn_data, dict) and 'pipeline' in knn_data:
        models['knn'] = knn_data['pipeline']
    else:
        models['knn'] = knn_data
    print("[OK] KNN model loaded")
except Exception as e:
    print(f"[ERROR] KNN model failed: {e}")

try:
    models['linear_regression'] = joblib.load(model_dir / "linear_regression" / "lr_final_pipeline.pkl")
    print("[OK] Linear Regression model loaded")
except Exception as e:
    print(f"[ERROR] Linear Regression model failed: {e}")

try:
    models['random_forest'] = joblib.load(model_dir / "random_forest" / "rf_final_pipeline.pkl")
    print("[OK] Random Forest model loaded")
except Exception as e:
    print(f"[ERROR] Random Forest model failed: {e}")

try:
    models['xgboost'] = joblib.load(model_dir / "xgboost" / "xgb_final_pipeline.pkl")
    print("[OK] XGBoost model loaded")
except Exception as e:
    print(f"[ERROR] XGBoost model failed: {e}")

class PredictionInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input_data: PredictionInput):
    """Get predictions from all available models"""
    try:
        feature_vector = build_feature_vector(input_data.data)

        # Create DataFrame with proper column names
        if _feature_names:
            X = pd.DataFrame([feature_vector], columns=_feature_names)
        else:
            X = pd.DataFrame([feature_vector])

        predictions = {}
        for model_name, model in models.items():
            try:
                pred = model.predict(X)[0]
                predictions[model_name] = float(pred)
            except Exception as e:
                predictions[model_name] = f"Error: {str(e)}"

        valid_preds = [p for p in predictions.values() if isinstance(p, (int, float))]
        avg = float(np.mean(valid_preds)) if valid_preds else 0.0

        return {
            "predictions": predictions,
            "average": avg
        }
    except Exception as e:
        return {
            "error": str(e),
            "predictions": {},
            "average": 0.0
        }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
