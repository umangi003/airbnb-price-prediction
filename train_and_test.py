import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import os
import sys

sys.path.insert(0, '/workspaces/airbnb-price-prediction/backend')
from feature_builder import build_feature_vector

print("=" * 70)
print("AIRBNB PRICE PREDICTION - COMPLETE TRAINING & TESTING")
print("=" * 70)
print()

print("STEP 1: Loading training data...")
X_train = pd.read_csv('/workspaces/airbnb-price-prediction/data/processed/X_train_scaled.csv')
y_train = pd.read_csv('/workspaces/airbnb-price-prediction/data/processed/y_train.csv')['log_price'].values
print(f"✅ Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features\n")

print("STEP 2: Training 4 ML models...")

xgb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
xgb_model.fit(X_train.iloc[:, :10], y_train)
os.makedirs('/workspaces/airbnb-price-prediction/models/xgboost', exist_ok=True)
joblib.dump(xgb_model, '/workspaces/airbnb-price-prediction/models/xgboost/xgb_final_pipeline.pkl')
print("✅ XGBoost trained")

rf_pipeline = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))])
rf_pipeline.fit(X_train, y_train)
os.makedirs('/workspaces/airbnb-price-prediction/models/random_forest', exist_ok=True)
joblib.dump(rf_pipeline, '/workspaces/airbnb-price-prediction/models/random_forest/rf_final_pipeline.pkl')
print("✅ Random Forest trained")

knn_pipeline = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor(n_neighbors=11, weights='distance', metric='manhattan'))])
knn_pipeline.fit(X_train, y_train)
os.makedirs('/workspaces/airbnb-price-prediction/models/knn', exist_ok=True)
joblib.dump({'pipeline': knn_pipeline}, '/workspaces/airbnb-price-prediction/models/knn/knn_final_pipeline.pkl')
print("✅ KNN trained")

lr_pipeline = Pipeline([('scaler', RobustScaler()), ('model', Ridge(alpha=1.0))])
lr_pipeline.fit(X_train, y_train)
os.makedirs('/workspaces/airbnb-price-prediction/models/linear_regression', exist_ok=True)
joblib.dump(lr_pipeline, '/workspaces/airbnb-price-prediction/models/linear_regression/lr_final_pipeline.pkl')
print("✅ Linear Regression trained\n")

print("STEP 3: Running 5 tests...\n")
print("=" * 70)

tests = [
    {"name": "Test 1: Standard Listing", "data": [0, 2, 1, 4, 1, 5, 1, 0.75, 1, 4.5]},
    {"name": "Test 2: Premium - Marais", "data": [0, 3, 2, 6, 1, 8, 1, 0.90, 1, 4.8]},
    {"name": "Test 3: Private Room", "data": [1, 1, 1, 2, 1, 12, 0, 0.60, 0, 4.0]},
    {"name": "Test 4: Shared Room", "data": [2, 1, 1, 1, 1, 20, 0, 0.30, 0, 3.5]},
    {"name": "Test 5: Luxury", "data": [0, 4, 3, 8, 1, 2, 1, 0.95, 1, 4.9]}
]

for t in tests:
    X_raw = np.array(t["data"]).reshape(1, -1)
    X_70 = np.array(build_feature_vector(t["data"])).reshape(1, -1)
    
    xgb_pred = np.expm1(xgb_model.predict(X_raw)[0])
    rf_pred = np.expm1(rf_pipeline.predict(X_70)[0])
    knn_pred = np.expm1(knn_pipeline.predict(X_70)[0])
    lr_pred = np.expm1(lr_pipeline.predict(X_70)[0])
    avg = (xgb_pred + rf_pred + knn_pred + lr_pred) / 4
    
    print(f"{t['name']}")
    print(f"  XGBoost:       €{xgb_pred:.2f}/night")
    print(f"  RandomForest:  €{rf_pred:.2f}/night")
    print(f"  KNN:           €{knn_pred:.2f}/night")
    print(f"  LinearReg:     €{lr_pred:.2f}/night")
    print(f"  AVERAGE:       €{avg:.2f}/night\n")

print("=" * 70)
print("✅ ALL TESTS COMPLETE")
print("=" * 70)
