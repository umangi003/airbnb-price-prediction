import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from pathlib import Path

# Load preprocessed data (already in data/processed/)
base = Path('data/processed')
X_train = pd.read_csv(base / 'X_train_scaled.csv')
X_test = pd.read_csv(base / 'X_test_scaled.csv')
y_train = pd.read_csv(base / 'y_train.csv').squeeze()
y_test = pd.read_csv(base / 'y_test.csv').squeeze()

print(f"✅ Data loaded: {X_train.shape}")

# 1. LINEAR REGRESSION
print("\n🔄 Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
Path('models/linear_regression').mkdir(exist_ok=True)
joblib.dump(lr, 'models/linear_regression/lr_final_pipeline.pkl')
print(f"✅ Linear Regression saved (R²: {lr.score(X_test, y_test):.4f})")

# 2. RANDOM FOREST
print("\n🔄 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
Path('models/random_forest').mkdir(exist_ok=True)
joblib.dump(rf, 'models/random_forest/rf_final_pipeline.pkl')
print(f"✅ Random Forest saved (R²: {rf.score(X_test, y_test):.4f})")

# 3. XGBOOST
print("\n🔄 Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
Path('models/xgboost').mkdir(exist_ok=True)
joblib.dump(xgb, 'models/xgboost/xgb_final_pipeline.pkl')
print(f"✅ XGBoost saved (R²: {xgb.score(X_test, y_test):.4f})")

print("\n🎉 All 3 models trained and saved!")