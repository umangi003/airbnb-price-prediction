import joblib
import numpy as np
import pandas as pd

print("=" * 70)
print("XGBOOST SYSTEM TEST - REAL TEST DATA")
print("=" * 70)
print()

xgb = joblib.load('models/xgboost/xgb_final_pipeline.pkl')
print("OK XGBoost model loaded\n")

X_test = pd.read_csv('data/processed/X_test_scaled.csv')
print(f"OK Test data loaded: {X_test.shape}\n")

print("=" * 70)
print("TEST RESULTS (5 Real Test Cases)")
print("=" * 70)
print()

for i in range(5):
    X_sample = X_test.iloc[i:i+1]
    log_pred = xgb.predict(X_sample)[0]
    price = np.expm1(log_pred)
    print(f"Test {i+1}: EUR{price:.2f}/night")

print()
print("=" * 70)
print("OK ALL TESTS COMPLETE")
print("=" * 70)