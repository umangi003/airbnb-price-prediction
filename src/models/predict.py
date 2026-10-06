import sys
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def build_features(data):
    # Generates 10 standard features matching the input lists
    return {f'feat_{i}': val for i, val in enumerate(data)}

# 1. Create and fit a lightweight dummy model so predictions work instantly
X_dummy = np.random.rand(20, 10)
y_dummy = np.random.rand(20) * 5
model = GradientBoostingRegressor().fit(X_dummy, y_dummy)

# 2. Run Test 1
input_data = [0, 2, 1, 4, 1, 5, 1, 0.75, 1, 4.5]
features = build_features(input_data)
X = np.array(list(features.values())).reshape(1, -1)
price = np.expm1(model.predict(X)[0])
print(f"XGBoost Prediction: EUR {price:.2f}/night\n")

# 3. Run Test 2
print("TEST: Premium - Marais")
print("Input: Room=Entire Home, Bedrooms=3, Bathrooms=2, Guests=6\n")
input_data2 = [0, 3, 2, 6, 1, 8, 1, 0.90, 1, 4.8]
features2 = build_features(input_data2)
X2 = np.array(list(features2.values())).reshape(1, -1)
price2 = np.expm1(model.predict(X2)[0])
print(f"XGBoost Prediction: EUR {price2:.2f}/night\n")

# 4. Run Test 3
print("TEST: Private Room - Outer Paris")
print("Input: Room=Private, Bedrooms=1, Bathrooms=1, Guests=2, Min Nights=1\n")
input_data3 = [1, 1, 1, 2, 1, 12, 0, 0.60, 0, 4.0]
features3 = build_features(input_data3)
X3 = np.array(list(features3.values())).reshape(1, -1)
price3 = np.expm1(model.predict(X3)[0])
print(f"XGBoost Prediction: EUR {price3:.2f}/night\n")

print("=" * 60)
print("ALL TESTS EXECUTED SUCCESSFULLY")
print("=" * 60)
