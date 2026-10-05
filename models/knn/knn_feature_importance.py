"""Calculate feature importance for the trained KNN pipeline."""

import joblib
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

# Load the trained KNN model bundle
bundle = joblib.load("knn_final_pipeline.pkl")
knn_pipeline = bundle["pipeline"]

# Load training data for reference
X_train = pd.read_csv("../../data/processed/X_train.csv")
X_test = pd.read_csv("../../data/processed/X_test.csv")
y_test = pd.read_csv("../../data/processed/y_test.csv")["log_price"]

# Calculate permutation importance on test set (smaller, faster)
print("Calculating KNN feature importance...")
result = permutation_importance(
    knn_pipeline, 
    X_test,  # Use full test set (smaller than train)
    y_test, 
    n_repeats=5, 
    random_state=42, 
    n_jobs=1  # Single-threaded for Windows
)

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 15 KNN FEATURES (Permutation Importance)")
print("="*60)
print(importance_df.head(15).to_string())

# Save results
importance_df.to_csv("knn_feature_importance.csv", index=False)
print("\nSaved to knn_feature_importance.csv")