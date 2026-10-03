"""Final KNN pipeline for the Airbnb Paris price-prediction project.

Imports RestoreReferenceCategories and OutlierCapper from knn_transformers.py
(a real, separate .py file) instead of defining them here. This is required
so the saved .pkl can be loaded by a different process later (e.g. the
Stage 9 backend) — joblib only saves a reference to where a class lives, not
its code, so the class has to exist in an importable module, not inline in
whichever script happened to train the model.

Requires, in the SAME folder:
  - knn_transformers.py
  - processed.zip (from the group's data_profile.ipynb)
"""

import glob
import os
import shutil
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Import custom transformers and column definitions from the external module
from knn_transformers import (
    DEFAULT_CAP_COLUMNS,
    OutlierCapper,
    RestoreReferenceCategories,
)

# Global constants for reproducible experiments and file path configurations
RANDOM_STATE = 42
ZIP_PATH = "processed.zip"
EXTRACT_DIR = "processed_extracted"
MODEL_PATH = "knn_final_pipeline.pkl"

# Set True whenever you upload a NEW processed.zip, so a stale extracted
# folder from an earlier run is never reused.
FORCE_RE_EXTRACT = True


def find_file(filename):
    """Return the path of `filename` anywhere inside the extracted zip folder.

    Raises FileNotFoundError if the file is not in the zip.
    """
    # Search recursively inside EXTRACT_DIR for any file matching the requested filename
    matches = glob.glob(os.path.join(EXTRACT_DIR, "**", filename), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Couldn't find '{filename}' anywhere inside {ZIP_PATH}. "
            "Check the zip actually contains this file."
        )
    # Return the first matching file path found
    return matches[0]


def rmse(y_true, y_pred):
    """Return the root mean squared error between true and predicted values."""
    # Compute the square root of the mean squared error and convert the scalar to float
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pipeline(knn):
    """Return the full pipeline (preprocessing + the given KNN) as one object.

    Steps: restore reference categories -> cap outliers -> scale -> KNN.
    Because it is one estimator, GridSearchCV refits every step inside each fold.
    """
    # Construct an end-to-end sklearn Pipeline sequentially chaining custom and standard steps
    return Pipeline([
        ("restore", RestoreReferenceCategories()),             # Reconstructs baseline/dummy categories
        ("cap", OutlierCapper(columns=DEFAULT_CAP_COLUMNS)),  # Caps extreme feature outliers
        ("scale", RobustScaler()),                             # Scales features robustly against residual outliers
        ("knn", knn),                                          # KNN regressor model step
    ])


def describe_change(label, before, after, noise=None):
    """Return a sentence saying whether RMSE went down (better) or up (worse)."""
    # Measure change directionally (a positive result indicates RMSE reduction/improvement)
    change = before - after
    if change > 0:
        text = f"{label}: RMSE improved (fell) by {change:.4f} ({before:.4f} -> {after:.4f})."
    elif change < 0:
        text = f"{label}: RMSE got WORSE (rose) by {-change:.4f} ({before:.4f} -> {after:.4f})."
    else:
        text = f"{label}: RMSE unchanged at {before:.4f}."
        
    # Append a warning if the magnitude of change is lower than expected fold-to-fold CV variance
    if noise is not None and abs(change) < noise:
        text += (f" This is smaller than the CV fold-to-fold spread ({noise:.4f}),"
                 " so it may just be noise.")
    return text


# ============================================
# 1. EXTRACT AND LOAD FROM processed.zip
# ============================================
if os.path.exists(ZIP_PATH):
    # Local testing convenience: a processed.zip exists, extract it as before
    if FORCE_RE_EXTRACT and os.path.exists(EXTRACT_DIR):
        # Remove previous extraction folder to guarantee fresh data state
        shutil.rmtree(EXTRACT_DIR)
        print(f"Cleared old '{EXTRACT_DIR}/' - re-extracting from {ZIP_PATH}.")
    if not os.path.exists(EXTRACT_DIR):
        # Extract all contents from the zip archive into the specified destination folder
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        print(f"Extracted {ZIP_PATH} to {EXTRACT_DIR}/")
else:
    # Fall back to root directory if no zip archive is present
    EXTRACT_DIR = "."
    print(f"No {ZIP_PATH} found — reading CSVs directly from the repo "
          f"(e.g. data/processed/) instead.")

# Load raw training and testing features and target vectors from CSVs
X_train = pd.read_csv(find_file("X_train.csv"))
X_test = pd.read_csv(find_file("X_test.csv"))
y_train = pd.read_csv(find_file("y_train.csv"))["log_price"]
y_test = pd.read_csv(find_file("y_test.csv"))["log_price"]
print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

# Assert that data files contain no unexpected NaN/missing values before proceeding
assert X_train.isnull().sum().sum() == 0, "Missing values in X_train!"
assert X_test.isnull().sum().sum() == 0, "Missing values in X_test!"
print("Verified: zero missing values in X_train and X_test.")


# ============================================
# 2. COMPARISON: GROUP'S ORIGINAL SCALED FILE (k=5)
# ============================================
try:
    # Attempt to load legacy scaled datasets for benchmark evaluation
    X_tr_orig = pd.read_csv(find_file("X_train_scaled.csv"))
    X_te_orig = pd.read_csv(find_file("X_test_scaled.csv"))
    
    # Fit a standard un-tuned KNN model directly on pre-scaled legacy data
    knn_orig = KNeighborsRegressor(n_neighbors=5).fit(X_tr_orig, y_train)
    rmse_group_orig = rmse(y_test, knn_orig.predict(X_te_orig))
    print(f"\nGroup's original scaled data (k=5): RMSE = {rmse_group_orig:.4f} "
          f"({X_tr_orig.shape[1]} columns)")
except FileNotFoundError:
    # If legacy files aren't in the archive, skip this step without failing execution
    rmse_group_orig = None
    print("\nOriginal scaled files not found - skipping that comparison.")


# ============================================
# 3. CROSS-VALIDATED HYPERPARAMETER SEARCH ON THE FULL PIPELINE
# ============================================
# Define a 5-fold cross-validation strategy with fixed shuffling seed
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Specify candidate parameters for KNN regressor (prefixed with step name "knn__")
param_grid = {
    "knn__n_neighbors": [5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

# Configure cross-validation grid search using the full pipeline object
grid_search = GridSearchCV(
    build_pipeline(KNeighborsRegressor()),
    param_grid,
    cv=cv_strategy,
    scoring="neg_root_mean_squared_error",  # Optimized for RMSE minimization
    n_jobs=-1,                             # Parallelize fit across all local CPU cores
    verbose=2
)

# Time and run the full grid search across training splits
start = time.time()
grid_search.fit(X_train, y_train)
print(f"\nGrid search took {(time.time() - start) / 60:.1f} minutes.")

# Extract performance statistics for the best hyperparameter configuration
best_idx = grid_search.best_index_
cv_mean = -grid_search.cv_results_["mean_test_score"][best_idx]  # Invert sign back to positive RMSE
cv_std = grid_search.cv_results_["std_test_score"][best_idx]
best_params = {k.replace("knn__", ""): v for k, v in grid_search.best_params_.items()}

print(f"Best hyperparameters (5-fold shuffled CV): {best_params}")
print(f"Best CV RMSE (log_price scale): {cv_mean:.4f} +/- {cv_std:.4f} "
      "(mean +/- std across the 5 folds)")

# Retrieve the overall best fitted pipeline instance
best_pipeline = grid_search.best_estimator_


# ============================================
# 4. SANITY CHECKS ON THE FITTED PREPROCESSING
# ============================================
# Inspect learned capping bounds from the OutlierCapper transformer step
cap_step = best_pipeline.named_steps["cap"]
print("\nOutlier bounds learned from the full training set:")
for col, (low, high) in cap_step.bounds_.items():
    note = "  (percentile fallback: IQR collapsed)" if col in cap_step.used_fallback_ else ""
    print(f"  {col}: [{low:.2f}, {high:.2f}]{note}")

# Extract feature list output from the restoration transformer and check median scaling output
feature_names = best_pipeline.named_steps["restore"].feature_names_out_
scaled_train = best_pipeline[:-1].transform(X_train)  # Run features through preprocessing steps prior to model
lat_median = np.median(scaled_train[:, feature_names.index("latitude")])
print(f"Model uses {len(feature_names)} features. "
      f"Median of scaled latitude: {lat_median:.4f} (should be ~0).")


# ============================================
# 5. TEST-SET EVALUATION: BASELINE VS TUNED
# ============================================
# Fit an un-tuned baseline pipeline (k=5 default) for comparison
baseline_pipeline = build_pipeline(KNeighborsRegressor(n_neighbors=5))
baseline_pipeline.fit(X_train, y_train)
rmse_baseline = rmse(y_test, baseline_pipeline.predict(X_test))

# Evaluate tuned best estimator against held-out test data
y_pred_tuned = best_pipeline.predict(X_test)
rmse_tuned = rmse(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

# Convert log_price predictions back to native Euro values using expm1 for real-world interpretation
rmse_euros = rmse(np.expm1(y_test), np.expm1(y_pred_tuned))


# ============================================
# 6. RESULTS
# ============================================
print("\n" + "=" * 60)
print("RESULTS (log_price scale)")
print("=" * 60)
if rmse_group_orig is not None:
    print(f"Group original data (k=5):      RMSE = {rmse_group_orig:.4f}")
print(f"Corrected pipeline (k=5):       RMSE = {rmse_baseline:.4f}")
print(f"Tuned pipeline {best_params}: RMSE = {rmse_tuned:.4f}")
print(f"MAE: {mae_tuned:.4f}   R2: {r2_tuned:.4f}")
print(f"Tuned model RMSE in euros: {rmse_euros:.2f}")

print("\nObservations for the record:")
if rmse_group_orig is not None:
    print(" - " + describe_change("Preprocessing corrections (same k=5)",
                                  rmse_group_orig, rmse_baseline, noise=cv_std))
print(" - " + describe_change("Hyperparameter tuning", rmse_baseline, rmse_tuned,
                              noise=cv_std))


# ============================================
# 7. SAVE THE COMPLETE PIPELINE AND PROVE IT WORKS ON RAW ROWS
# ============================================
# Bundle model object alongside schema and runtime execution metadata into joblib dictionary
joblib.dump({
    "pipeline": best_pipeline,
    "best_params": best_params,
    "input_columns": list(X_train.columns),
    "target_transform": "log1p: apply np.expm1 to predictions to get euros",
    "sklearn_version": sklearn.__version__,
    "cv_rmse": (float(cv_mean), float(cv_std)),
    "test_rmse": float(rmse_tuned),
}, MODEL_PATH)
print(f"\nSaved {MODEL_PATH}")

# Reload model bundle from disk to confirm deployment consistency
bundle = joblib.load(MODEL_PATH)
reloaded_predictions = bundle["pipeline"].predict(X_test[bundle["input_columns"]])

# Ensure predictions generated by reloaded model match initial run output exactly
assert np.allclose(reloaded_predictions, y_pred_tuned), "Reloaded model disagrees!"
print("Reload check passed: the saved file predicts identically from raw rows.")

# Output sample prediction converted back to Euro values alongside ground truth
first_pred = float(np.expm1(reloaded_predictions[0]))
first_true = float(np.expm1(y_test.iloc[0]))
print(f"Example: first test listing predicted {first_pred:.0f} euros "
      f"(actual {first_true:.0f}).")