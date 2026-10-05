"""Final XGBoost model for the Airbnb Paris price-prediction project.

Trains the tuned XGBoost on the 51 features that are known when a NEW listing
is priced (no review, occupancy or availability columns), evaluates it, and
saves everything the backend needs in one file: models/xgboost_final.pkl

The hyperparameters below are the best settings from the RandomizedSearchCV in
xgboost_model.ipynb (40 combinations x 5 folds, set C). The search is not
repeated here, so this script runs in a few minutes.

Usage (from the repo root):
    python xgboost_final_pipeline.py
    python xgboost_final_pipeline.py --skip-cv        # faster, skips the 5-fold CV
    python xgboost_final_pipeline.py --data data/processed --out models/xgboost_final.pkl
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

RANDOM_STATE = 42

# Best settings found by the random search on set C (see Experiment 2 in the notebook)
BEST_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.02,
    "max_depth": 10,
    "min_child_weight": 1,
    "subsample": 0.6,
    "colsample_bytree": 0.5,
    "reg_lambda": 1,
    "reg_alpha": 1,
}

# Dummy columns dropped by the shared preprocessing (reference categories).
# Taken from the KNN transformers file; if the user picks one of these, all
# the other dummies of that group are 0.
REFERENCE_CATEGORIES = {"room": "Entire home/apt", "nbhd": "Batignolles-Monceau"}


def find_data_dir(cli_value):
    """Return the folder containing X_train.csv, trying a few likely places."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [cli_value] if cli_value else []
    candidates += ["data/processed", "../data/processed",
                   os.path.join(here, "data", "processed"),
                   os.path.join(here, "..", "data", "processed")]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "X_train.csv")):
            return path
    sys.exit("Could not find X_train.csv. Pass the folder with --data.")


def select_final_features(columns):
    """The 51 features known at pricing time for a new listing (set C)."""
    post_booking = [c for c in columns if "review" in c] + ["estimated_occupancy_l365d"]
    return [c for c in columns
            if c not in post_booking and not c.startswith("availability_")]


def make_model():
    return XGBRegressor(random_state=RANDOM_STATE, tree_method="hist",
                        n_jobs=-1, **BEST_PARAMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="folder with X_train.csv etc.")
    parser.add_argument("--out", default="models/xgboost_final.pkl")
    parser.add_argument("--skip-cv", action="store_true", help="skip the 5-fold CV")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data)
    print("Reading data from", data_dir)
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv")).astype(float)
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv")).astype(float)
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["log_price"]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["log_price"]
    print(f"Loaded X_train {X_train.shape}, X_test {X_test.shape}")

    cols = select_final_features(list(X_train.columns))
    print(f"Using {len(cols)} features (expected 51)")
    assert len(cols) == 51, f"Expected 51 features, got {len(cols)}"
    assert X_test[cols].isnull().sum().sum() == 0, "Missing values in X_test!"

    room_cols = [c for c in cols if c.startswith("room_")]
    nbhd_cols = [c for c in cols if c.startswith("nbhd_")]
    for group, ref in REFERENCE_CATEGORIES.items():
        ref_col = f"{group}_{ref}"
        if ref_col in cols:
            print(f"WARNING: reference column {ref_col} is present in the features; "
                  "check REFERENCE_CATEGORIES.")

    # ---- 5-fold CV (same folds as the other models) ----
    cv_rmse = cv_std = None
    if not args.skip_cv:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv = cross_validate(make_model(), X_train[cols], y_train, cv=kf,
                            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                            return_train_score=True)
        cv_rmse = float(-cv["test_rmse"].mean())
        cv_std = float(cv["test_rmse"].std())
        print(f"CV RMSE (log): {cv_rmse:.4f} +/- {cv_std:.4f}   "
              f"CV R2: {cv['test_r2'].mean():.4f}   Train R2: {cv['train_r2'].mean():.4f}")

    # ---- Final fit on the full training set, evaluate on the test set ----
    model = make_model().fit(X_train[cols], y_train)
    pred = model.predict(X_test[cols])
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    test_r2 = float(r2_score(y_test, pred))
    test_mae_eur = float(mean_absolute_error(np.expm1(y_test), np.expm1(pred)))
    print(f"Test RMSE (log): {test_rmse:.4f}   Test R2: {test_r2:.4f}   "
          f"Test MAE: EUR {test_mae_eur:.2f}")
    print("Notebook values for comparison: CV 0.3743 +/- 0.0067, test RMSE 0.3626, "
          "R2 0.7736, MAE EUR 77.13")

    # ---- Save everything the backend needs ----
    bundle = {
        "model": model,
        "input_columns": cols,                                   # exact order the model expects
        "default_row": X_train[cols].median().to_dict(),         # fallback for fields the form skips
        "value_ranges": {c: (float(X_train[c].min()), float(X_train[c].max())) for c in cols},
        "one_hot_groups": {"room": room_cols, "nbhd": nbhd_cols},
        "reference_categories": REFERENCE_CATEGORIES,
        "target_transform": "log1p: apply np.expm1 to the prediction to get euros per night",
        "best_params": BEST_PARAMS,
        "cv_rmse": (cv_rmse, cv_std),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae_eur": test_mae_eur,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\nSaved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    # ---- Reload check: the saved file must predict exactly like the model in memory ----
    b = joblib.load(args.out)
    reloaded = b["model"].predict(X_test[b["input_columns"]])
    assert np.allclose(reloaded, pred), "Reloaded model disagrees!"
    print("Reload check passed.")
    print(f"Example: first test listing predicted {np.expm1(reloaded[0]):.0f} euros "
          f"(actual {np.expm1(y_test.iloc[0]):.0f}).")


if __name__ == "__main__":
    main()