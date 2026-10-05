# XGBoost Model: Airbnb Paris Price Prediction

A tuned XGBoost regressor that predicts the nightly price (in euros) of a **new** Airbnb listing in Paris, using only information a host knows when creating the listing.

## Final model

| Item | Value |
|---|---|
| Model | `XGBRegressor` |
| Target | `log1p(price)` (convert back with `np.expm1`) |
| Features | 51 (set C): no review, occupancy or availability columns |
| Scaling | None (not needed for trees) |
| Validation | 5-fold CV, shuffled, `random_state=42` (same folds as the KNN model) |
| Saved file | `models/xgboost_final.pkl` |

**Hyperparameters** (best of 40 random combinations × 5 folds):

```python
n_estimators=800, learning_rate=0.02, max_depth=10, min_child_weight=1,
subsample=0.6, colsample_bytree=0.5, reg_lambda=1, reg_alpha=1,
tree_method="hist", random_state=42
```

## Results

| Model (set C, 51 features) | CV RMSE (log) | Test RMSE (log) | Test R² | Test MAE (€) |
|---|---|---|---|---|
| Baseline (default settings) | 0.3848 | 0.3743 | 0.759 | 80.27 |
| **Tuned (final)** | **0.3743 ± 0.0067** | **0.3626** | **0.774** | **77.13** |

The test set was used only for reporting, never for choosing features or settings.

## Why 51 features, and why the R² is 0.77 and not 0.82

The notebook compares three feature sets. The higher scores come from columns that do not exist for a new listing:

| Feature set | Features | Model | Test R² | Test MAE (€) |
|---|---|---|---|---|
| A: all columns | 70 | default | 0.828 | 68.97 |
| B: minus review and occupancy columns | 56 | tuned | 0.828 | 68.61 |
| **C: minus review, occupancy and availability (final)** | **51** | **tuned** | **0.774** | **77.13** |

Review, occupancy and availability information only appears after a listing has had guests. A pricing tool for a new listing cannot use it, so the final model uses set C. Removing the availability columns raises CV RMSE from 0.3266 to 0.3743 (about 15%), and 0.3743 is the realistic estimate for a new listing.

## What drives the predictions

- `minimum_nights` is the strongest feature. Listings with a 30+ night minimum stay have a median price of about €81, against €185 to €239 for shorter stays.
- Location (`latitude`, `longitude`) and size (`accommodates`, `bedrooms`) come next. Correlated size features share importance, so no single one looks as strong as the group.

## Limitations

- The most expensive quarter of listings is under-predicted by about €129 on average and produces about 59% of the total error.
- The train/CV gap is large (train R² 0.905 vs CV R² 0.759), and tuning increased it. The model fits the training data much more closely than new data.
- Random search tried 40 of 20,736 combinations, and six of the eight best values sit at the edge of the search space, so a wider search might do better. The best-of-40 CV score is slightly optimistic.
- Availability may partly be a legitimate host-calendar signal, so dropping it may cost more than a pure leakage fix would.
- `minimum_nights` is a host rule that acts as a proxy for listing type, so a pricing tool needs it as an input.
- Preprocessing issues outside this model, reported to the team: median imputation before the train/test split, and listings above the 99th price percentile removed. One random train/test split was used.

## Train and save the model

Run from the repo root:

```powershell
pip install xgboost scikit-learn pandas numpy joblib
python models\xgboost_final_pipeline.py
```

Options:

- `--skip-cv` skips the 5-fold CV for a faster run.
- `--data <folder>` sets the folder with `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` (default: `data/processed`).
- `--out <path>` sets the output file (default: `models/xgboost_final.pkl`).

The script trains on the full training set, prints test metrics, saves the bundle, and checks that the reloaded model predicts exactly like the one in memory. Expected test metrics: RMSE 0.3626, R² 0.7736, MAE about €77.

## Use the saved model

```python
import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("models/xgboost_final.pkl")
model = bundle["model"]

# Start from the median listing, then overwrite the fields you know
row = pd.DataFrame([bundle["default_row"]])[bundle["input_columns"]]

price_eur = np.expm1(model.predict(row)[0])
print(f"Predicted price: €{price_eur:.0f} per night")
```

The bundle contains:

| Key | Description |
|---|---|
| `model` | trained XGBRegressor |
| `input_columns` | exact column order the model expects |
| `default_row` | median of each feature, for fields the user leaves blank |
| `value_ranges` | min and max of each feature in the training data |
| `one_hot_groups` | dummy columns for room type and neighbourhood |
| `reference_categories` | categories dropped in encoding (all dummies are 0 for these) |
| `test_rmse`, `test_r2`, `test_mae_eur`, `cv_rmse` | evaluation metrics (`cv_rmse` is `None` if `--skip-cv` was used) |
| `xgboost_version`, `sklearn_version` | library versions used for training |

Load the file with the same `xgboost` and `scikit-learn` versions stored in the bundle.

## Possible next steps

- Widen the search space and re-tune.
- Correct the log-to-euro bias, or model price quantiles.
- Add features that describe listing quality.
- Treat mid-term rentals (30+ nights) as a separate model.