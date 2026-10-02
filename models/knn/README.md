# KNN Regressor

KNN model for the Airbnb price prediction project.

## Files

- `knn_transformers.py` — custom preprocessing steps (kept separate so the saved model can be loaded outside this script)
- `knn_final_pipeline.py` — trains, tunes, and evaluates the model
- `knn_final_pipeline.pkl` — the trained model, ready to load with `joblib.load()`

## How to run

```bash
pip install pandas numpy scikit-learn joblib
python knn_final_pipeline.py
```

If `processed.zip` is present it'll use that, otherwise it reads the CSVs straight from `data/processed/`.

## What's different about this model

The shared preprocessing works for Linear Regression and the tree models but not for KNN, since KNN relies on distance between points. This version fixes three things before training:
- Restores a one-hot category the shared pipeline drops
- Fixes outlier capping for columns where it was breaking (e.g. bathrooms)
- Scales latitude/longitude, which the shared pipeline left untouched

## Results

Best model: k=11, manhattan distance, distance-weighted.

- RMSE: 0.3782 (~€149)
- MAE: 0.2830
- R²: 0.7537

Fixing the preprocessing helped more than tuning did (RMSE improved 0.18 from preprocessing vs 0.03 from tuning).

## Loading the trained model

```python
import joblib
bundle = joblib.load("knn_final_pipeline.pkl")
model = bundle["pipeline"]
predictions = model.predict(new_data)  # new_data must have the same raw columns as X_train.csv
```

Note: this `.pkl` was saved with scikit-learn 1.9.1. If you get a version warning when loading it, check your installed scikit-learn version — it may still work, but results aren't guaranteed to match exactly on a different version.