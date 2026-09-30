# KNN Regressor

KNN model for the Airbnb price prediction project.

## Files

- `knn_transformers.py` — custom preprocessing steps (kept separate so the saved model can be loaded outside this script)
- `knn_final_pipeline.py` — trains, tunes, and evaluates the model

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

- RMSE: 0.3623 (~€143)
- MAE: 0.2663
- R²: 0.7739

Fixing the preprocessing helped more than tuning did (RMSE improved 0.08 from preprocessing vs 0.02 from tuning).
