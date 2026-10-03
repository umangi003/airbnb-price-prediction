# Random Forest Model Artifacts
This folder contains model artifacts and scripts for the Random Forest regressor.

### Model Performance (51 features, 5-Fold Cross Validation)
- **Best Hyperparameters**: `n_estimators=300`, `max_depth=25`, `max_features='sqrt'`, `min_samples_split=2`, `min_samples_leaf=1`
- **5-Fold CV RMSE (log)**: `0.3891 ± 0.0070`
- **Test RMSE (log)**: `0.3786`
- **Test $R^2$**: `0.7532`
- **Test MAE (€)**: `€81.48`

### Reproducing the Model
To generate the local `.pkl` model artifact (`random_forest_final_model.pkl`), run:
```bash
python train_rf.py
```
*(Note: `.pkl` files >100MB are excluded from git tracking per GitHub file size limits).*
