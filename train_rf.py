import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
import joblib
import os

DATA = 'd:/Achintha/FDM/airbnb-price-prediction/data/processed/'
X_train = pd.read_csv(DATA + 'X_train.csv')
X_test = pd.read_csv(DATA + 'X_test.csv')
y_train = pd.read_csv(DATA + 'y_train.csv').squeeze()
y_test = pd.read_csv(DATA + 'y_test.csv').squeeze()

post_booking = [c for c in X_train.columns if 'review' in c] + ['estimated_occupancy_l365d']
cols_B = [c for c in X_train.columns if c not in post_booking]
cols_final = [c for c in cols_B if not c.startswith('availability_')]

print('Total cols in X_train:', X_train.shape[1])
print('Cols in cols_final (51 features):', len(cols_final))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Running Baseline Random Forest (51 features) ---")
rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
cv_base = cross_validate(rf_base, X_train[cols_final], y_train, cv=kf,
                         scoring={'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'},
                         return_train_score=True)
rf_base.fit(X_train[cols_final], y_train)
pred_base = rf_base.predict(X_test[cols_final])

rmse_log_base = np.sqrt(mean_squared_error(y_test, pred_base))
mae_log_base = mean_absolute_error(y_test, pred_base)
r2_base = r2_score(y_test, pred_base)
mae_eur_base = mean_absolute_error(np.expm1(y_test), np.expm1(pred_base))
rmse_eur_base = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_base)))

cv_rmse_mean_base = -cv_base['test_rmse'].mean()
cv_rmse_std_base = cv_base['test_rmse'].std()
cv_r2_base = cv_base['test_r2'].mean()
train_r2_base = cv_base['train_r2'].mean()

print(f"Base CV RMSE: {cv_rmse_mean_base:.4f} +/- {cv_rmse_std_base:.4f}")
print(f"Base CV R2: {cv_r2_base:.4f}")
print(f"Base Train R2: {train_r2_base:.4f}")
print(f"Base Test RMSE (log): {rmse_log_base:.4f}")
print(f"Base Test MAE (log): {mae_log_base:.4f}")
print(f"Base Test R2: {r2_base:.4f}")
print(f"Base Test MAE (EUR): {mae_eur_base:.2f}")
print(f"Base Test RMSE (EUR): {rmse_eur_base:.2f}")

print("\n--- Running Hyperparameter Tuning for Random Forest ---")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.5, 0.8]
}

rs_rf = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=1),
    param_distributions=param_grid,
    n_iter=15,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    verbose=1,
    n_jobs=-1
)
rs_rf.fit(X_train[cols_final], y_train)
print("Best RF Params:", rs_rf.best_params_)

rf_tuned = RandomForestRegressor(**rs_rf.best_params_, random_state=42, n_jobs=-1)
cv_tuned = cross_validate(rf_tuned, X_train[cols_final], y_train, cv=kf,
                          scoring={'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'},
                          return_train_score=True)
rf_tuned.fit(X_train[cols_final], y_train)
pred_tuned = rf_tuned.predict(X_test[cols_final])

rmse_log_tuned = np.sqrt(mean_squared_error(y_test, pred_tuned))
mae_log_tuned = mean_absolute_error(y_test, pred_tuned)
r2_tuned = r2_score(y_test, pred_tuned)
mae_eur_tuned = mean_absolute_error(np.expm1(y_test), np.expm1(pred_tuned))
rmse_eur_tuned = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_tuned)))

cv_rmse_mean_tuned = -cv_tuned['test_rmse'].mean()
cv_rmse_std_tuned = cv_tuned['test_rmse'].std()
cv_r2_tuned = cv_tuned['test_r2'].mean()
train_r2_tuned = cv_tuned['train_r2'].mean()

print(f"\nTuned CV RMSE: {cv_rmse_mean_tuned:.4f} +/- {cv_rmse_std_tuned:.4f}")
print(f"Tuned CV R2: {cv_r2_tuned:.4f}")
print(f"Tuned Train R2: {train_r2_tuned:.4f}")
print(f"Tuned Test RMSE (log): {rmse_log_tuned:.4f}")
print(f"Tuned Test MAE (log): {mae_log_tuned:.4f}")
print(f"Tuned Test R2: {r2_tuned:.4f}")
print(f"Tuned Test MAE (EUR): {mae_eur_tuned:.2f}")
print(f"Tuned Test RMSE (EUR): {rmse_eur_tuned:.2f}")

os.makedirs('d:/Achintha/FDM/airbnb-price-prediction/models/random_forest', exist_ok=True)
joblib.dump(rf_tuned, 'd:/Achintha/FDM/airbnb-price-prediction/models/random_forest/random_forest_final_model.pkl')
print("\nSaved tuned RF model to models/random_forest/random_forest_final_model.pkl")
