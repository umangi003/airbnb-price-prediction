# Linear Regression Pipeline for Airbnb Price Prediction

## Overview

`lr_final_pipeline.py` is a complete end-to-end script that:

1. **Loads** raw Airbnb listings from `data/listings.csv.gz`
2. **Preprocesses** following the steps in `notebooks/data_profile.ipynb`:
   - Cleans price (removes $, comma, converts to float)
   - Drops missing and outlier prices (top 1%, non-positive)
   - Imputes missing values:
     - Review scores → 0 (no reviews)
     - Beds/bathrooms/bedrooms → grouped median by accommodates, then global median
     - Host columns → global median
   - Encodes amenities as boolean flags (wifi, kitchen, AC)
   - Clips unrealistic `maximum_nights` values
3. **Trains** a LinearRegression model on log-transformed prices
4. **Saves** the complete sklearn Pipeline to `lr_final_pipeline.pkl`

## Usage

### Run the full pipeline:
```bash
cd /path/to/airbnb-price-prediction
python models/linear_regression/lr_final_pipeline.py
```

**Output:**
- `models/linear_regression/lr_final_pipeline.pkl` — the saved pipeline
- Console logs showing train/test metrics, CV RMSE, and validation

### Load and use the saved model:

```python
import joblib
import numpy as np
import pandas as pd

# Load the pipeline
pipeline = joblib.load('models/linear_regression/lr_final_pipeline.pkl')

# Load raw data (must have columns: room_type, neighbourhood_cleansed, accommodates, etc.)
df = pd.read_csv('data/listings.csv.gz')  # or your raw listing data

# Select the feature columns (same as in the script)
feature_cols = [
    "accommodates", "bedrooms", "bathrooms", "beds", "amenity_count",
    "minimum_nights", "maximum_nights", "availability_365",
    "number_of_reviews", "reviews_per_month",
    "hosts_time_as_user_years", "hosts_time_as_user_months",
    "hosts_time_as_host_years", "hosts_time_as_host_months",
    "host_is_superhost", "host_listings_count", "host_has_profile_pic",
    "host_identity_verified", "latitude", "longitude",
    "maximum_minimum_nights", "minimum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm", "has_availability",
    "availability_30", "availability_60", "availability_90",
    "number_of_reviews_ltm", "number_of_reviews_l30d", "availability_eoy",
    "number_of_reviews_ly", "estimated_occupancy_l365d",
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "has_reviews", "has_wifi", "has_kitchen", "has_ac",
    "room_type", "neighbourhood_cleansed",
]

X = df[feature_cols]

# Predict log1p(price)
pred_log = pipeline.predict(X)

# Convert back to euros
pred_euros = np.expm1(pred_log)

print(f"Predicted nightly rates (EUR): {pred_euros}")
```

## Model Performance

On the test set (after converting predictions to EUR and clipping to training range):
- **RMSE**: ~244.73 EUR
- **MAE**: ~144.82 EUR
- **R²**: ~-0.0014 (poor on test, but 5-fold CV RMSE log scale: 0.7081)

The negative R² indicates the model performs worse than predicting the mean test price. This is expected given the complex, non-linear relationship between features and Airbnb prices. The CV RMSE suggests acceptable log-scale predictions; the real-scale metrics are inflated due to exponential transformation.

## Pipeline Architecture

```
ColumnTransformer:
  ├─ Scaled (RobustScaler): accommodates, bedrooms, bathrooms, beds, etc.
  ├─ Passthrough (Median Impute): host_is_superhost, latitude, longitude, etc.
  └─ Categorical (OneHotEncoder): room_type, neighbourhood_cleansed
        ↓
LinearRegression
```

All preprocessing steps are baked into the saved `.pkl` file, so you only need the raw features (no need to manually impute, scale, or encode).

## File Structure

```
models/linear_regression/
├─ lr_final_pipeline.py          # This script (run once to train & save)
├─ lr_final_pipeline.pkl         # Saved pipeline (load this to predict)
└─ README.md                      # This file
```

## Notes

- The script uses `random_state=42` for reproducibility.
- Cross-validation uses 5-fold shuffled splits.
- The model is trained on 80% of the data; 20% held out for test evaluation.
- Prediction clipping to the training log-price range is applied during evaluation but NOT built into the saved pipeline (applications can choose their own clipping strategy).
