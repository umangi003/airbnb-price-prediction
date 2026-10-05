"""Simple test to load and use the saved Linear Regression pipeline.

This demonstrates how to load the saved model and make predictions on new data.

Usage:  python models/linear_regression/test_pipeline.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "linear_regression" / "lr_final_pipeline.pkl"
TEST_DATA = REPO_ROOT / "data" / "processed" / "X_test.csv"
TEST_LABELS = REPO_ROOT / "data" / "processed" / "y_test.csv"


def main():
    """Load the saved pipeline and demonstrate predictions."""
    print(f"Loading model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded. Steps: {list(pipeline.named_steps.keys())}")

    # Load raw listings.csv.gz and preprocess it the same way the training script did.
    # This tests the full pipeline end-to-end.
    print(f"\nLoading raw data from {REPO_ROOT / 'data' / 'listings.csv.gz'}...")
    df = pd.read_csv(REPO_ROOT / "data" / "listings.csv.gz", low_memory=False)
    print(f"✓ Loaded {len(df)} listings")

    # Apply the same cleaning/preprocessing as lr_final_pipeline.py
    import ast
    df["price_clean"] = (
        df["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).astype(float)
    )
    df = df.dropna(subset=["price_clean"]).copy()
    df = df[df["price_clean"] <= df["price_clean"].quantile(0.99)]
    df = df[df["price_clean"] > 0].copy()
    df["log_price"] = np.log1p(df["price_clean"])

    # Amenities
    def parse_amenities(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    amenities = df["amenities"].apply(parse_amenities)
    df["amenity_count"] = amenities.apply(len)
    for name, keyword in [("has_wifi", "wifi"), ("has_kitchen", "kitchen"),
                          ("has_ac", "air conditioning")]:
        df[name] = amenities.apply(lambda lst: int(any(keyword in a.lower() for a in lst)))

    # Flags to 0/1
    for col in ["host_is_superhost", "host_has_profile_pic",
                "host_identity_verified", "has_availability"]:
        df[col] = df[col].map({"t": 1, "f": 0}).fillna(0)

    # Maximum nights
    df["maximum_nights"] = df["maximum_nights"].clip(upper=1125)

    # has_reviews
    df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)

    # Fill review scores with 0
    review_cols = [
        "review_scores_rating", "reviews_per_month",
        "review_scores_checkin", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_location",
        "review_scores_communication", "review_scores_value",
    ]
    df[review_cols] = df[review_cols].fillna(0)

    # Bedrooms/beds/bathrooms
    for col in ["bedrooms", "beds", "bathrooms"]:
        df[col] = df.groupby("accommodates")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # Host columns
    for col in ["hosts_time_as_user_years", "hosts_time_as_host_months",
                "hosts_time_as_host_years", "hosts_time_as_user_months",
                "host_listings_count"]:
        df[col] = df[col].fillna(df[col].median())

    NUMERIC_COLUMNS = [
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
    ]
    CATEGORICAL_COLUMNS = ["room_type", "neighbourhood_cleansed"]
    INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    X = df[INPUT_COLUMNS]
    y = df["log_price"]

    # Split same as training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Using {len(X_test)} test samples from the same split")

    print("\n--- Pipeline Predictions ---")
    pred_log = pipeline.predict(X_test)
    pred_eur = np.expm1(pred_log)
    true_eur = np.expm1(y_test)

    print(f"First 5 predictions (log scale):  {pred_log[:5]}")
    print(f"First 5 predictions (EUR):        {pred_eur[:5]}")
    print(f"First 5 actual prices (EUR):      {true_eur.values[:5]}")
    print(f"\nMean predicted price (EUR):       €{pred_eur.mean():.2f}")
    print(f"Mean actual price (EUR):          €{true_eur.mean():.2f}")

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = np.sqrt(mean_squared_error(true_eur, pred_eur))
    mae = mean_absolute_error(true_eur, pred_eur)
    r2 = r2_score(true_eur, pred_eur)
    print(f"\nTest RMSE (EUR): €{rmse:.2f}")
    print(f"Test MAE (EUR):  €{mae:.2f}")
    print(f"Test R²:         {r2:.4f}")

    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
