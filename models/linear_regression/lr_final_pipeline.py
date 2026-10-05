"""Final Linear Regression pipeline for the Airbnb Paris price-prediction project.

Loads data/listings.csv.gz, reproduces the cleaning steps from
notebooks/data_profile.ipynb (which produced the data used in
notebooks/linear_regression_modeling.ipynb), trains a LinearRegression on
log1p(price) and saves the complete pipeline with joblib.

The saved pipeline only uses scikit-learn classes, so it can be loaded
anywhere without importing this script. It expects a DataFrame with the
columns in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS (see `input_columns`
printed at the end) and predicts log1p(price): apply np.expm1 to get euros.

Run from the repo root:  python models/linear_regression/lr_final_pipeline.py
"""

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

RANDOM_STATE = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "listings.csv.gz"
MODEL_PATH = REPO_ROOT / "models" / "linear_regression" / "lr_final_pipeline.pkl"

# Columns scaled with RobustScaler (same list as data_profile.ipynb).
SCALED_COLUMNS = [
    "accommodates", "bedrooms", "bathrooms", "beds", "amenity_count",
    "minimum_nights", "maximum_nights", "availability_365",
    "number_of_reviews", "reviews_per_month",
]

# Remaining numeric / 0-1 columns, passed through unscaled (as in the notebook).
PASSTHROUGH_COLUMNS = [
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

# One-hot encoded inside the pipeline (drop first level, like get_dummies(drop_first=True)).
CATEGORICAL_COLUMNS = ["room_type", "neighbourhood_cleansed"]

NUMERIC_COLUMNS = SCALED_COLUMNS + PASSTHROUGH_COLUMNS
INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def parse_amenities(value):
    """Return the amenities list, or [] when the cell is missing/malformed."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def load_and_clean(path):
    """Load the raw listings and apply the data_profile.ipynb cleaning steps.

    Returns (X, y) where y is log1p(price). Row filtering, grouped-median
    imputation and feature construction happen here because they depend on the
    target or on other rows; the per-column imputing, scaling and encoding
    live in the pipeline.
    """
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Target: clean price, drop missing, drop top 1% and non-positive prices
    df["price_clean"] = (
        df["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).astype(float)
    )
    df = df.dropna(subset=["price_clean"]).copy()
    df = df[df["price_clean"] <= df["price_clean"].quantile(0.99)]
    df = df[df["price_clean"] > 0].copy()
    df["log_price"] = np.log1p(df["price_clean"])
    print(f"Rows after price cleaning / outlier removal: {len(df)}")

    # --- Missing values: reviews
    df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)
    review_cols = [
        "review_scores_rating", "reviews_per_month",
        "review_scores_checkin", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_location",
        "review_scores_communication", "review_scores_value",
    ]
    df[review_cols] = df[review_cols].fillna(0)

    # --- Missing values: bedrooms/beds/bathrooms (median of same-size listings)
    for col in ["bedrooms", "beds", "bathrooms"]:
        df[col] = df.groupby("accommodates")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # --- Missing values: host columns
    for col in ["hosts_time_as_user_years", "hosts_time_as_host_months",
                "hosts_time_as_host_years", "hosts_time_as_user_months",
                "host_listings_count"]:
        df[col] = df[col].fillna(df[col].median())

    # --- Amenities features
    amenities = df["amenities"].apply(parse_amenities)
    df["amenity_count"] = amenities.apply(len)
    for name, keyword in [("has_wifi", "wifi"), ("has_kitchen", "kitchen"),
                          ("has_ac", "air conditioning")]:
        df[name] = amenities.apply(lambda lst: int(any(keyword in a.lower() for a in lst)))

    # --- t/f flags to 0/1
    for col in ["host_is_superhost", "host_has_profile_pic",
                "host_identity_verified", "has_availability"]:
        df[col] = df[col].map({"t": 1, "f": 0}).fillna(0)

    # --- maximum_nights has 2147483647 placeholders; cap at the 1125-night platform limit
    df["maximum_nights"] = df["maximum_nights"].clip(upper=1125)

    return df[INPUT_COLUMNS], df["log_price"]


def build_pipeline():
    """Imputation + scaling + one-hot encoding + LinearRegression as one estimator."""
    preprocessor = ColumnTransformer(
        [
            ("scaled", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]), SCALED_COLUMNS),
            ("passthrough", SimpleImputer(strategy="median"), PASSTHROUGH_COLUMNS),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                         sparse_output=False)),
            ]), CATEGORICAL_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])


def main():
    X, y = load_and_clean(DATA_PATH)

    # Same split as the notebooks (80/20, random_state=42).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    pipeline = build_pipeline()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error")
    print(f"CV RMSE (log scale): {-cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    # Evaluate like the notebook: clip predicted log prices to the training range,
    # then convert back to euros.
    pred_log = np.clip(pipeline.predict(X_test), y_train.min(), y_train.max())
    pred_eur, true_eur = np.expm1(pred_log), np.expm1(y_test)
    print(f"Test RMSE: EUR {np.sqrt(mean_squared_error(true_eur, pred_eur)):.2f}, "
          f"MAE: EUR {mean_absolute_error(true_eur, pred_eur):.2f}, "
          f"R2: {r2_score(true_eur, pred_eur):.4f}")
    print(f"Prediction clip range (log price): [{y_train.min():.4f}, {y_train.max():.4f}]")

    # Fit the final model on all data? No - keep the train-only fit so the saved
    # model matches the evaluated one.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

    reloaded = joblib.load(MODEL_PATH)
    assert np.allclose(reloaded.predict(X_test), pipeline.predict(X_test)), \
        "Reloaded pipeline disagrees!"
    print("Reload check passed.")


if __name__ == "__main__":
    main()
