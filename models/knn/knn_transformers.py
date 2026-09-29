"""Custom scikit-learn transformers for the KNN price-prediction pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_REFERENCE_GROUPS = {
    "room_": "room_Entire home/apt",
    "nbhd_": "nbhd_Batignolles-Monceau",
}
DEFAULT_CAP_COLUMNS = [
    "bathrooms",
    "bedrooms",
    "beds",
    "host_listings_count",
    "calculated_host_listings_count",
]


class RestoreReferenceCategories(TransformerMixin, BaseEstimator):
    def __init__(self, groups=None):
        self.groups = groups

    def _restore(self, X):
        X = X.copy()
        for col in X.select_dtypes(include="bool").columns:
            X[col] = X[col].astype(int)
        for prefix, dropped in self.groups_.items():
            if dropped not in X.columns:
                group_cols = [c for c in X.columns if c.startswith(prefix)]
                X[dropped] = 1 - X[group_cols].sum(axis=1)
                if (X[dropped] < 0).any():
                    raise ValueError(
                        f"More than one '{prefix}*' column is 1 in some row, "
                        "so the reference category cannot be restored."
                    )
        return X

    def fit(self, X, y=None):
        self.groups_ = (
            dict(self.groups) if self.groups is not None
            else dict(DEFAULT_REFERENCE_GROUPS)
        )
        self.input_columns_ = list(X.columns)
        self.feature_names_out_ = list(self._restore(X).columns)
        return self

    def transform(self, X):
        restored = self._restore(X)
        missing = [c for c in self.feature_names_out_ if c not in restored.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} required column(s): {missing[:5]}..."
            )
        return restored[self.feature_names_out_]


class OutlierCapper(TransformerMixin, BaseEstimator):
    def __init__(self, columns=None, iqr_multiplier=1.5,
                 fallback_quantiles=(0.01, 0.99)):
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier
        self.fallback_quantiles = fallback_quantiles

    def fit(self, X, y=None):
        wanted = self.columns if self.columns is not None else DEFAULT_CAP_COLUMNS
        self.columns_ = [c for c in wanted if c in X.columns]
        self.bounds_ = {}
        self.used_fallback_ = []
        for col in self.columns_:
            values = X[col].dropna()
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            lower = max(q1 - self.iqr_multiplier * iqr, 0)
            upper = q3 + self.iqr_multiplier * iqr
            if upper - lower <= 0:
                low_q, high_q = self.fallback_quantiles
                lower = max(values.quantile(low_q), 0)
                upper = values.quantile(high_q)
                self.used_fallback_.append(col)
            self.bounds_[col] = (float(lower), float(upper))
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower, upper)
        return X
