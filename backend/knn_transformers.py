"""Custom scikit-learn transformers for the KNN price-prediction pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#VALIDATION STRATEGY - SCIKIT-LEARN STANDARD API INTEGRATION - BASEESTIMATOR, TRANSFORMERMIXIN



# Dictionary defining dummy variable groups and their dropped baseline/reference columns
DEFAULT_REFERENCE_GROUPS = {
    "room_": "room_Entire home/apt",
    "nbhd_": "nbhd_Batignolles-Monceau",
}

# List of numerical feature names targeted for outlier capping by default
DEFAULT_CAP_COLUMNS = [
    "bathrooms",
    "bedrooms",
    "beds",
    "host_listings_count",
    "calculated_host_listings_count",
]

#TECHNICAL DECISIONS/Optimization Decisions - RESTORE DROPPED LOCATION OF AIRBNB AND ROOMTYPE
class RestoreReferenceCategories(TransformerMixin, BaseEstimator):
    """Transformer to reconstruct dropped reference dummy columns and cast booleans."""
    
    def __init__(self, groups=None):
        # Store user-provided group mapping (defaults to DEFAULT_REFERENCE_GROUPS if None)
        self.groups = groups

    def _restore(self, X):
        """Helper method to convert boolean types and calculate missing dummy variables."""
        # Create a shallow copy of the DataFrame to prevent modifying original input data in place
        X = X.copy()
        
        # Convert all boolean columns to integer representations (1 and 0) for model compatibility
        for col in X.select_dtypes(include="bool").columns:
            X[col] = X[col].astype(int)
            
        # Iterate over each dummy group prefix and its designated reference column
        for prefix, dropped in self.groups_.items():
            # Only reconstruct the reference column if it does not already exist in the DataFrame
            if dropped not in X.columns:
                # Find all active dummy columns belonging to the current prefix group
                group_cols = [c for c in X.columns if c.startswith(prefix)]
                
                # Deduce the reference column value: 1 minus the sum of all other group dummies
                X[dropped] = 1 - X[group_cols].sum(axis=1)
                
                # Raise an error if a row has multiple 1s in the same dummy group (invalid state)
                if (X[dropped] < 0).any():
                    raise ValueError(
                        f"More than one '{prefix}*' column is 1 in some row, "
                        "so the reference category cannot be restored."
                    )
        return X

    def fit(self, X, y=None):
        """Learn and record expected input and transformed output feature schema."""
        # Initialize groups dictionary from init parameter or fallback default
        self.groups_ = (
            dict(self.groups) if self.groups is not None
            else dict(DEFAULT_REFERENCE_GROUPS)
        )
        # Store original input feature names before transformation
        self.input_columns_ = list(X.columns)
        
        # Determine and store output column order after reference restoration
        self.feature_names_out_ = list(self._restore(X).columns)
        return self

    def transform(self, X):
        """Transform input DataFrame by restoring reference categories and ensuring column alignment."""
        # Apply restoration helper to input data
        restored = self._restore(X)
        
        # Check if any required output columns are missing from the restored DataFrame
        missing = [c for c in self.feature_names_out_ if c not in restored.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} required column(s): {missing[:5]}..."
            )
        # Return DataFrame with column order strictly matching feature_names_out_
        return restored[self.feature_names_out_]


#TECHNICAL DECISIONS - OUTLIER CAPPER - REMOVE OUTLIERS AND HAVE FALL BACK OUTLIER IF CAPPING FAILS
# -bathrooms, bedrooms, beds, host_listings_count, calculated_host_listings_count
class OutlierCapper(TransformerMixin, BaseEstimator):
    """Transformer that caps extreme values in numeric columns using IQR or quantile bounds."""
    
    def __init__(self, columns=None, iqr_multiplier=1.5,
                 fallback_quantiles=(0.01, 0.99)):
        # Configuration parameters for targeted columns, IQR multiplier, and fallback quantiles
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier
        self.fallback_quantiles = fallback_quantiles

    def fit(self, X, y=None):
        """Compute lower and upper capping boundaries for each specified column from training data."""
        # Determine target columns present in input data (defaults to DEFAULT_CAP_COLUMNS if None)
        wanted = self.columns if self.columns is not None else DEFAULT_CAP_COLUMNS
        self.columns_ = [c for c in wanted if c in X.columns]
        
        self.bounds_ = {}          # Dictionary mapping column names to (lower, upper) float bounds
        self.used_fallback_ = []   # List tracking columns where IQR range collapsed and fallback was used

        #Optimization Decisions-calculates lower and upper capping limits
        for col in self.columns_:
            # Drop missing values to compute accurate statistical distribution metrics
            values = X[col].dropna()
            
            # Calculate 25th (Q1) and 75th (Q3) percentiles along with Interquartile Range (IQR)
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            
            # Set lower bound (constrained to >= 0 for non-negative count features) and upper bound
            lower = max(q1 - self.iqr_multiplier * iqr, 0)
            upper = q3 + self.iqr_multiplier * iqr
            
            # Check if IQR collapsed (range is zero/negative) due to heavily skewed or sparse data
            if upper - lower <= 0:
                low_q, high_q = self.fallback_quantiles
                # Calculate bounds using direct percentile fallbacks instead
                lower = max(values.quantile(low_q), 0)
                upper = values.quantile(high_q)
                self.used_fallback_.append(col)
                
            # Store calculated bounds as floats for serialization compatibility
            self.bounds_[col] = (float(lower), float(upper))
        return self

    def transform(self, X):
        """Cap values outside learned (lower, upper) boundaries for each target column."""
        # Create a shallow copy of DataFrame to avoid modifying caller's data in place
        X = X.copy()
        
        # Clip column values to learned minimum and maximum bounds
        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower, upper)
        return X