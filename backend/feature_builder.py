"""Build feature vectors for predictions from form inputs."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load training data statistics for reasonable defaults
def load_feature_statistics():
    """Load mean values from training data for default features."""
    data_path = Path("../data/processed/X_train_scaled.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df.mean().to_dict(), list(df.columns), df
    return {}, [], None

# Cache loaded data
_feature_means, _feature_names, _training_df = load_feature_statistics()

NEIGHBOURHOODS = [
    'Bourse', 'Buttes-Chaumont', 'Buttes-Montmartre', 'Entrepôt', 'Gobelins',
    'Hôtel-de-Ville', 'Louvre', 'Luxembourg', 'Ménilmontant', 'Observatoire',
    'Opéra', 'Palais-Bourbon', 'Panthéon', 'Passy', 'Popincourt', 'Reuilly',
    'Temple', 'Vaugirard', 'Élysée'
]

ROOM_TYPES = ['Entire Home', 'Private Room', 'Shared Room', 'Hotel Room']

def build_feature_vector(form_data):
    """
    Convert form inputs to full feature vector.

    Form inputs: [room_type_idx, bedrooms, bathrooms, accommodates, minimum_stay,
                  neighbourhood_idx, ac, availability_90, is_superhost, review_rating]

    Returns: list of 70 features in proper order
    """
    room_type_idx, bedrooms, bathrooms, accommodates, minimum_stay, nbhd_idx, ac, availability_90, is_superhost, review_rating = form_data

    # Start with means from training data
    features = dict(_feature_means)

    # Override with form inputs - core listing features
    features['accommodates'] = accommodates
    features['bathrooms'] = bathrooms
    features['bedrooms'] = bedrooms
    features['beds'] = bedrooms
    features['minimum_nights'] = minimum_stay
    features['availability_90'] = availability_90 * 100

    # Room type one-hot encoding (following the CSV structure)
    # Room types in CSV: Hotel room (0), Private room (1), Shared room (2)
    # Form maps: Entire Home (0), Private Room (1), Shared Room (2), Hotel Room (3)
    # So: Entire Home -> Hotel room, Hotel Room -> Hotel room
    features['room_Hotel room'] = 1.0 if room_type_idx in [0, 3] else 0.0
    features['room_Private room'] = 1.0 if room_type_idx == 1 else 0.0
    features['room_Shared room'] = 1.0 if room_type_idx == 2 else 0.0

    # Neighbourhood one-hot encoding
    for i, nbhd in enumerate(NEIGHBOURHOODS):
        nbhd_col = f'nbhd_{nbhd}'
        features[nbhd_col] = 1.0 if i == nbhd_idx else 0.0

    # Air conditioning: maps to has_ac
    if 'has_ac' in features:
        features['has_ac'] = 1.0 if ac else 0.0

    # Host features - NEW
    if 'host_is_superhost' in features:
        features['host_is_superhost'] = 1.0 if is_superhost else 0.0

    # Review rating - NEW
    if 'review_scores_rating' in features:
        features['review_scores_rating'] = review_rating

    # Compute derived features if they exist
    if 'accommodates' in features and 'bedrooms' in features:
        # These might be calculated features
        pass

    # Build output list in correct order
    if _feature_names:
        output = [features.get(fname, 0.0) for fname in _feature_names]
    else:
        # Fallback if we can't load feature names
        output = list(features.values())

    return output