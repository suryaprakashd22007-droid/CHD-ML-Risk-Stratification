"""
Feature Analysis Module.

Extracts model-based feature importance from trained models:
- Random Forest Gini feature importances
- Logistic Regression absolute standardized coefficients
Provides human-friendly descriptions and scientific disclaimers.
"""

import pandas as pd
import numpy as np
from preprocessing.preprocessing import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS

# Notable predictors identified in literature and academic Cleveland benchmarks
NOTABLE_FEATURES = ["thalach", "cp", "thal", "ca", "oldpeak"]


def extract_feature_importance(model_pipeline, model_name: str) -> pd.DataFrame:
    """
    Extracts feature importance weights or coefficients from a trained scikit-learn Pipeline.
    Supports Random Forest and Logistic Regression.
    """
    estimator = model_pipeline.named_steps.get("model")
    if estimator is None:
        return pd.DataFrame()

    importances = None

    if hasattr(estimator, "feature_importances_"):
        # For tree-based models like Random Forest
        importances = estimator.feature_importances_
        metric_name = "Importance (Mean Decrease Impurity)"
    elif hasattr(estimator, "coef_"):
        # For linear models like Logistic Regression (use absolute magnitude)
        raw_coefs = estimator.coef_[0]
        importances = np.abs(raw_coefs)
        metric_name = "Absolute Coefficient Magnitude"
    else:
        return pd.DataFrame()

    df_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Score": importances,
        "Description": [FEATURE_DESCRIPTIONS.get(col, "") for col in FEATURE_COLUMNS]
    })

    # Sort descending by score
    df_importance = df_importance.sort_values(by="Score", ascending=False).reset_index(drop=True)
    df_importance["Metric"] = metric_name
    return df_importance
