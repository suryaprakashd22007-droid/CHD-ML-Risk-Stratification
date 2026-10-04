"""
Exploratory Data Analysis (EDA) Module.

Provides helper routines and summary functions for the Streamlit UI:
- Dataset shape, missing values summary, and descriptive statistics
- Distribution metrics for continuous and categorical attributes
- Correlation matrices
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Returns high-level dataset metrics: total rows, total columns,
    missing value count per column, and data types.
    """
    summary = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_counts": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return summary


def get_target_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns counts and percentages for binary target:
    0: No Heart Disease
    1: Heart Disease Present
    """
    if "target" not in df.columns:
        return pd.DataFrame()

    counts = df["target"].value_counts().reset_index()
    counts.columns = ["Target", "Count"]
    counts["Outcome"] = counts["Target"].map({0: "No Heart Disease", 1: "Heart Disease Present"})
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(1)
    return counts


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation across all numerical features and the target.
    Missing values in ca/thal are filled with median prior to calculation.
    """
    features_to_corr = [col for col in df.columns if col != "diagnosis"]
    sub_df = df[features_to_corr].copy()
    sub_df = sub_df.fillna(sub_df.median(numeric_only=True))
    return sub_df.corr().round(2)
