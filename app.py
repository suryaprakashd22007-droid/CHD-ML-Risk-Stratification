"""
Main Streamlit Application for Coronary Heart Disease Early Detection System.

Handles:
- Page configuration and responsive layout
- Dynamic loading of trained model metadata
- Sidebar navigation between modules:
  1. Dashboard
  2. Patient Risk Prediction
  3. Dataset Analysis
  4. Model Performance
  5. Feature Analysis
  6. Prediction History
  7. About Project
"""

import os
import json
import streamlit as st

from views.dashboard import render_dashboard
from views.prediction import render_prediction_page
from views.dataset_analysis import render_dataset_analysis_page
from views.model_performance import render_model_performance_page
from views.feature_analysis import render_feature_analysis_page
from views.prediction_history import render_prediction_history_page
from views.about import render_about_page
from database.database import create_database


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
METRICS_FILE = os.path.join(SAVED_MODELS_DIR, "model_metrics.json")
FEATURE_INFO_FILE = os.path.join(SAVED_MODELS_DIR, "feature_information.json")


def load_metadata():
    """
    Loads saved model evaluation metrics and feature descriptions from JSON.
    Returns (metrics_data, feature_info).
    """
    metrics_data = None
    feature_info = None

    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = None

    if os.path.exists(FEATURE_INFO_FILE):
        try:
            with open(FEATURE_INFO_FILE, "r") as f:
                feature_info = json.load(f)
        except Exception:
            feature_info = None

    return metrics_data, feature_info


def main():
    # 1. Setup Streamlit page configuration
    st.set_page_config(
        page_title="CHD Risk Stratification System",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Ensure database schema exists
    try:
        create_database()
    except Exception as e:
        st.sidebar.error(f"Database init warning: {e}")

    # 3. Load model metrics and dataset metadata
    metrics_data, feature_info = load_metadata()

    # 4. Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("**CHD Risk Stratification**")

    menu_options = [
        "Dashboard",
        "Patient Risk Prediction",
        "Dataset Analysis",
        "Model Performance",
        "Feature Analysis",
        "Prediction History",
        "About Project"
    ]

    choice = st.sidebar.radio("Go to:", menu_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**System Status**\n"
        f"- Pipeline: {'Ready' if metrics_data else 'Models Not Trained'}\n"
        "- Dataset: Cleveland (303 records)\n"
        "- Database: SQLite (`heart_disease.db`)"
    )

    # 5. Route to appropriate page module
    if choice == "Dashboard":
        render_dashboard(metrics_data, feature_info)
    elif choice == "Patient Risk Prediction":
        render_prediction_page(metrics_data, feature_info)
    elif choice == "Dataset Analysis":
        render_dataset_analysis_page()
    elif choice == "Model Performance":
        render_model_performance_page(metrics_data)
    elif choice == "Feature Analysis":
        render_feature_analysis_page()
    elif choice == "Prediction History":
        render_prediction_history_page()
    elif choice == "About Project":
        render_about_page()


if __name__ == "__main__":
    main()
