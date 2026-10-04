"""
Training Script for Coronary Heart Disease Risk Stratification.

This script executes the complete machine learning workflow:
1. Loads the actual Cleveland dataset (data/heart-disease-cleveland.csv)
2. Cleans missing values ('?' -> NaN)
3. Creates the binary target (0 = No Heart Disease, 1 = Heart Disease Present)
4. Splits the dataset using stratified 80/20 train/test split
5. Trains and tunes 3 models using GridSearchCV:
   - Logistic Regression
   - Support Vector Machine (SVC)
   - Random Forest Classifier
6. Evaluates all models on the held-out test set
7. Selects the best screening model prioritizing Recall, ROC-AUC, and F1-score
8. Saves all model artifacts, metrics, and metadata to models/saved_models/
"""

import os
import json
import joblib
import pandas as pd

from preprocessing.preprocessing import (
    load_dataset,
    clean_dataset,
    create_target,
    split_data,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS
)
from models.model_training import train_and_tune_models
from models.model_evaluation import evaluate_model


SAVED_MODELS_DIR = os.path.join("models", "saved_models")


def select_best_model(comparison_df: pd.DataFrame) -> tuple:
    """
    Selects the best model for a clinical early-detection/screening decision support system.
    In disease screening, missing a positive case (False Negative) is the highest risk.
    Therefore, selection criteria prioritizes:
    1. Recall (Sensitivity)
    2. ROC-AUC (Overall discrimination capability across thresholds)
    3. F1-Score (Harmonic balance of precision and recall)
    """
    # Sort strictly by Recall descending, then ROC-AUC descending, then F1-Score descending
    sorted_df = comparison_df.sort_values(
        by=["Recall", "ROC-AUC", "F1-Score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    best_model_name = best_row["Model"]

    reason = (
        f"Selected {best_model_name} because it achieved the highest Recall ({best_row['Recall']:.4f}) "
        f"and strong ROC-AUC ({best_row['ROC-AUC']:.4f}) on the test set. "
        "In screening for coronary heart disease, maximizing sensitivity (Recall) is paramount "
        "to avoid missing high-risk patients who require early clinical intervention."
    )

    return best_model_name, reason


def save_artifacts(trained_models, evaluation_results, comparison_df, best_model_name, selection_reason, dataset_stats):
    """
    Saves trained pipeline objects and JSON summaries into models/saved_models/.
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # 1. Save specific model pipelines
    name_to_filename = {
        "Logistic Regression": "logistic_model.pkl",
        "Support Vector Machine": "svm_model.pkl",
        "Random Forest": "random_forest_model.pkl"
    }

    for model_name, data in trained_models.items():
        fname = name_to_filename.get(model_name)
        if fname:
            save_path = os.path.join(SAVED_MODELS_DIR, fname)
            joblib.dump(data["pipeline"], save_path)
            print(f"[Saved] {model_name} pipeline -> {save_path}")

    # 2. Save selected model pipeline
    best_pipeline = trained_models[best_model_name]["pipeline"]
    selected_model_path = os.path.join(SAVED_MODELS_DIR, "selected_model.pkl")
    joblib.dump(best_pipeline, selected_model_path)
    print(f"[Saved] Selected Best Model ({best_model_name}) -> {selected_model_path}")

    # 3. Save comprehensive metrics and comparison JSON
    metrics_summary = {
        "selected_model": best_model_name,
        "selection_reason": selection_reason,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "models": {}
    }

    for model_name, eval_metrics in evaluation_results.items():
        metrics_summary["models"][model_name] = {
            "metrics": {
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                "roc_auc": eval_metrics["roc_auc"]
            },
            "best_hyperparameters": trained_models[model_name]["best_params"],
            "cross_val_recall": trained_models[model_name]["cv_score"],
            "confusion_matrix": eval_metrics["confusion_matrix"],
            "roc_curve": eval_metrics["roc_curve"]
        }

    metrics_path = os.path.join(SAVED_MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[Saved] Model metrics & ROC data -> {metrics_path}")

    # 4. Save feature and dataset information JSON
    feature_info = {
        "dataset_statistics": dataset_stats,
        "feature_columns": FEATURE_COLUMNS,
        "feature_descriptions": FEATURE_DESCRIPTIONS
    }
    feature_path = os.path.join(SAVED_MODELS_DIR, "feature_information.json")
    with open(feature_path, "w") as f:
        json.dump(feature_info, f, indent=4)
    print(f"[Saved] Feature descriptions & dataset info -> {feature_path}")


def main():
    print("=" * 70)
    print(" MACHINE LEARNING-DRIVEN RISK STRATIFICATION FOR CORONARY HEART DISEASE")
    print(" Model Training and Evaluation Pipeline")
    print("=" * 70)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading Cleveland Heart Disease dataset...")
    raw_df = load_dataset("data/heart-disease-cleveland.csv")
    print(f"Loaded dataset: {raw_df.shape[0]} records, {raw_df.shape[1]} raw columns")

    # Step 2: Clean Dataset
    print("\n[Step 2] Cleaning missing values and parsing types...")
    cleaned_df = clean_dataset(raw_df)

    # Step 3: Create Binary Target
    print("\n[Step 3] Converting diagnosis to binary classification target (0=No Disease, 1=Disease Present)...")
    processed_df = create_target(cleaned_df)
    target_counts = processed_df["target"].value_counts().to_dict()
    print(f"Target distribution -> No Disease (0): {target_counts.get(0, 0)}, Disease Present (1): {target_counts.get(1, 0)}")

    dataset_stats = {
        "total_records": int(len(processed_df)),
        "feature_count": len(FEATURE_COLUMNS),
        "target_distribution": {
            "no_disease": int(target_counts.get(0, 0)),
            "disease_present": int(target_counts.get(1, 0))
        },
        "missing_counts_raw": {k: int(v) for k, v in raw_df.isnull().sum().items()}
    }

    # Step 4: Stratified Train/Test Split
    print("\n[Step 4] Performing stratified 80/20 train/test split...")
    train_features, test_features, train_target, test_target = split_data(
        processed_df, test_size=0.20, random_state=42
    )
    print(f"Training set: {train_features.shape[0]} samples | Testing set: {test_features.shape[0]} samples")

    # Step 5: Train and Tune Models via GridSearchCV
    print("\n[Step 5] Training and tuning 3 models with Stratified 5-Fold Cross Validation...")
    trained_models = train_and_tune_models(train_features, train_target)

    # Step 6: Evaluate Models on the held-out Test Set
    print("\n[Step 6] Evaluating models on the unseen test set...")
    comparison_rows = []
    evaluation_results = {}

    for model_name, info in trained_models.items():
        pipeline = info["pipeline"]
        eval_metrics = evaluate_model(pipeline, test_features, test_target)
        evaluation_results[model_name] = eval_metrics

        comparison_rows.append({
            "Model": model_name,
            "Accuracy": eval_metrics["accuracy"],
            "Precision": eval_metrics["precision"],
            "Recall": eval_metrics["recall"],
            "F1-Score": eval_metrics["f1"],
            "ROC-AUC": eval_metrics["roc_auc"]
        })

    comparison_df = pd.DataFrame(comparison_rows)
    print("\n--- Model Test Evaluation Results ---")
    print(comparison_df.to_string(index=False))

    # Step 7: Transparent Model Selection
    print("\n[Step 7] Selecting best model prioritizing Recall and ROC-AUC...")
    best_model_name, selection_reason = select_best_model(comparison_df)
    print(f"\n>> Selected Model: {best_model_name}")
    print(f">> Selection Reason: {selection_reason}\n")

    # Step 8: Save Model Artifacts
    print("[Step 8] Saving pipeline files and metrics to models/saved_models/...")
    save_artifacts(
        trained_models,
        evaluation_results,
        comparison_df,
        best_model_name,
        selection_reason,
        dataset_stats
    )

    print("\nTraining workflow completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
