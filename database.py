"""
Database Access Module for SQLite & SQLAlchemy.

Provides straightforward functions to:
1. Initialize the SQLite database tables
2. Save patient demographic, biometric, and prediction records
3. Query prediction history with patient and clinical details
"""

import os
import uuid
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.database_models import Base, Patient, ClinicalBiometrics, RiskAssessment

# Default SQLite database path in local database/ directory
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "heart_disease.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create the SQLite engine
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_database():
    """
    Initializes all database tables defined in database_models.py.
    Safe to call multiple times as it only creates tables if they don't already exist.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def generate_patient_id():
    """
    Generates a unique, non-identifying patient identifier.
    Example: PAT-7A9B1C
    """
    unique_suffix = uuid.uuid4().hex[:6].upper()
    return f"PAT-{unique_suffix}"


def save_prediction(patient_data: dict, predicted_class: int, predicted_risk: float, model_name: str) -> str:
    """
    Saves a patient record, their clinical measurements, and the ML prediction in a single transaction.
    Returns the generated patient_id.
    """
    create_database()
    session = SessionLocal()

    try:
        # Generate anonymous patient ID
        patient_id = generate_patient_id()

        # 1. Create Patient demographic row
        patient = Patient(
            patient_id=patient_id,
            age=int(patient_data["age"]),
            sex=int(patient_data["sex"]),
            created_at=datetime.utcnow()
        )
        session.add(patient)

        # 2. Create Clinical Biometrics row
        biometrics = ClinicalBiometrics(
            patient_id=patient_id,
            cp=int(patient_data["cp"]),
            trestbps=float(patient_data["trestbps"]),
            chol=float(patient_data["chol"]),
            fbs=int(patient_data["fbs"]),
            restecg=int(patient_data["restecg"]),
            thalach=float(patient_data["thalach"]),
            exang=int(patient_data["exang"]),
            oldpeak=float(patient_data["oldpeak"]),
            slope=int(patient_data["slope"]),
            ca=float(patient_data["ca"]),
            thal=float(patient_data["thal"])
        )
        session.add(biometrics)

        # 3. Create Risk Assessment row
        assessment = RiskAssessment(
            patient_id=patient_id,
            predicted_risk_score=float(predicted_risk),
            predicted_class=int(predicted_class),
            model_name=str(model_name),
            prediction_date=datetime.utcnow()
        )
        session.add(assessment)

        # Commit transaction
        session.commit()
        return patient_id

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_prediction_history() -> pd.DataFrame:
    """
    Retrieves all past predictions joined with demographic and biometric inputs.
    Returns a pandas DataFrame sorted by prediction date descending.
    """
    create_database()
    session = SessionLocal()

    try:
        # Query risk assessments joined with patient and biometrics
        query = (
            session.query(
                RiskAssessment.prediction_id,
                RiskAssessment.patient_id,
                Patient.age,
                Patient.sex,
                ClinicalBiometrics.trestbps,
                ClinicalBiometrics.chol,
                ClinicalBiometrics.thalach,
                ClinicalBiometrics.cp,
                RiskAssessment.predicted_class,
                RiskAssessment.predicted_risk_score,
                RiskAssessment.model_name,
                RiskAssessment.prediction_date
            )
            .join(Patient, RiskAssessment.patient_id == Patient.patient_id)
            .join(ClinicalBiometrics, RiskAssessment.patient_id == ClinicalBiometrics.patient_id)
            .order_by(RiskAssessment.prediction_date.desc())
        )

        results = query.all()
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            rows.append({
                "Prediction ID": r.prediction_id,
                "Patient ID": r.patient_id,
                "Age": r.age,
                "Sex": "Male" if r.sex == 1 else "Female",
                "Resting BP (mm Hg)": r.trestbps,
                "Cholesterol (mg/dl)": r.chol,
                "Max Heart Rate": r.thalach,
                "Chest Pain Type": r.cp,
                "Prediction": "Heart Disease Present" if r.predicted_class == 1 else "No Heart Disease Predicted",
                "Risk Probability (%)": round(r.predicted_risk_score * 100, 1),
                "Model Used": r.model_name,
                "Date": r.prediction_date.strftime("%Y-%m-%d %H:%M:%S")
            })

        return pd.DataFrame(rows)

    finally:
        session.close()
