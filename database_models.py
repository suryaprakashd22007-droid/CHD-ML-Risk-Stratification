"""
Database Schema for Coronary Heart Disease Early Detection System.

Defines three SQLite tables via SQLAlchemy ORM:
1. Patient - Basic demographic identifier (patient_id, age, sex)
2. ClinicalBiometrics - Clinical observations (chest pain, blood pressure, etc.)
3. RiskAssessment - Predictions produced by machine learning models
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    """
    Stores non-identifying patient demographic data.
    Generated patient_id (e.g., PAT-1001) avoids storing sensitive personal data.
    """
    __tablename__ = "patients"

    patient_id = Column(String(50), primary_key=True)
    age = Column(Integer, nullable=False)
    sex = Column(Integer, nullable=False)  # 1 = Male, 0 = Female
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships to biometrics and risk assessments
    biometrics = relationship("ClinicalBiometrics", back_populates="patient", cascade="all, delete-orphan")
    assessments = relationship("RiskAssessment", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient(id='{self.patient_id}', age={self.age}, sex={self.sex})>"


class ClinicalBiometrics(Base):
    """
    Stores the 13 clinical biomarkers and risk indicators for each patient record.
    """
    __tablename__ = "clinical_biometrics"

    biometric_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False)

    cp = Column(Integer, nullable=False)
    trestbps = Column(Float, nullable=False)
    chol = Column(Float, nullable=False)
    fbs = Column(Integer, nullable=False)
    restecg = Column(Integer, nullable=False)
    thalach = Column(Float, nullable=False)
    exang = Column(Integer, nullable=False)
    oldpeak = Column(Float, nullable=False)
    slope = Column(Integer, nullable=False)
    ca = Column(Float, nullable=False)
    thal = Column(Float, nullable=False)

    patient = relationship("Patient", back_populates="biometrics")

    def __repr__(self):
        return f"<ClinicalBiometrics(id={self.biometric_id}, patient='{self.patient_id}')>"


class RiskAssessment(Base):
    """
    Stores the machine learning prediction score, class outcome, model name, and timestamp.
    """
    __tablename__ = "risk_assessments"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False)
    predicted_risk_score = Column(Float, nullable=False)  # Probability between 0.0 and 1.0
    predicted_class = Column(Integer, nullable=False)      # 0 = No Heart Disease, 1 = Heart Disease Present
    model_name = Column(String(100), nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="assessments")

    def __repr__(self):
        return f"<RiskAssessment(id={self.prediction_id}, patient='{self.patient_id}', risk={self.predicted_risk_score:.2f})>"
