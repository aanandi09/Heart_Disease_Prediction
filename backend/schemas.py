"""
Pydantic Request & Response Schemas for CardioANN FastAPI Backend
==================================================================
Defines strict type validation, field ranges, and response structures.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class HeartPredictionRequest(BaseModel):
    """
    Patient physiological input features for heart disease risk estimation.
    Notice: Extra fields (such as 'target') are forbidden to prevent data contamination.
    """
    model_config = ConfigDict(extra="forbid")

    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Patient chronological age in years",
        examples=[55]
    )
    sex: int = Field(
        ...,
        ge=0,
        le=1,
        description="Biological sex: 1 = Male, 0 = Female",
        examples=[1]
    )
    cp: int = Field(
        ...,
        ge=0,
        le=3,
        description="Chest pain type (0: Typical Angina, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic)",
        examples=[0]
    )
    trestbps: float = Field(
        ...,
        ge=60.0,
        le=260.0,
        description="Resting systolic blood pressure upon admission (mm Hg)",
        examples=[130.0]
    )
    chol: float = Field(
        ...,
        ge=80.0,
        le=650.0,
        description="Serum cholesterol measurement (mg/dl)",
        examples=[240.0]
    )
    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)",
        examples=[0]
    )
    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting electrocardiographic results (0: Normal, 1: ST-T wave, 2: LV hypertrophy)",
        examples=[0]
    )
    thalach: float = Field(
        ...,
        ge=50.0,
        le=250.0,
        description="Maximum heart rate achieved during exercise stress test (bpm)",
        examples=[150.0]
    )
    exang: int = Field(
        ...,
        ge=0,
        le=1,
        description="Exercise-induced angina (1 = Yes, 0 = No)",
        examples=[0]
    )
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=8.0,
        description="ST depression induced by exercise relative to rest (mm)",
        examples=[1.0]
    )
    slope: int = Field(
        ...,
        ge=0,
        le=2,
        description="Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)",
        examples=[1]
    )
    ca: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of major vessels (0-4) colored by fluoroscopy",
        examples=[0]
    )
    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thallium scintigraphy defect (0: Null, 1: Normal, 2: Fixed defect, 3: Reversible defect)",
        examples=[2]
    )
    threshold: Optional[float] = Field(
        default=0.50,
        ge=0.10,
        le=0.90,
        description="Custom classification threshold for positive risk assignment",
        examples=[0.50]
    )


class HeartPredictionResponse(BaseModel):
    """Structured response model returned by the /predict endpoint."""
    predicted_class: int = Field(description="Binary classification outcome: 0 (Low Risk) or 1 (Elevated Risk)")
    predicted_label: str = Field(description="Descriptive text label")
    probability: float = Field(description="Raw continuous probability output from neural network Sigmoid layer")
    probability_percentage: str = Field(description="Formatted percentage representation")
    threshold_applied: float = Field(description="Operating decision threshold used for classification")
    risk_stratum: str = Field(description="Clinical risk tier: Low Risk Pattern or Elevated Risk Pattern")
    explanation: str = Field(description="Neutral, non-diagnostic interpretation of model estimation")
    disclaimer: str = Field(description="Mandatory educational research prototype disclaimer")
    timestamp: str = Field(description="ISO timestamp of inference execution")


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str
    model_loaded: bool
    pipeline_loaded: bool
    expected_features: List[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Metadata detailing the verified ANN model and preprocessing pipeline."""
    model_name: str
    architecture: str
    evaluation_protocol: str
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_roc_auc: float
    input_features_count: int
    transformed_dimensions: int
    feature_names: List[str]


# -----------------------------------------------------------------------------
# Authentication & User Management Schemas
# -----------------------------------------------------------------------------
class UserSummary(BaseModel):
    """Clean public summary of an authenticated user."""
    id: int
    username: str
    email: str
    full_name: str
    role: str
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    created_at: str


class PatientRegisterRequest(BaseModel):
    """Patient registration payload with field validation."""
    full_name: str = Field(..., min_length=2, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)
    confirm_password: str = Field(..., min_length=6, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=20)
    age: Optional[int] = Field(default=None, ge=1, le=125)
    gender: Optional[str] = Field(default=None, max_length=20)


class LoginRequest(BaseModel):
    """Unified login payload accepting either email or username."""
    email_or_username: str = Field(..., min_length=2)
    password: str = Field(..., min_length=1)


class ForgotPasswordRequest(BaseModel):
    """Password reset request payload."""
    email_or_username: str = Field(..., min_length=2)
    new_password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., min_length=6)


class AuthResponse(BaseModel):
    """Response returned upon authentication or registration."""
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[UserSummary] = None


# -----------------------------------------------------------------------------
# Patient Profile & Medical History Schemas
# -----------------------------------------------------------------------------
class PatientProfileUpdateRequest(BaseModel):
    """Payload to update personal and emergency details."""
    full_name: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    emergency_contact: Optional[str] = None
    address: Optional[str] = None


class PatientProfileResponse(BaseModel):
    """Full patient profile details."""
    user: UserSummary
    blood_group: Optional[str] = None
    emergency_contact: Optional[str] = None
    address: Optional[str] = None
    updated_at: Optional[str] = None


class MedicalHistoryUpdateRequest(BaseModel):
    """Patient-reported clinical background and lifestyle information."""
    existing_conditions: Optional[str] = None
    heart_problems: Optional[str] = None
    family_history: Optional[str] = None
    current_medications: Optional[str] = None
    lifestyle_notes: Optional[str] = None
    additional_notes: Optional[str] = None


class MedicalHistoryResponse(BaseModel):
    """Stored patient-reported medical background."""
    patient_id: int
    existing_conditions: Optional[str] = None
    heart_problems: Optional[str] = None
    family_history: Optional[str] = None
    current_medications: Optional[str] = None
    lifestyle_notes: Optional[str] = None
    additional_notes: Optional[str] = None
    updated_at: str


# -----------------------------------------------------------------------------
# Prediction History & Records Schemas
# -----------------------------------------------------------------------------
class SavePredictionRequest(BaseModel):
    """Payload to persist an ANN inference result into patient's personal history."""
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    probability: float
    threshold: float
    risk_stratum: str
    prediction_result: str
    model_score_experimental: float
    model_version: Optional[str] = "v1.0-MLP-ZeroLeakage"
    notes: Optional[str] = None


class PredictionRecordResponse(BaseModel):
    """Historical prediction record model."""
    id: int
    patient_id: Optional[int]
    timestamp: str
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    prediction_result: str
    probability: float
    threshold: float
    risk_stratum: str
    model_score_experimental: float
    model_version: str
    notes: Optional[str] = None


# -----------------------------------------------------------------------------
# Doctor Portal & Access Control Schemas
# -----------------------------------------------------------------------------
class DoctorGrantUpdateRequest(BaseModel):
    """Patient action to approve or revoke doctor access."""
    doctor_id: int
    status: str = Field(..., pattern="^(approved|revoked)$")


class DoctorPatientSummary(BaseModel):
    """Summary of a patient shown in Doctor's patient list."""
    patient_id: int
    full_name: str
    email: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    access_status: str
    last_prediction_date: Optional[str] = None
    predictions_count: int = 0


class DoctorPatientDetailResponse(BaseModel):
    """Full patient record returned to doctor only if authorization is granted."""
    authorized: bool
    message: str
    patient_profile: Optional[PatientProfileResponse] = None
    medical_history: Optional[MedicalHistoryResponse] = None
    predictions: List[PredictionRecordResponse] = []


class DoctorDashboardSummaryResponse(BaseModel):
    """Doctor dashboard summary metrics."""
    doctor_id: int
    doctor_name: str
    total_patients: int
    authorized_patients: int
    recent_predictions_reviewed: int


class AuditLogResponse(BaseModel):
    """Audit log item representing clinical review activity."""
    id: int
    user_id: int
    user_role: str
    action: str
    target_patient_id: Optional[int] = None
    timestamp: str
    details: Optional[str] = None

