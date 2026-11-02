"""
CardioANN FastAPI Application
==============================
Production-ready REST API for Heart Disease Risk Prediction using ANN,
featuring Patient and Doctor Portals, secure credential authentication,
patient medical history, and clinical audit logging.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.database import init_db
from backend.schemas import (
    HeartPredictionRequest,
    HeartPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientRegisterRequest,
    LoginRequest,
    ForgotPasswordRequest,
    AuthResponse,
    PatientProfileResponse,
    PatientProfileUpdateRequest,
    MedicalHistoryResponse,
    MedicalHistoryUpdateRequest,
    SavePredictionRequest,
    PredictionRecordResponse,
    DoctorPatientSummary,
    DoctorPatientDetailResponse,
    DoctorDashboardSummaryResponse,
    DoctorGrantUpdateRequest,
    AuditLogResponse,
)
from backend.services.prediction_service import prediction_service
from backend.services.auth_service import auth_service
from backend.services.patient_service import patient_service
from backend.services.doctor_service import doctor_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    1. Initializes SQLite database and seeds default authorized doctors.
    2. Loads and warms up the neural network model and Scikit-Learn pipeline.
    """
    print("[FastAPI] Initializing CardioANN SQLite Database...")
    init_db()

    print("[FastAPI] Initializing Machine Learning artifacts...")
    success, err = prediction_service.load_artifacts()
    if success:
        print("[FastAPI] Artifacts successfully loaded into memory and ready for inference.")
    else:
        print(f"[FastAPI WARNING] Artifact initialization failed: {err}")
    yield
    print("[FastAPI] Shutting down CardioANN backend server.")


app = FastAPI(
    title="CardioANN Clinical Intelligence & Portal REST API",
    description=(
        "High-performance REST API backend for cardiovascular disease risk marker estimation "
        "using an Artificial Neural Network with zero-leakage evaluation, Patient Portal, and Doctor Portal."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware for local Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8000", "http://127.0.0.1:8000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# System & Diagnostic Endpoints (Preserved)
# -----------------------------------------------------------------------------
@app.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    response_description="Basic API identification and documentation links",
)
def read_root():
    """Identifies the CardioANN backend service and provides navigation links."""
    return {
        "service": "CardioANN Clinical Intelligence REST API",
        "version": "2.0.0",
        "status": "online",
        "portals": ["Patient Portal", "Doctor Clinical Portal"],
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "health_check": "/health",
        "model_info": "/model-info",
        "prediction_endpoint": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="API Health & Artifact Readiness Check",
    response_description="System status confirming whether model and preprocessor are active",
)
def check_health():
    """Returns the operational status of the service and model artifact readiness."""
    health = prediction_service.get_health_status()
    if health.status != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health.model_dump(),
        )
    return health


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model Inspection"],
    summary="Neural Network & Pipeline Metadata",
    response_description="Empirical test metrics, architecture layers, and expected feature schema",
)
def get_model_information():
    """Returns verified empirical test metrics and preprocessing schema details."""
    return prediction_service.get_model_info()


@app.post(
    "/predict",
    response_model=HeartPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Execute Heart Disease Risk Prediction (Public / Guest)",
    response_description="Continuous probability score, predicted risk category, and neutral explanation",
)
def predict_heart_disease(request: HeartPredictionRequest):
    """Executes real-time inference on the 13 clinical physiological parameters."""
    try:
        response = prediction_service.predict_heart_disease(request)
        return response
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


# -----------------------------------------------------------------------------
# Authentication Endpoints
# -----------------------------------------------------------------------------
@app.post(
    "/auth/register-patient",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Register a New Patient Account",
)
def register_patient(request: PatientRegisterRequest):
    """Registers a new patient with encrypted password and sets up their initial medical record."""
    res = auth_service.register_patient(request)
    if not res.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.message)
    return res


@app.post(
    "/auth/login-patient",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Patient Login",
)
def login_patient(request: LoginRequest):
    """Authenticates a patient using email or username and password."""
    res = auth_service.login_patient(request)
    if not res.success:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=res.message)
    return res


@app.post(
    "/auth/login-doctor",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Doctor Clinical Login",
)
def login_doctor(request: LoginRequest):
    """
    Authenticates a credentialed clinical doctor.
    Doctor accounts cannot be created through public registration.
    """
    res = auth_service.login_doctor(request)
    if not res.success:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=res.message)
    return res


@app.post(
    "/auth/forgot-password",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Password Reset Option",
)
def forgot_password(request: ForgotPasswordRequest):
    """Allows user to securely reset their password."""
    res = auth_service.forgot_password(request)
    if not res.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.message)
    return res


# -----------------------------------------------------------------------------
# Patient Portal Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/patient/profile/{patient_id}",
    response_model=PatientProfileResponse,
    tags=["Patient Portal"],
    summary="Get Patient Profile",
)
def get_patient_profile(patient_id: int):
    """Retrieves patient personal profile and demographics."""
    profile = patient_service.get_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient profile not found.")
    return profile


@app.put(
    "/patient/profile/{patient_id}",
    tags=["Patient Portal"],
    summary="Update Patient Profile",
)
def update_patient_profile(patient_id: int, request: PatientProfileUpdateRequest):
    """Updates patient personal and emergency contact details."""
    success = patient_service.update_profile(patient_id, request)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to update profile.")
    return {"success": True, "message": "Profile updated successfully."}


@app.get(
    "/patient/medical-history/{patient_id}",
    response_model=MedicalHistoryResponse,
    tags=["Patient Portal"],
    summary="Get Patient-Reported Medical History",
)
def get_patient_medical_history(patient_id: int):
    """Retrieves patient-reported health background (conditions, heart history, medications)."""
    return patient_service.get_medical_history(patient_id)


@app.put(
    "/patient/medical-history/{patient_id}",
    tags=["Patient Portal"],
    summary="Update Patient Medical History",
)
def update_patient_medical_history(patient_id: int, request: MedicalHistoryUpdateRequest):
    """Updates patient-reported medical conditions, family history, and medications."""
    success = patient_service.update_medical_history(patient_id, request)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to update medical history.")
    return {"success": True, "message": "Medical history updated successfully."}


@app.delete(
    "/patient/medical-history/{patient_id}",
    tags=["Patient Portal"],
    summary="Clear Patient Medical History",
)
def delete_patient_medical_history(patient_id: int):
    """Clears manually entered medical background record."""
    success = patient_service.delete_medical_history(patient_id)
    return {"success": success, "message": "Medical history cleared."}


@app.post(
    "/patient/predict-and-save/{patient_id}",
    tags=["Patient Portal"],
    summary="Run Prediction and Automatically Save to Patient Record",
)
def predict_and_save_patient_record(patient_id: int, request: HeartPredictionRequest):
    """
    Executes neural network inference and links the prediction directly to the
    authenticated patient's longitudinal record.
    """
    pred_res = prediction_service.predict_heart_disease(request)

    save_req = SavePredictionRequest(
        age=request.age,
        sex=request.sex,
        cp=request.cp,
        trestbps=request.trestbps,
        chol=request.chol,
        fbs=request.fbs,
        restecg=request.restecg,
        thalach=request.thalach,
        exang=request.exang,
        oldpeak=request.oldpeak,
        slope=request.slope,
        ca=request.ca,
        thal=request.thal,
        probability=pred_res.probability,
        threshold=pred_res.threshold_applied,
        risk_stratum=pred_res.risk_stratum,
        prediction_result=pred_res.predicted_label,
        model_score_experimental=pred_res.probability,
        model_version="v1.0-MLP-ZeroLeakage",
        notes=pred_res.explanation,
    )

    pred_id = patient_service.save_prediction(patient_id, save_req)

    return {
        "prediction_id": pred_id,
        "inference": pred_res,
        "message": f"Prediction #{pred_id} successfully saved to patient record.",
    }


@app.get(
    "/patient/predictions/{patient_id}",
    response_model=List[PredictionRecordResponse],
    tags=["Patient Portal"],
    summary="Get Patient's Prediction History",
)
def get_patient_prediction_history(patient_id: int):
    """Retrieves all previous ANN model risk predictions executed by this patient."""
    return patient_service.get_patient_predictions(patient_id)


@app.get(
    "/patient/doctor-grants/{patient_id}",
    tags=["Patient Portal"],
    summary="List Registered Doctors & Access Grant Status",
)
def get_doctor_access_grants(patient_id: int):
    """Retrieves all registered doctors and patient's authorization status for each."""
    return patient_service.get_doctor_grants(patient_id)


@app.post(
    "/patient/doctor-grants/{patient_id}",
    tags=["Patient Portal"],
    summary="Grant or Revoke Doctor Access",
)
def update_doctor_access_grant(patient_id: int, request: DoctorGrantUpdateRequest):
    """Allows patient to approve or revoke doctor access to their health records."""
    success = patient_service.update_doctor_grant(patient_id, request.doctor_id, request.status)
    return {"success": success, "message": f"Doctor access grant set to '{request.status}'."}


# -----------------------------------------------------------------------------
# Doctor Portal Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/doctor/summary/{doctor_id}",
    response_model=DoctorDashboardSummaryResponse,
    tags=["Doctor Portal"],
    summary="Doctor Dashboard Overview Metrics",
)
def get_doctor_dashboard_summary(doctor_id: int):
    """Computes total patients, authorized access grants, and review activity."""
    return doctor_service.get_summary(doctor_id)


@app.get(
    "/doctor/patients/{doctor_id}",
    response_model=List[DoctorPatientSummary],
    tags=["Doctor Portal"],
    summary="Search & List Patients with Authorization Status",
)
def list_doctor_patients(doctor_id: int, search: Optional[str] = Query(default="", description="Search by ID or Name")):
    """Searches patient directory and displays whether patient has granted access."""
    return doctor_service.list_patients(doctor_id, search_term=search or "")


@app.get(
    "/doctor/patient/{doctor_id}/{patient_id}",
    response_model=DoctorPatientDetailResponse,
    tags=["Doctor Portal"],
    summary="Inspect Patient Clinical History (Authorization Enforced)",
)
def get_patient_details_for_doctor(doctor_id: int, patient_id: int):
    """
    Retrieves full medical background and prediction history if authorized.
    Automatically generates an entry in the clinical audit log.
    Withholds records if patient has not granted consent.
    """
    return doctor_service.get_patient_details(doctor_id, patient_id)


@app.get(
    "/doctor/audit-logs/{doctor_id}",
    response_model=List[AuditLogResponse],
    tags=["Doctor Portal"],
    summary="View Doctor Clinical Review Audit Log",
)
def get_doctor_audit_logs(doctor_id: int):
    """Retrieves clinical access trail for compliance auditability."""
    return doctor_service.get_audit_logs(doctor_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
