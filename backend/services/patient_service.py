"""
Patient Medical History & Profile Management Service
=====================================================
Encapsulates profile modifications, patient-reported health history updates,
ANN risk prediction logging, and clinical doctor access grant controls.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from backend.database import get_db_connection
from backend.schemas import (
    UserSummary,
    PatientProfileResponse,
    PatientProfileUpdateRequest,
    MedicalHistoryResponse,
    MedicalHistoryUpdateRequest,
    SavePredictionRequest,
    PredictionRecordResponse,
)


class PatientService:
    """Operations restricted to the authenticated patient's personal record."""

    def get_profile(self, patient_id: int) -> Optional[PatientProfileResponse]:
        """Retrieves patient demographic and profile information."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT u.id, u.username, u.email, u.full_name, u.role, u.phone, u.age, u.gender, u.created_at,
                       p.blood_group, p.emergency_contact, p.address, p.updated_at
                FROM users u
                LEFT JOIN patient_profiles p ON u.id = p.patient_id
                WHERE u.id = ? AND u.role = 'patient'
                """,
                (patient_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            user_sum = UserSummary(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                full_name=row["full_name"],
                role=row["role"],
                phone=row["phone"],
                age=row["age"],
                gender=row["gender"],
                created_at=row["created_at"],
            )

            return PatientProfileResponse(
                user=user_sum,
                blood_group=row["blood_group"],
                emergency_contact=row["emergency_contact"],
                address=row["address"],
                updated_at=row["updated_at"],
            )

    def update_profile(self, patient_id: int, req: PatientProfileUpdateRequest) -> bool:
        """Updates patient demographic fields and profile metadata."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Update core user demographics
            cursor.execute(
                """
                UPDATE users
                SET full_name = COALESCE(?, full_name),
                    phone = COALESCE(?, phone),
                    age = COALESCE(?, age),
                    gender = COALESCE(?, gender)
                WHERE id = ? AND role = 'patient'
                """,
                (req.full_name, req.phone, req.age, req.gender, patient_id),
            )

            # Ensure profile record exists
            cursor.execute("SELECT id FROM patient_profiles WHERE patient_id = ?", (patient_id,))
            if cursor.fetchone():
                cursor.execute(
                    """
                    UPDATE patient_profiles
                    SET blood_group = COALESCE(?, blood_group),
                        emergency_contact = COALESCE(?, emergency_contact),
                        address = COALESCE(?, address),
                        updated_at = ?
                    WHERE patient_id = ?
                    """,
                    (req.blood_group, req.emergency_contact, req.address, now_str, patient_id),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO patient_profiles (patient_id, blood_group, emergency_contact, address, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (patient_id, req.blood_group, req.emergency_contact, req.address, now_str),
                )

            # Audit log
            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, 'patient', 'PROFILE_UPDATED', ?, ?, 'Personal demographic details updated')
                """,
                (patient_id, patient_id, now_str),
            )

            conn.commit()
            return True

    def get_medical_history(self, patient_id: int) -> MedicalHistoryResponse:
        """Retrieves patient-reported health background and medications."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT patient_id, existing_conditions, heart_problems, family_history,
                       current_medications, lifestyle_notes, additional_notes, updated_at
                FROM medical_history
                WHERE patient_id = ?
                """,
                (patient_id,),
            )
            row = cursor.fetchone()

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not row:
                # Initialize record if missing
                cursor.execute(
                    """
                    INSERT INTO medical_history (patient_id, existing_conditions, heart_problems, family_history, current_medications, lifestyle_notes, additional_notes, updated_at)
                    VALUES (?, NULL, NULL, NULL, NULL, NULL, NULL, ?)
                    """,
                    (patient_id, now_str),
                )
                conn.commit()
                return MedicalHistoryResponse(patient_id=patient_id, updated_at=now_str)

            return MedicalHistoryResponse(
                patient_id=row["patient_id"],
                existing_conditions=row["existing_conditions"],
                heart_problems=row["heart_problems"],
                family_history=row["family_history"],
                current_medications=row["current_medications"],
                lifestyle_notes=row["lifestyle_notes"],
                additional_notes=row["additional_notes"],
                updated_at=row["updated_at"] or now_str,
            )

    def update_medical_history(self, patient_id: int, req: MedicalHistoryUpdateRequest) -> bool:
        """Saves patient-reported conditions, heart events, and medications."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO medical_history (patient_id, existing_conditions, heart_problems, family_history, current_medications, lifestyle_notes, additional_notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    existing_conditions = excluded.existing_conditions,
                    heart_problems = excluded.heart_problems,
                    family_history = excluded.family_history,
                    current_medications = excluded.current_medications,
                    lifestyle_notes = excluded.lifestyle_notes,
                    additional_notes = excluded.additional_notes,
                    updated_at = excluded.updated_at
                """,
                (
                    patient_id,
                    req.existing_conditions,
                    req.heart_problems,
                    req.family_history,
                    req.current_medications,
                    req.lifestyle_notes,
                    req.additional_notes,
                    now_str,
                ),
            )

            # Audit log
            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, 'patient', 'MEDICAL_HISTORY_UPDATED', ?, ?, 'Patient medical background updated')
                """,
                (patient_id, patient_id, now_str),
            )

            conn.commit()
            return True

    def delete_medical_history(self, patient_id: int) -> bool:
        """Clears manually entered medical background fields."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE medical_history
                SET existing_conditions = NULL,
                    heart_problems = NULL,
                    family_history = NULL,
                    current_medications = NULL,
                    lifestyle_notes = NULL,
                    additional_notes = NULL,
                    updated_at = ?
                WHERE patient_id = ?
                """,
                (now_str, patient_id),
            )

            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, 'patient', 'MEDICAL_HISTORY_CLEARED', ?, ?, 'Patient cleared medical history record')
                """,
                (patient_id, patient_id, now_str),
            )
            conn.commit()
            return True

    def save_prediction(self, patient_id: Optional[int], req: SavePredictionRequest) -> int:
        """
        Persists an ANN model inference record.
        Maintains an unalterable history of clinical features, continuous probability,
        experimental model score, and classification threshold.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (
                    patient_id, timestamp, age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal, prediction_result,
                    probability, threshold, risk_stratum, model_score_experimental,
                    model_version, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    now_str,
                    req.age,
                    req.sex,
                    req.cp,
                    req.trestbps,
                    req.chol,
                    req.fbs,
                    req.restecg,
                    req.thalach,
                    req.exang,
                    req.oldpeak,
                    req.slope,
                    req.ca,
                    req.thal,
                    req.prediction_result,
                    req.probability,
                    req.threshold,
                    req.risk_stratum,
                    req.model_score_experimental,
                    req.model_version or "v1.0-MLP-ZeroLeakage",
                    req.notes,
                ),
            )
            pred_id = cursor.lastrowid

            if patient_id:
                cursor.execute(
                    """
                    INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                    VALUES (?, 'patient', 'PREDICTION_SAVED', ?, ?, ?)
                    """,
                    (patient_id, patient_id, now_str, f"ANN Risk Prediction #{pred_id} saved (Score: {req.model_score_experimental:.4f})"),
                )

            conn.commit()
            return pred_id

    def get_patient_predictions(self, patient_id: int) -> List[PredictionRecordResponse]:
        """Retrieves chronological prediction history for a specific patient."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, patient_id, timestamp, age, sex, cp, trestbps, chol, fbs,
                       restecg, thalach, exang, oldpeak, slope, ca, thal, prediction_result,
                       probability, threshold, risk_stratum, model_score_experimental,
                       model_version, notes
                FROM predictions
                WHERE patient_id = ?
                ORDER BY id DESC
                """,
                (patient_id,),
            )
            rows = cursor.fetchall()
            return [PredictionRecordResponse(**dict(r)) for r in rows]

    def get_doctor_grants(self, patient_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves list of all registered doctors along with the patient's
        current authorization grant status for each doctor.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT u.id as doctor_id, u.full_name, u.email, u.phone,
                       COALESCE(g.status, 'revoked') as access_status,
                       g.granted_at, g.revoked_at
                FROM users u
                LEFT JOIN doctor_access_grants g ON u.id = g.doctor_id AND g.patient_id = ?
                WHERE u.role = 'doctor'
                ORDER BY u.id ASC
                """,
                (patient_id,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def update_doctor_grant(self, patient_id: int, doctor_id: int, status: str) -> bool:
        """Allows patient to grant ('approved') or revoke ('revoked') medical record access to a doctor."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO doctor_access_grants (patient_id, doctor_id, status, granted_at, revoked_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(patient_id, doctor_id) DO UPDATE SET
                    status = excluded.status,
                    granted_at = CASE WHEN excluded.status = 'approved' THEN excluded.granted_at ELSE granted_at END,
                    revoked_at = CASE WHEN excluded.status = 'revoked' THEN excluded.revoked_at ELSE revoked_at END
                """,
                (
                    patient_id,
                    doctor_id,
                    status,
                    now_str if status == "approved" else None,
                    now_str if status == "revoked" else None,
                ),
            )

            # Audit trail
            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, 'patient', 'DOCTOR_ACCESS_GRANT_CHANGED', ?, ?, ?)
                """,
                (patient_id, patient_id, now_str, f"Doctor access grant set to '{status}' for Doctor #{doctor_id}"),
            )

            conn.commit()
            return True


# Singleton Instance
patient_service = PatientService()
