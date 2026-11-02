"""
Doctor Clinical Portal & Audited Patient Record Access Service
==============================================================
Provides patient directory querying, authorization checks, medical history
inspection, prediction review, and tamper-evident access audit logging.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from backend.database import get_db_connection
from backend.schemas import (
    UserSummary,
    PatientProfileResponse,
    MedicalHistoryResponse,
    PredictionRecordResponse,
    DoctorPatientSummary,
    DoctorPatientDetailResponse,
    DoctorDashboardSummaryResponse,
    AuditLogResponse,
)


class DoctorService:
    """Operations restricted to authorized clinical medical staff."""

    def get_summary(self, doctor_id: int) -> DoctorDashboardSummaryResponse:
        """Computes summary metrics for doctor portal home dashboard."""
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Doctor name
            cursor.execute("SELECT full_name FROM users WHERE id = ? AND role = 'doctor'", (doctor_id,))
            doc_row = cursor.fetchone()
            doc_name = doc_row["full_name"] if doc_row else f"Doctor #{doctor_id}"

            # Total registered patients
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'patient'")
            total_patients = cursor.fetchone()["count"]

            # Authorized patients
            cursor.execute(
                """
                SELECT COUNT(DISTINCT patient_id) as count
                FROM doctor_access_grants
                WHERE doctor_id = ? AND status = 'approved'
                """,
                (doctor_id,),
            )
            auth_patients = cursor.fetchone()["count"]

            # Recent predictions reviewed
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM audit_logs
                WHERE user_id = ? AND action = 'DOCTOR_VIEWED_PATIENT_RECORDS'
                """,
                (doctor_id,),
            )
            reviewed_count = cursor.fetchone()["count"]

            return DoctorDashboardSummaryResponse(
                doctor_id=doctor_id,
                doctor_name=doc_name,
                total_patients=total_patients,
                authorized_patients=auth_patients,
                recent_predictions_reviewed=reviewed_count,
            )

    def list_patients(self, doctor_id: int, search_term: str = "") -> List[DoctorPatientSummary]:
        """
        Lists registered patients and explicitly displays authorization grant status.
        Supports filtering by Patient ID (e.g., '1' or 'PAT-1') or patient name.
        """
        clean_search = search_term.strip().lower()
        if clean_search.startswith("pat-"):
            clean_search = clean_search.replace("pat-", "").strip()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT u.id as patient_id, u.full_name, u.email, u.age, u.gender, u.phone,
                       COALESCE(g.status, 'revoked') as access_status,
                       (SELECT MAX(timestamp) FROM predictions WHERE patient_id = u.id) as last_prediction_date,
                       (SELECT COUNT(*) FROM predictions WHERE patient_id = u.id) as predictions_count
                FROM users u
                LEFT JOIN doctor_access_grants g ON u.id = g.patient_id AND g.doctor_id = ?
                WHERE u.role = 'patient'
            """
            params: List[Any] = [doctor_id]

            if clean_search:
                if clean_search.isdigit():
                    query += " AND (u.id = ? OR lower(u.full_name) LIKE ? OR lower(u.email) LIKE ?)"
                    params.extend([int(clean_search), f"%{clean_search}%", f"%{clean_search}%"])
                else:
                    query += " AND (lower(u.full_name) LIKE ? OR lower(u.email) LIKE ?)"
                    params.extend([f"%{clean_search}%", f"%{clean_search}%"])

            query += " ORDER BY u.id ASC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [DoctorPatientSummary(**dict(r)) for r in rows]

    def get_patient_details(self, doctor_id: int, patient_id: int) -> DoctorPatientDetailResponse:
        """
        Retrieves complete patient clinical record ONLY if authorization is granted.
        If access is revoked or not granted, detailed medical data is withheld.
        Every authorized access is permanently recorded in the audit log.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Verify patient exists
            cursor.execute("SELECT id, username, email, full_name, role, phone, age, gender, created_at FROM users WHERE id = ? AND role = 'patient'", (patient_id,))
            u_row = cursor.fetchone()
            if not u_row:
                return DoctorPatientDetailResponse(
                    authorized=False,
                    message=f"Patient record #{patient_id} does not exist.",
                )

            # Check authorization grant
            cursor.execute(
                """
                SELECT status
                FROM doctor_access_grants
                WHERE patient_id = ? AND doctor_id = ?
                """,
                (patient_id, doctor_id),
            )
            grant_row = cursor.fetchone()
            is_authorized = grant_row is not None and grant_row["status"] == "approved"

            if not is_authorized:
                # Log unauthorized inspection attempt
                cursor.execute(
                    """
                    INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                    VALUES (?, 'doctor', 'UNAUTHORIZED_ACCESS_ATTEMPT', ?, ?, 'Doctor attempted to view patient records without active consent')
                    """,
                    (doctor_id, patient_id, now_str),
                )
                conn.commit()

                return DoctorPatientDetailResponse(
                    authorized=False,
                    message=(
                        f"Access Restricted: Patient {u_row['full_name']} (PAT-{patient_id}) has not granted consent "
                        "for your clinical review. The patient must approve access through their Patient Portal."
                    ),
                )

            # Log authorized clinical record review in Audit Log
            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, 'doctor', 'DOCTOR_VIEWED_PATIENT_RECORDS', ?, ?, 'Full clinical record, medical history, and ANN prediction timeline reviewed')
                """,
                (doctor_id, patient_id, now_str),
            )
            conn.commit()

            # Retrieve profile
            cursor.execute(
                "SELECT blood_group, emergency_contact, address, updated_at FROM patient_profiles WHERE patient_id = ?",
                (patient_id,),
            )
            p_row = cursor.fetchone()

            user_sum = UserSummary(
                id=u_row["id"],
                username=u_row["username"],
                email=u_row["email"],
                full_name=u_row["full_name"],
                role=u_row["role"],
                phone=u_row["phone"],
                age=u_row["age"],
                gender=u_row["gender"],
                created_at=u_row["created_at"],
            )

            profile_resp = PatientProfileResponse(
                user=user_sum,
                blood_group=p_row["blood_group"] if p_row else None,
                emergency_contact=p_row["emergency_contact"] if p_row else None,
                address=p_row["address"] if p_row else None,
                updated_at=p_row["updated_at"] if p_row else None,
            )

            # Retrieve patient medical history
            cursor.execute(
                """
                SELECT patient_id, existing_conditions, heart_problems, family_history,
                       current_medications, lifestyle_notes, additional_notes, updated_at
                FROM medical_history
                WHERE patient_id = ?
                """,
                (patient_id,),
            )
            m_row = cursor.fetchone()
            med_resp = None
            if m_row:
                med_resp = MedicalHistoryResponse(
                    patient_id=m_row["patient_id"],
                    existing_conditions=m_row["existing_conditions"],
                    heart_problems=m_row["heart_problems"],
                    family_history=m_row["family_history"],
                    current_medications=m_row["current_medications"],
                    lifestyle_notes=m_row["lifestyle_notes"],
                    additional_notes=m_row["additional_notes"],
                    updated_at=m_row["updated_at"] or now_str,
                )

            # Retrieve prediction records
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
            pred_rows = cursor.fetchall()
            preds = [PredictionRecordResponse(**dict(r)) for r in pred_rows]

            return DoctorPatientDetailResponse(
                authorized=True,
                message=f"Access Granted: Clinical records for {u_row['full_name']} (PAT-{patient_id}) retrieved successfully.",
                patient_profile=profile_resp,
                medical_history=med_resp,
                predictions=preds,
            )

    def get_audit_logs(self, doctor_id: int) -> List[AuditLogResponse]:
        """Retrieves clinical access audit trail for compliance verification."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, user_role, action, target_patient_id, timestamp, details
                FROM audit_logs
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 100
                """,
                (doctor_id,),
            )
            rows = cursor.fetchall()
            return [AuditLogResponse(**dict(r)) for r in rows]


# Singleton Instance
doctor_service = DoctorService()
