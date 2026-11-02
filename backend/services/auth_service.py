"""
Authentication & Access Control Service for CardioANN
======================================================
Handles user registration, credential verification, secure password hashing,
session token generation, role verification, and password resets.
"""

from datetime import datetime
import secrets
from typing import Optional, Dict, Any

from backend.database import get_db_connection, hash_password, verify_password
from backend.schemas import (
    PatientRegisterRequest,
    LoginRequest,
    ForgotPasswordRequest,
    AuthResponse,
    UserSummary,
)

# In-memory session store mapping token -> user info dict
# Provides fast session validation without external Redis dependencies
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}


def generate_session_token(user_id: int, username: str, role: str) -> str:
    """Generates a cryptographically random session token and stores session context."""
    token = f"{role}_{secrets.token_hex(24)}"
    ACTIVE_SESSIONS[token] = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "created_at": datetime.now().isoformat(),
    }
    return token


def get_current_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Validates session token and returns active user context."""
    return ACTIVE_SESSIONS.get(token)


def invalidate_token(token: str):
    """Terminates active session."""
    if token in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[token]


class AuthService:
    """Authentication operations for Patients and Credentialed Doctors."""

    def register_patient(self, req: PatientRegisterRequest) -> AuthResponse:
        """
        Registers a new patient account with encrypted credentials.
        Automatically initializes patient profile and medical history containers.
        """
        # Validate password match
        if req.password != req.confirm_password:
            return AuthResponse(success=False, message="Passwords do not match. Please re-enter identical passwords.")

        if len(req.password) < 6:
            return AuthResponse(success=False, message="Password must be at least 6 characters long.")

        clean_email = req.email.strip().lower()
        clean_username = req.username.strip().lower()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check uniqueness
            cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (clean_email, clean_username))
            existing = cursor.fetchone()
            if existing:
                return AuthResponse(
                    success=False,
                    message="An account with this email address or username already exists. Please login instead.",
                )

            # Hash password securely
            pwd_hash, salt = hash_password(req.password)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, full_name, role, phone, age, gender, created_at)
                VALUES (?, ?, ?, ?, ?, 'patient', ?, ?, ?, ?)
                """,
                (clean_username, clean_email, pwd_hash, salt, req.full_name.strip(), req.phone, req.age, req.gender, now_str),
            )
            patient_id = cursor.lastrowid

            # Initialize profile
            cursor.execute(
                """
                INSERT INTO patient_profiles (patient_id, blood_group, emergency_contact, address, updated_at)
                VALUES (?, NULL, NULL, NULL, ?)
                """,
                (patient_id, now_str),
            )

            # Initialize empty medical history
            cursor.execute(
                """
                INSERT INTO medical_history (patient_id, existing_conditions, heart_problems, family_history, current_medications, lifestyle_notes, additional_notes, updated_at)
                VALUES (?, NULL, NULL, NULL, NULL, NULL, NULL, ?)
                """,
                (patient_id, now_str),
            )

            # Audit record
            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, 'patient', 'PATIENT_REGISTRATION', ?, ?, 'New patient account registered successfully')
                """,
                (patient_id, patient_id, now_str),
            )

            conn.commit()

            user_summary = UserSummary(
                id=patient_id,
                username=clean_username,
                email=clean_email,
                full_name=req.full_name.strip(),
                role="patient",
                phone=req.phone,
                age=req.age,
                gender=req.gender,
                created_at=now_str,
            )

            token = generate_session_token(patient_id, clean_username, "patient")

            return AuthResponse(
                success=True,
                message=f"Welcome to HeartCare, {req.full_name}! Your account has been registered successfully.",
                token=token,
                user=user_summary,
            )

    def login_patient(self, req: LoginRequest) -> AuthResponse:
        """Authenticates a registered patient."""
        return self._login_user(req, expected_role="patient")

    def login_doctor(self, req: LoginRequest) -> AuthResponse:
        """
        Authenticates a credentialed clinical doctor.
        Doctor accounts cannot be created by public registration.
        """
        return self._login_user(req, expected_role="doctor")

    def _login_user(self, req: LoginRequest, expected_role: str) -> AuthResponse:
        clean_identifier = req.email_or_username.strip().lower()

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, email, password_hash, salt, full_name, role, phone, age, gender, created_at
                FROM users
                WHERE lower(email) = ? OR lower(username) = ?
                """,
                (clean_identifier, clean_identifier),
            )
            row = cursor.fetchone()

            if not row:
                role_label = "doctor" if expected_role == "doctor" else "patient"
                return AuthResponse(
                    success=False,
                    message=f"No {role_label} account found matching '{req.email_or_username}'. Please verify credentials.",
                )

            # Role verification check
            if row["role"] != expected_role:
                if expected_role == "doctor":
                    return AuthResponse(
                        success=False,
                        message="Access denied: This portal is exclusively reserved for authorized clinical staff.",
                    )
                else:
                    return AuthResponse(
                        success=False,
                        message="Doctor credentials cannot be used in the Patient Portal. Please use the Doctor Portal.",
                    )

            # Password verification
            if not verify_password(req.password, row["password_hash"], row["salt"]):
                return AuthResponse(success=False, message="Incorrect password. Please verify your credentials.")

            user_summary = UserSummary(
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

            token = generate_session_token(row["id"], row["username"], row["role"])

            # Log access
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, ?, ?, ?, ?, 'Successful authentication session established')
                """,
                (row["id"], row["role"], f"{row['role'].upper()}_LOGIN", row["id"] if row["role"] == "patient" else None, now_str),
            )
            conn.commit()

            return AuthResponse(
                success=True,
                message=f"Welcome back, {row['full_name']}!",
                token=token,
                user=user_summary,
            )

    def forgot_password(self, req: ForgotPasswordRequest) -> AuthResponse:
        """Resets a user's password with verification."""
        if req.new_password != req.confirm_password:
            return AuthResponse(success=False, message="Passwords do not match. Please re-enter identical passwords.")

        if len(req.new_password) < 6:
            return AuthResponse(success=False, message="Password must be at least 6 characters long.")

        clean_identifier = req.email_or_username.strip().lower()

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, full_name, role FROM users WHERE lower(email) = ? OR lower(username) = ?",
                (clean_identifier, clean_identifier),
            )
            row = cursor.fetchone()

            if not row:
                return AuthResponse(success=False, message=f"No account found with identifier '{req.email_or_username}'.")

            pwd_hash, salt = hash_password(req.new_password)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                (pwd_hash, salt, row["id"]),
            )

            cursor.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, target_patient_id, timestamp, details)
                VALUES (?, ?, 'PASSWORD_RESET', ?, ?, 'Password credentials successfully updated')
                """,
                (row["id"], row["role"], row["id"] if row["role"] == "patient" else None, now_str),
            )
            conn.commit()

            return AuthResponse(
                success=True,
                message=f"Password for {row['full_name']} has been reset successfully. You may now login.",
            )


# Singleton Instance
auth_service = AuthService()
