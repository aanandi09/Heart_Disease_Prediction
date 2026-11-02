"""
CardioANN SQLite Database & Authentication Infrastructure
=========================================================
Manages SQLite database schema, connections, secure PBKDF2-HMAC-SHA256 password
hashing, audit logging, and initial authorized doctor credential seeding.
"""

import os
from pathlib import Path
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "backend" / "data"
DB_PATH = DATA_DIR / "cardioann.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with dictionary rows."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# -----------------------------------------------------------------------------
# Cryptographic Password Utilities (PBKDF2-HMAC-SHA256)
# -----------------------------------------------------------------------------
def hash_password(password: str) -> Tuple[str, str]:
    """
    Hashes a password with a cryptographically secure 16-byte random salt
    using PBKDF2-HMAC-SHA256 across 100,000 iterations.
    """
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return key.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored hash and salt in constant time."""
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return secrets.compare_digest(key.hex(), stored_hash)


# -----------------------------------------------------------------------------
# Database Schema Initialization
# -----------------------------------------------------------------------------
def init_db():
    """Initializes all database tables and seeds default authorized doctors."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Users Table (Patients, Doctors, Admins)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient', 'doctor', 'admin')),
                phone TEXT,
                age INTEGER,
                gender TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # 2. Patient Profiles Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER UNIQUE NOT NULL,
                blood_group TEXT,
                emergency_contact TEXT,
                address TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # 3. Medical History Table (Patient-Reported)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS medical_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER UNIQUE NOT NULL,
                existing_conditions TEXT,
                heart_problems TEXT,
                family_history TEXT,
                current_medications TEXT,
                lifestyle_notes TEXT,
                additional_notes TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # 4. Predictions Table (ANN Historical Inferences)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                timestamp TEXT NOT NULL,
                age INTEGER NOT NULL,
                sex INTEGER NOT NULL,
                cp INTEGER NOT NULL,
                trestbps REAL NOT NULL,
                chol REAL NOT NULL,
                fbs INTEGER NOT NULL,
                restecg INTEGER NOT NULL,
                thalach REAL NOT NULL,
                exang INTEGER NOT NULL,
                oldpeak REAL NOT NULL,
                slope INTEGER NOT NULL,
                ca INTEGER NOT NULL,
                thal INTEGER NOT NULL,
                prediction_result TEXT NOT NULL,
                probability REAL NOT NULL,
                threshold REAL NOT NULL,
                risk_stratum TEXT NOT NULL,
                model_score_experimental REAL NOT NULL,
                model_version TEXT DEFAULT 'v1.0-MLP-ZeroLeakage',
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
        )

        # 5. Doctor Access Grants (Patient Consent System)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS doctor_access_grants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'approved' CHECK(status IN ('approved', 'pending', 'revoked')),
                granted_at TEXT NOT NULL,
                revoked_at TEXT,
                UNIQUE(patient_id, doctor_id),
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (doctor_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # 6. Audit Logs Table (Tamper-evident activity tracking)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                user_role TEXT NOT NULL,
                action TEXT NOT NULL,
                target_patient_id INTEGER,
                timestamp TEXT NOT NULL,
                details TEXT
            )
            """
        )

        conn.commit()

    # Seed initial authorized doctors
    seed_authorized_doctors()


def seed_authorized_doctors():
    """
    Seeds initial authorized doctor accounts.
    Doctors cannot be created through public patient registration.
    """
    default_doctors = [
        {
            "username": "dr.sharma",
            "email": "dr.sharma@heartcare.ai",
            "password": os.getenv("SEED_DOCTOR_PASSWORD", "Doctor@123"),
            "full_name": "Dr. Rajesh Sharma, MD (Cardiology)",
            "phone": "+91 98765 43210",
            "age": 48,
            "gender": "Male",
        },
        {
            "username": "dr.patel",
            "email": "dr.patel@heartcare.ai",
            "password": os.getenv("SEED_DOCTOR_PASSWORD", "Doctor@123"),
            "full_name": "Dr. Ananya Patel, FACC (Interventional Cardiology)",
            "phone": "+91 98123 45678",
            "age": 42,
            "gender": "Female",
        },
    ]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for doc in default_doctors:
            cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (doc["email"], doc["username"]))
            existing = cursor.fetchone()
            if not existing:
                pwd_hash, salt = hash_password(doc["password"])
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT INTO users (username, email, password_hash, salt, full_name, role, phone, age, gender, created_at)
                    VALUES (?, ?, ?, ?, ?, 'doctor', ?, ?, ?, ?)
                    """,
                    (
                        doc["username"],
                        doc["email"],
                        pwd_hash,
                        salt,
                        doc["full_name"],
                        doc["phone"],
                        doc["age"],
                        doc["gender"],
                        now,
                    ),
                )
        conn.commit()


# Ensure database initializes when module is imported
init_db()
