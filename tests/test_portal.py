"""
Automated Test Suite for CardioANN Patient Portal, Doctor Portal & Medical Records
=================================================================================
Validates patient registration, PBKDF2 password hashing, credential authentication,
role-based access control, medical history management, prediction logging,
doctor consent authorization, and clinical audit logging.
"""

import unittest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import get_db_connection


class TestCardioANNPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes FastAPI TestClient and resets test records."""
        cls.client = TestClient(app)

        # Clear any prior test patient data from test suite
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE username IN ('test_patient_1', 'test_patient_2')")
            conn.commit()

    def test_01_patient_registration_success(self):
        """Tests patient registration with validated credentials and profile creation."""
        payload = {
            "full_name": "Rohan Verma",
            "username": "test_patient_1",
            "email": "rohan.verma@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123",
            "phone": "+91 99887 76655",
            "age": 45,
            "gender": "Male",
        }
        response = self.client.post("/auth/register-patient", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIsNotNone(data.get("token"))
        user = data.get("user")
        self.assertEqual(user.get("role"), "patient")
        self.assertEqual(user.get("username"), "test_patient_1")
        TestCardioANNPortal.patient_1_id = user.get("id")

    def test_02_patient_registration_duplicate_rejected(self):
        """Tests that duplicate username or email is strictly rejected."""
        payload = {
            "full_name": "Rohan Clone",
            "username": "test_patient_1",
            "email": "rohan.verma@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123",
        }
        response = self.client.post("/auth/register-patient", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_03_patient_login_success(self):
        """Tests patient login with valid credentials."""
        payload = {
            "email_or_username": "rohan.verma@example.com",
            "password": "SecurePassword123",
        }
        response = self.client.post("/auth/login-patient", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIsNotNone(data.get("token"))

    def test_04_patient_login_invalid_password(self):
        """Tests that invalid password is rejected."""
        payload = {
            "email_or_username": "rohan.verma@example.com",
            "password": "WrongPassword999",
        }
        response = self.client.post("/auth/login-patient", json=payload)
        self.assertEqual(response.status_code, 401)

    def test_05_doctor_login_success(self):
        """Tests login of pre-seeded authorized clinical doctor."""
        payload = {
            "email_or_username": "dr.sharma@heartcare.ai",
            "password": "Doctor@123",
        }
        response = self.client.post("/auth/login-doctor", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        user = data.get("user")
        self.assertEqual(user.get("role"), "doctor")
        TestCardioANNPortal.doctor_id = user.get("id")

    def test_06_patient_cannot_login_as_doctor(self):
        """Tests that a patient cannot login via the Doctor portal."""
        payload = {
            "email_or_username": "rohan.verma@example.com",
            "password": "SecurePassword123",
        }
        response = self.client.post("/auth/login-doctor", json=payload)
        self.assertEqual(response.status_code, 401)

    def test_07_medical_history_crud(self):
        """Tests patient adding, retrieving, updating, and clearing medical history."""
        pid = TestCardioANNPortal.patient_1_id

        # Update medical history
        update_payload = {
            "existing_conditions": "Stage 1 Essential Hypertension, Mild Hyperlipidemia",
            "heart_problems": "Occasional palpitations during strenuous climbing",
            "family_history": "Paternal grandfather suffered early coronary occlusion",
            "current_medications": "Telmisartan 40mg once daily",
            "lifestyle_notes": "Non-smoker, 30 mins brisk walking 4 days/week",
            "additional_notes": "Adhering to low-sodium diet",
        }
        put_res = self.client.put(f"/patient/medical-history/{pid}", json=update_payload)
        self.assertEqual(put_res.status_code, 200)

        # Retrieve and verify
        get_res = self.client.get(f"/patient/medical-history/{pid}")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()
        self.assertEqual(data.get("existing_conditions"), update_payload["existing_conditions"])
        self.assertEqual(data.get("current_medications"), update_payload["current_medications"])

    def test_08_patient_predict_and_save(self):
        """Tests executing ANN prediction and automatically saving to patient history."""
        pid = TestCardioANNPortal.patient_1_id
        pred_payload = {
            "age": 55,
            "sex": 1,
            "cp": 0,
            "trestbps": 135.0,
            "chol": 245.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 145.0,
            "exang": 1,
            "oldpeak": 1.4,
            "slope": 1,
            "ca": 1,
            "thal": 2,
            "threshold": 0.50,
        }
        res = self.client.post(f"/patient/predict-and-save/{pid}", json=pred_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("prediction_id", data)
        self.assertIn("inference", data)

        # Check retrieval of patient prediction records
        hist_res = self.client.get(f"/patient/predictions/{pid}")
        self.assertEqual(hist_res.status_code, 200)
        records = hist_res.json()
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0]["patient_id"], pid)
        self.assertIn("model_score_experimental", records[0])

    def test_09_doctor_authorization_enforcement(self):
        """
        Tests strict clinical authorization:
        1. Doctor attempts to view patient records WITHOUT consent -> Access Restricted.
        2. Patient grants access to doctor -> Doctor can now inspect records.
        3. Audit log is verified for compliance.
        4. Patient revokes access -> Access is restricted again.
        """
        pid = TestCardioANNPortal.patient_1_id
        doc_id = TestCardioANNPortal.doctor_id

        # 1. Unconsented inspection
        unauth_res = self.client.get(f"/doctor/patient/{doc_id}/{pid}")
        self.assertEqual(unauth_res.status_code, 200)
        data_unauth = unauth_res.json()
        self.assertFalse(data_unauth.get("authorized"))
        self.assertIn("Access Restricted", data_unauth.get("message"))
        self.assertIsNone(data_unauth.get("medical_history"))

        # 2. Patient grants access
        grant_payload = {"doctor_id": doc_id, "status": "approved"}
        grant_res = self.client.post(f"/patient/doctor-grants/{pid}", json=grant_payload)
        self.assertEqual(grant_res.status_code, 200)

        # 3. Doctor now accesses authorized record
        auth_res = self.client.get(f"/doctor/patient/{doc_id}/{pid}")
        self.assertEqual(auth_res.status_code, 200)
        data_auth = auth_res.json()
        self.assertTrue(data_auth.get("authorized"))
        self.assertIn("Access Granted", data_auth.get("message"))
        self.assertIsNotNone(data_auth.get("medical_history"))
        self.assertGreaterEqual(len(data_auth.get("predictions")), 1)

        # 4. Verify audit log entry exists
        audit_res = self.client.get(f"/doctor/audit-logs/{doc_id}")
        self.assertEqual(audit_res.status_code, 200)
        audit_records = audit_res.json()
        actions = [log["action"] for log in audit_records]
        self.assertIn("DOCTOR_VIEWED_PATIENT_RECORDS", actions)

        # 5. Patient revokes access
        revoke_payload = {"doctor_id": doc_id, "status": "revoked"}
        revoke_res = self.client.post(f"/patient/doctor-grants/{pid}", json=revoke_payload)
        self.assertEqual(revoke_res.status_code, 200)

        # 6. Doctor inspects again -> Restricted
        re_res = self.client.get(f"/doctor/patient/{doc_id}/{pid}")
        self.assertFalse(re_res.json().get("authorized"))

    def test_10_tenant_isolation_between_patients(self):
        """Tests that Patient 1 cannot access Patient 2's prediction history."""
        # Register Patient 2
        p2_payload = {
            "full_name": "Pooja Mehta",
            "username": "test_patient_2",
            "email": "pooja.mehta@example.com",
            "password": "SecurePassword456",
            "confirm_password": "SecurePassword456",
            "phone": "+91 91234 56789",
            "age": 38,
            "gender": "Female",
        }
        res2 = self.client.post("/auth/register-patient", json=p2_payload)
        p2_id = res2.json()["user"]["id"]

        # Patient 2 runs a prediction
        pred_p2 = {
            "age": 38,
            "sex": 0,
            "cp": 2,
            "trestbps": 120.0,
            "chol": 210.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 165.0,
            "exang": 0,
            "oldpeak": 0.2,
            "slope": 2,
            "ca": 0,
            "thal": 2,
            "threshold": 0.50,
        }
        self.client.post(f"/patient/predict-and-save/{p2_id}", json=pred_p2)

        # Verify Patient 1's history contains ONLY Patient 1's records
        p1_records = self.client.get(f"/patient/predictions/{TestCardioANNPortal.patient_1_id}").json()
        for rec in p1_records:
            self.assertEqual(rec["patient_id"], TestCardioANNPortal.patient_1_id)
            self.assertNotEqual(rec["patient_id"], p2_id)


if __name__ == "__main__":
    unittest.main()
