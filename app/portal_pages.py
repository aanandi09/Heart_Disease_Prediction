"""
HeartCare portal pages (Streamlit frontend)
===========================================
- Animated stethoscope background
- Patient login / register / forgot password page
- Doctor login page
- Patient portal: profile, medical history, prediction history, doctor access
- Doctor portal: patient list, patient record (consent enforced), audit log

All data comes from the FastAPI backend through api_client.py.
"""

import html

import pandas as pd
import streamlit as st

from api_client import api_delete, api_get, api_post, api_put

GENDER_OPTIONS = ["Male", "Female", "Other"]
BLOOD_GROUPS = ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
SEX_LABEL = {1: "Male", 0: "Female"}

PORTAL_FOR_ROLE = {"patient": "My Portal", "doctor": "Doctor Portal"}


# -----------------------------------------------------------------------------
# Stethoscope background animation (pure CSS, no extra files)
# -----------------------------------------------------------------------------
def render_stethoscope_background():
    """Fixed, non-interactive stethoscope: tube sways, chest piece thumps, rings pulse."""
    st.markdown(
        """
        <style>
        .stetho-wrap {
            position: fixed;
            right: 2%;
            bottom: 4%;
            width: 130px;
            opacity: 0.40;
            pointer-events: none;
            user-select: none;
            z-index: 0;
        }
        .stetho-wrap svg { overflow: visible; }
        .stetho-sway {
            transform-origin: 100px 20px;
            animation: stethoSway 6s ease-in-out infinite;
        }
        @keyframes stethoSway {
            0%, 100% { transform: rotate(-3deg); }
            50% { transform: rotate(3deg); }
        }
        .stetho-chest {
            transform-box: fill-box;
            transform-origin: center;
            animation: stethoThump 1.6s ease-in-out infinite;
        }
        @keyframes stethoThump {
            0%, 100% { transform: scale(1); }
            15% { transform: scale(1.12); }
            30% { transform: scale(1); }
            45% { transform: scale(1.07); }
        }
        .stetho-ring {
            transform-box: fill-box;
            transform-origin: center;
            animation: stethoBeat 1.6s ease-out infinite;
        }
        .stetho-ring.r2 { animation-delay: 0.5s; }
        @keyframes stethoBeat {
            0% { transform: scale(0.6); opacity: 0.7; }
            100% { transform: scale(2.4); opacity: 0; }
        }
        @media (max-width: 700px) { .stetho-wrap { width: 84px; } }
        @media (prefers-reduced-motion: reduce) {
            .stetho-wrap * { animation: none !important; }
        }
        </style>
        <div class="stetho-wrap" aria-hidden="true">
            <svg viewBox="0 0 200 260" xmlns="http://www.w3.org/2000/svg" fill="none"
                 stroke="#6B8A61" stroke-width="5" stroke-linecap="round">
                <g class="stetho-sway">
                    <circle cx="60" cy="20" r="8" fill="#C7E3B8"/>
                    <circle cx="140" cy="20" r="8" fill="#C7E3B8"/>
                    <path d="M60 28 C60 90 100 100 100 130"/>
                    <path d="M140 28 C140 90 100 100 100 130"/>
                    <path d="M100 130 C100 185 150 185 150 215"/>
                    <g class="stetho-chest">
                        <circle cx="150" cy="232" r="18" fill="#E2F1D9"/>
                        <circle cx="150" cy="232" r="8" fill="#6B8A61" stroke="none"/>
                    </g>
                    <circle class="stetho-ring" cx="150" cy="232" r="18" stroke="#A8C59A" stroke-width="3"/>
                    <circle class="stetho-ring r2" cx="150" cy="232" r="18" stroke="#A8C59A" stroke-width="3"/>
                </g>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def current_user():
    auth = st.session_state.get("auth")
    return auth["user"] if auth else None


def logout():
    st.session_state.pop("auth", None)
    st.session_state.pop("opened_record", None)
    st.session_state.active_nav_section = "Home"
    st.rerun()


def _escape(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _metric(label: str, value, sub: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{_escape(label)}</div>
            <div class="metric-value" style="font-size:1.45rem;">{_escape(value)}</div>
            <div class="metric-sub">{_escape(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _portal_header(badge: str, title: str, caption: str):
    st.markdown(
        f"""
        <div class="hero-container" style="padding:26px 32px;">
            <div class="hero-badge"><span class="pulse-dot"></span> {_escape(badge)}</div>
            <h1 class="hero-title" style="font-size:1.8rem;">{_escape(title)}</h1>
            <p class="hero-subtitle" style="margin-bottom:0;">{_escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _finish_login(data: dict, role: str):
    st.session_state.auth = {"token": data["token"], "user": data["user"]}
    st.session_state.active_nav_section = PORTAL_FOR_ROLE[role]
    st.rerun()


def _do_login(role: str, identifier: str, password: str):
    if not identifier.strip() or not password:
        st.warning("Enter both email/username and password.")
        return
    path = "/auth/login-patient" if role == "patient" else "/auth/login-doctor"
    ok, data = api_post(path, {"email_or_username": identifier, "password": password})
    if not ok:
        st.error(data)
        return
    _finish_login(data, role)


def _redirect_if_logged_in():
    user = current_user()
    if user:
        st.session_state.active_nav_section = PORTAL_FOR_ROLE.get(user["role"], "Home")
        st.rerun()


def _require_role(role: str, login_section: str) -> bool:
    user = current_user()
    if not user or user["role"] != role:
        st.session_state.active_nav_section = login_section
        st.rerun()
        return False
    return True


# -----------------------------------------------------------------------------
# Patient login page
# -----------------------------------------------------------------------------
def render_patient_login():
    _redirect_if_logged_in()
    _portal_header(
        "Patient Portal",
        "Patient Login",
        "Login to save your risk predictions, manage your medical history, and control which doctors can view your records.",
    )

    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        tab_login, tab_register, tab_forgot = st.tabs(["Login", "Register", "Forgot Password"])

        with tab_login:
            with st.form("patient_login_form"):
                identifier = st.text_input("Email or Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login as Patient", type="primary", width="stretch")
            if submitted:
                _do_login("patient", identifier, password)

        with tab_register:
            with st.form("patient_register_form"):
                full_name = st.text_input("Full name")
                c1, c2 = st.columns(2)
                username = c1.text_input("Username")
                email = c2.text_input("Email")
                phone = c1.text_input("Phone (optional)")
                age = c2.number_input("Age", min_value=1, max_value=125, value=30, step=1)
                gender = st.selectbox("Gender", GENDER_OPTIONS)
                p1 = st.text_input("Password (minimum 6 characters)", type="password")
                p2 = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create Patient Account", type="primary", width="stretch")
            if submitted:
                payload = {
                    "full_name": full_name.strip(),
                    "username": username.strip(),
                    "email": email.strip(),
                    "password": p1,
                    "confirm_password": p2,
                    "phone": phone.strip() or None,
                    "age": int(age),
                    "gender": gender,
                }
                ok, data = api_post("/auth/register-patient", payload)
                if ok:
                    _finish_login(data, "patient")
                else:
                    st.error(data)

        with tab_forgot:
            with st.form("patient_forgot_form"):
                identifier = st.text_input("Email or Username", key="forgot_ident")
                n1 = st.text_input("New password", type="password")
                n2 = st.text_input("Confirm new password", type="password")
                submitted = st.form_submit_button("Reset Password", type="primary", width="stretch")
            if submitted:
                ok, data = api_post(
                    "/auth/forgot-password",
                    {"email_or_username": identifier, "new_password": n1, "confirm_password": n2},
                )
                if ok:
                    st.success(data["message"])
                else:
                    st.error(data)


# -----------------------------------------------------------------------------
# Doctor login page
# -----------------------------------------------------------------------------
def render_doctor_login():
    _redirect_if_logged_in()
    _portal_header(
        "Clinical Staff Only",
        "Doctor Login",
        "Doctor accounts are issued by the administrator and cannot be created here. Patient records are visible only after the patient grants access.",
    )

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        with st.form("doctor_login_form"):
            identifier = st.text_input("Doctor email or username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login as Doctor", type="primary", width="stretch")
        if submitted:
            _do_login("doctor", identifier, password)


# -----------------------------------------------------------------------------
# Patient portal
# -----------------------------------------------------------------------------
def save_prediction_for_patient(patient_id: int, payload: dict):
    """Runs the backend prediction and saves it to this patient's history."""
    return api_post(f"/patient/predict-and-save/{patient_id}", payload)


def _patient_profile_tab(pid: int):
    ok, prof = api_get(f"/patient/profile/{pid}")
    if not ok:
        st.error(prof)
        return
    u = prof["user"]

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        full_name = c1.text_input("Full name", u.get("full_name") or "")
        phone = c2.text_input("Phone", u.get("phone") or "")
        age = c1.number_input("Age", min_value=1, max_value=125, value=int(u.get("age") or 30), step=1)
        g_val = u.get("gender") or GENDER_OPTIONS[0]
        gender = c2.selectbox(
            "Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(g_val) if g_val in GENDER_OPTIONS else 0
        )
        b_val = prof.get("blood_group") or ""
        blood = c1.selectbox(
            "Blood group", BLOOD_GROUPS, index=BLOOD_GROUPS.index(b_val) if b_val in BLOOD_GROUPS else 0
        )
        emergency = c2.text_input("Emergency contact", prof.get("emergency_contact") or "")
        address = st.text_area("Address", prof.get("address") or "")
        saved = st.form_submit_button("Save Profile", type="primary")

    if saved:
        payload = {
            "full_name": full_name.strip() or None,
            "phone": phone.strip() or None,
            "age": int(age),
            "gender": gender,
            "blood_group": blood or None,
            "emergency_contact": emergency.strip() or None,
            "address": address.strip() or None,
        }
        ok, res = api_put(f"/patient/profile/{pid}", payload)
        if ok:
            if payload["full_name"]:
                st.session_state.auth["user"]["full_name"] = payload["full_name"]
            st.success(res["message"])
        else:
            st.error(res)


def _patient_history_form_tab(pid: int):
    ok, mh = api_get(f"/patient/medical-history/{pid}")
    if not ok:
        st.error(mh)
        return

    fields = [
        ("existing_conditions", "Existing conditions (diabetes, hypertension, etc.)"),
        ("heart_problems", "Previous heart problems or procedures"),
        ("family_history", "Family history of heart disease"),
        ("current_medications", "Current medications"),
        ("lifestyle_notes", "Lifestyle (smoking, exercise, diet)"),
        ("additional_notes", "Additional notes"),
    ]

    with st.form("medical_history_form"):
        values = {key: st.text_area(label, mh.get(key) or "", height=90) for key, label in fields}
        saved = st.form_submit_button("Save Medical History", type="primary")

    if saved:
        ok, res = api_put(
            f"/patient/medical-history/{pid}",
            {key: (val.strip() or None) for key, val in values.items()},
        )
        st.success(res["message"]) if ok else st.error(res)

    if st.button("Clear Medical History", key="btn_clear_mh"):
        ok, res = api_delete(f"/patient/medical-history/{pid}")
        if ok:
            st.success(res["message"])
            st.rerun()
        else:
            st.error(res)


def _prediction_rows(records: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": r["timestamp"],
                "Age": r["age"],
                "Sex": SEX_LABEL.get(r["sex"], r["sex"]),
                "Blood Pressure": r["trestbps"],
                "Cholesterol": r["chol"],
                "Max Heart Rate": r["thalach"],
                "ST Depression": r["oldpeak"],
                "Vessels": r["ca"],
                "Probability (%)": round(r["probability"] * 100, 1),
                "Threshold": r["threshold"],
                "Result": r["risk_stratum"],
            }
            for r in records
        ]
    )


def _patient_predictions_tab(pid: int):
    ok, records = api_get(f"/patient/predictions/{pid}")
    if not ok:
        st.error(records)
        return
    if not records:
        st.info("No saved predictions yet. Open the Prediction page while logged in; every result is saved here automatically.")
        return

    df = _prediction_rows(records)
    st.dataframe(df, width="stretch", hide_index=True)

    if len(df) > 1:
        st.markdown("**Risk probability over time**")
        st.line_chart(df.iloc[::-1].set_index("Date")["Probability (%)"])

    st.download_button(
        "Download History (.csv)",
        data=df.to_csv(index=False),
        file_name=f"patient_{pid}_prediction_history.csv",
        mime="text/csv",
    )


def _patient_access_tab(pid: int):
    ok, grants = api_get(f"/patient/doctor-grants/{pid}")
    if not ok:
        st.error(grants)
        return
    st.caption("A doctor can view your records only while the status below is 'approved'.")

    for g in grants:
        c1, c2, c3 = st.columns([3, 1.2, 1.4], vertical_alignment="center")
        c1.markdown(f"**{_escape(g['full_name'])}**<br><span style='font-size:0.85rem;'>{_escape(g['email'])}</span>", unsafe_allow_html=True)
        approved = g["access_status"] == "approved"
        c2.markdown("Approved" if approved else "No access")
        new_status = "revoked" if approved else "approved"
        label = "Revoke Access" if approved else "Grant Access"
        if c3.button(label, key=f"grant_{g['doctor_id']}", width="stretch"):
            ok, res = api_post(
                f"/patient/doctor-grants/{pid}",
                {"doctor_id": g["doctor_id"], "status": new_status},
            )
            if ok:
                st.rerun()
            else:
                st.error(res)


def render_patient_portal():
    if not _require_role("patient", "Patient Login"):
        return
    user = current_user()
    pid = user["id"]

    head, out = st.columns([4, 1], vertical_alignment="center")
    with head:
        st.markdown(
            f'<div class="section-title">Welcome, {_escape(user["full_name"])}</div>'
            f'<div class="section-caption">Patient ID: PAT-{pid}</div>',
            unsafe_allow_html=True,
        )
    with out:
        if st.button("Logout", key="btn_logout_patient", width="stretch"):
            logout()

    tab_profile, tab_history, tab_preds, tab_access = st.tabs(
        ["Profile", "Medical History", "Prediction History", "Doctor Access"]
    )
    with tab_profile:
        _patient_profile_tab(pid)
    with tab_history:
        _patient_history_form_tab(pid)
    with tab_preds:
        _patient_predictions_tab(pid)
    with tab_access:
        _patient_access_tab(pid)


# -----------------------------------------------------------------------------
# Doctor portal
# -----------------------------------------------------------------------------
def _show_patient_record(rec: dict):
    if not rec["authorized"]:
        st.warning(rec["message"])
        return
    st.success(rec["message"])

    prof = rec["patient_profile"]
    u = prof["user"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Patient", u["full_name"], f"PAT-{u['id']}")
    with c2:
        _metric("Age", u.get("age") or "-", "years")
    with c3:
        _metric("Gender", u.get("gender") or "-")
    with c4:
        _metric("Blood Group", prof.get("blood_group") or "-")

    st.markdown("**Patient-reported medical history**")
    mh = rec.get("medical_history")
    if mh:
        labels = {
            "existing_conditions": "Existing conditions",
            "heart_problems": "Heart problems",
            "family_history": "Family history",
            "current_medications": "Current medications",
            "lifestyle_notes": "Lifestyle",
            "additional_notes": "Additional notes",
        }
        st.table(pd.DataFrame({"Field": list(labels.values()), "Details": [mh.get(k) or "-" for k in labels]}))

    st.markdown("**Prediction history**")
    if rec["predictions"]:
        st.dataframe(_prediction_rows(rec["predictions"]), width="stretch", hide_index=True)
    else:
        st.info("This patient has no saved predictions.")


def _doctor_patients_tab(did: int):
    search = st.text_input("Search by patient name, email, or ID (example: PAT-3)", key="doc_search")
    ok, patients = api_get(f"/doctor/patients/{did}", params={"search": search})
    if not ok:
        st.error(patients)
        return
    if not patients:
        st.info("No patients found.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Patient ID": f"PAT-{p['patient_id']}",
                    "Name": p["full_name"],
                    "Age": p.get("age"),
                    "Gender": p.get("gender"),
                    "Access": p["access_status"],
                    "Predictions": p["predictions_count"],
                    "Last Prediction": p.get("last_prediction_date"),
                }
                for p in patients
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    options = {f"PAT-{p['patient_id']} - {p['full_name']} ({p['access_status']})": p["patient_id"] for p in patients}
    choice = st.selectbox("Select patient", list(options.keys()))
    if st.button("Open Patient Record", key="btn_open_record", type="primary"):
        ok, rec = api_get(f"/doctor/patient/{did}/{options[choice]}")
        if ok:
            st.session_state.opened_record = rec
        else:
            st.error(rec)

    if st.session_state.get("opened_record"):
        st.divider()
        _show_patient_record(st.session_state.opened_record)


def _doctor_audit_tab(did: int):
    ok, logs = api_get(f"/doctor/audit-logs/{did}")
    if not ok:
        st.error(logs)
        return
    if not logs:
        st.info("No audit entries yet.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Time": l["timestamp"],
                    "Action": l["action"],
                    "Patient": f"PAT-{l['target_patient_id']}" if l.get("target_patient_id") else "-",
                    "Details": l.get("details"),
                }
                for l in logs
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render_doctor_portal():
    if not _require_role("doctor", "Doctor Login"):
        return
    user = current_user()
    did = user["id"]

    head, out = st.columns([4, 1], vertical_alignment="center")
    with head:
        st.markdown(
            f'<div class="section-title">{_escape(user["full_name"])}</div>'
            f'<div class="section-caption">Doctor Clinical Portal</div>',
            unsafe_allow_html=True,
        )
    with out:
        if st.button("Logout", key="btn_logout_doctor", width="stretch"):
            logout()

    ok, summary = api_get(f"/doctor/summary/{did}")
    if ok:
        m1, m2, m3 = st.columns(3)
        with m1:
            _metric("Registered Patients", summary["total_patients"])
        with m2:
            _metric("Access Granted", summary["authorized_patients"])
        with m3:
            _metric("Records Reviewed", summary["recent_predictions_reviewed"])
        st.write("")
    else:
        st.error(summary)

    tab_patients, tab_audit = st.tabs(["Patients", "Audit Log"])
    with tab_patients:
        _doctor_patients_tab(did)
    with tab_audit:
        _doctor_audit_tab(did)