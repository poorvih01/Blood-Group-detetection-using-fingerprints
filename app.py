import streamlit as st
import cv2
import numpy as np
import joblib
import hashlib
import re
from datetime import datetime
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.feature import local_binary_pattern
from fpdf import FPDF

# ================= CONFIG =================
IMG_SIZE = 224
USERS_FILE = "users.pkl"

st.set_page_config(page_title="Fingerprint Blood Group Detection", layout="centered")

# ================= LOAD MODEL =================
model = joblib.load("bloodgroup_model.pkl")
scaler = joblib.load("scaler.pkl")
label_map = joblib.load("label_map.pkl")
rev_label_map = {v: k for k, v in label_map.items()}

# ================= AUTH =================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    try:
        return joblib.load(USERS_FILE)
    except:
        return {}

def save_users(users):
    joblib.dump(users, USERS_FILE)

users = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================= IMAGE PROCESSING =================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t = threshold_otsu(blur)
    binary = blur < t
    skeleton = skeletonize(binary)
    return skeleton.astype(np.uint8)

def is_fingerprint(skeleton):
    ridge_density = np.sum(skeleton) / (IMG_SIZE * IMG_SIZE)
    return ridge_density > 0.02

def extract_features(skeleton):
    ridge_density = np.sum(skeleton) / (IMG_SIZE * IMG_SIZE)
    y, x = np.where(skeleton == 1)
    curvature = np.std(np.gradient(x)) if len(x) > 10 else 0
    lbp = local_binary_pattern(skeleton, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return np.concatenate([[ridge_density, curvature], lbp_hist])

# ================= REPORT =================
def generate_report(patient, blood_group):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "FINGERPRINT BASED BLOOD GROUP DETECTION REPORT", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", size=11)

    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ✅ FIXED: correct name mapping
    pdf.cell(0, 8, f"Name           : {patient['name']}", ln=True)
    pdf.cell(0, 8, f"Age            : {patient['age']}", ln=True)
    pdf.cell(0, 8, f"Gender         : {patient['gender']}", ln=True)
    pdf.cell(0, 8, f"Phone Number   : {patient['phone']}", ln=True)

    pdf.ln(6)
    pdf.cell(0, 8, "Test Details", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.cell(0, 8, "Input Type     : Fingerprint Image", ln=True)
    pdf.cell(0, 8, "Method Used    : Machine Learning Classification", ln=True)
    pdf.cell(0, 8, f"Date & Time    : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Predicted Blood Group : {blood_group}", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(
        0, 8,
        "IMPORTANT NOTE:\n"
        "This result is generated purely for academic and research purposes. "
        "This system is NOT intended for medical diagnosis. "
        
    )

    pdf.ln(4)
    

    file_path = "Fingerprint_Blood_Group_Report.pdf"
    pdf.output(file_path)
    return file_path

# ================= LOGIN UI =================
def login_ui():
    tab1, tab2 = st.tabs(["Login", "New User"])

    with tab1:
        st.subheader("🔐 Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in users and users[u] == hash_pw(p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.subheader("📝 New User Registration")
        nu = st.text_input("New Username")
        npw = st.text_input("New Password", type="password")

        if st.button("Register"):
            if nu.strip() == "" or npw.strip() == "":
                st.error("Fields cannot be empty")
            elif nu in users:
                st.error("User already exists")
            else:
                users[nu] = hash_pw(npw)
                save_users(users)
                st.success("Registration successful. Please login.")

# ================= PATIENT DETAILS =================
def patient_form():
    st.subheader("🧾 Patient Details")

    with st.form("patient_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        phone = st.text_input("Phone Number")
        submit = st.form_submit_button("Proceed")

    if submit:
        if name.strip() == "":
            st.error("Enter patient name")
            return False
        if gender == "Select":
            st.error("Select gender")
            return False
        if not re.fullmatch(r"\d{10}", phone):
            st.error("Phone number must contain exactly 10 digits")
            return False

        st.session_state.patient = {
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone
        }
        return True
    return False

# ================= MAIN =================
st.title("🩸 Fingerprint Blood Group Detection System")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

if "patient" not in st.session_state:
    if patient_form():
        st.rerun()
    st.stop()

st.subheader("📤 Upload Fingerprint Images")

uploaded_files = st.file_uploader(
    "Upload fingerprint images ",
    type=["bmp", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 2:
        st.error("❌ You can upload only 2 images at a time.")
        st.stop()

    predicted_group = None

    for file in uploaded_files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        skeleton = preprocess(img)

        if not is_fingerprint(skeleton):
            continue

        features = extract_features(skeleton)
        features = scaler.transform([features])
        pred = model.predict(features)[0]
        predicted_group = rev_label_map[pred]
        break

    if predicted_group is None:
        st.error("❌ Unable to recognize fingerprint. Please upload a clear fingerprint image.")
    else:
        st.success(f"🩸 Predicted Blood Group: **{predicted_group}**")

        report_path = generate_report(st.session_state.patient, predicted_group)
        with open(report_path, "rb") as f:
            st.download_button(
                "📄 Download Result Report",
                f,
                file_name="Fingerprint_Blood_Group_Report.pdf"
            )

        st.success("✔ Report generated successfully.")

        # ✅ CLEAR UPLOADED IMAGE & RESET FLOW
        if st.button("🔄 Upload New Fingerprint"):
            del st.session_state["patient"]
            st.rerun()

# ================= LOGOUT =================
if st.button("Logout"):
    st.session_state.clear()
    st.rerun()
