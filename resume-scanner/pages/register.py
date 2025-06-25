import streamlit as st
import sqlite3

# --- Page config ---
st.set_page_config(page_title="Register", page_icon="📝", layout="wide")

# --- Load CSS ---
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# --- Initialize DB ---
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT CHECK(role IN ('recruiter', 'applicant')),
        company TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter_username TEXT,
        job_title TEXT,
        job_description TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_username TEXT,
        job_id INTEGER,
        resume BLOB,
        resume_filename TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs(id)
    )''')
    conn.commit()
    conn.close()

init_db()

# --- Register User Function ---
def register_user(username, password, role, company):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, role, company) VALUES (?, ?, ?, ?)",
                  (username, password, role, company))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# --- Centered Header ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>📝 Create Your Account</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Start your journey as a recruiter or applicant.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Centered Form ---
col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    role = st.selectbox("Register as", ["Recruiter", "Applicant"])
    with st.form("registration_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        company = st.text_input("Company Name") if role == "Recruiter" else None

        submitted = st.form_submit_button("Register", type="primary")

        if submitted:
            if not username or not password or (role == "Recruiter" and not company):
                st.warning("Please fill in all required fields.")
            else:
                success = register_user(username, password, role.lower(), company)
                if success:
                    st.success("✅ Registration successful! You can now log in.")
                else:
                    st.error("❌ Username already exists.")
