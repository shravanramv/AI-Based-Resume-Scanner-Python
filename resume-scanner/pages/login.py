import streamlit as st
import sqlite3
import os

# --- Page Config ---
st.set_page_config(page_title="Login", page_icon="🔐", layout="wide")

# --- Load CSS ---
def load_css():
    try:
        with open("static/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found.")

load_css()

# --- Database path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DB_PATH = os.path.join(BASE_DIR, "..", "databases", "users.db")

# --- Login logic ---
def login_user(username, password):
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# --- Centered Header ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>🔐 Login to Your Account</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Access your dashboard and manage jobs or applications.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Centered Login Form ---
col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login", type="primary")

        if login_button:
            if not username or not password:
                st.warning("⚠️ Please enter both username and password.")
            else:
                role = login_user(username, password)
                if role:
                    st.session_state["username"] = username
                    st.session_state["role"] = role
                    st.success(f"✅ Logged in as {role.capitalize()}!")

                    if role == "recruiter":
                        st.switch_page("pages/recruiter_dashboard.py")
                    else:
                        st.switch_page("pages/applicant_dashboard.py")
                else:
                    st.error("❌ Invalid username or password.")
