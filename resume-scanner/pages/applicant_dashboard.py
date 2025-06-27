import streamlit as st
import sqlite3
from utils.database import connect_db  # Shared DB layer

# --- Page config ---
st.set_page_config(page_title="Applicant Dashboard", page_icon="🧑‍🎓", layout="wide")

# --- Load CSS ---
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# --- Logout Button ---
top_col1, top_col2 = st.columns([9, 1])
with top_col2:
    if st.button("Logout", key="logout_button"):
        st.session_state.clear()
        st.switch_page("app.py")

# --- Auth Check ---
if 'username' not in st.session_state or st.session_state['role'] != 'applicant':
    st.error("Unauthorized access")
    st.stop()

username = st.session_state['username']
st.markdown(f"<h2 style='text-align: center;'>🧑‍🎓 Welcome, {username}!</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Search Bar ---
st.subheader("💼 Available Jobs")
search_query = st.text_input("🔍 Search jobs by title or description")

# --- DB Setup ---
conn = connect_db()
c = conn.cursor()

# --- Fetch Applied Job IDs ---
c.execute("SELECT job_id FROM applications WHERE applicant_username = ?", (username,))
applied_job_ids = set(row[0] for row in c.fetchall())

# --- Fetch Jobs ---
if search_query:
    like_query = f"%{search_query.lower()}%"
    c.execute("""
        SELECT id, job_title, job_description FROM jobs
        WHERE LOWER(job_title) LIKE ? OR LOWER(job_description) LIKE ?
    """, (like_query, like_query))
else:
    c.execute("SELECT id, job_title, job_description FROM jobs")

all_jobs = c.fetchall()

# --- Categorize ---
available_jobs = [job for job in all_jobs if job[0] not in applied_job_ids]
applied_jobs = [job for job in all_jobs if job[0] in applied_job_ids]

# --- Available Jobs ---
if available_jobs:
    for job_id, title, desc in available_jobs:
        with st.expander(f"🔹 {title}"):
            st.write(desc)
            with st.form(f"apply_form_{job_id}"):
                uploaded_file = st.file_uploader(
                    "📤 Upload your resume (PDF, DOC, DOCX, TXT)",
                    type=["pdf", "doc", "docx", "txt"]
                )
                submitted = st.form_submit_button("Apply")

                if submitted:
                    if uploaded_file:
                        resume_data = uploaded_file.read()
                        c.execute("""
                            INSERT INTO applications (applicant_username, job_id, resume, resume_filename)
                            VALUES (?, ?, ?, ?)
                        """, (username, job_id, resume_data, uploaded_file.name))
                        conn.commit()
                        st.success("✅ Resume submitted! This job will now appear in 'Applied Jobs'.")
                        st.rerun()
                    else:
                        st.warning("Please upload a resume before submitting.")
else:
    st.info("No new jobs available, or you've already applied to all of them.")

# --- Applied Jobs Section ---
st.subheader("📁 Applied Jobs")
if applied_jobs:
    for job_id, title, desc in applied_jobs:
        with st.expander(f"✅ {title}"):
            st.write(desc)

            c.execute("""
                SELECT resume, resume_filename FROM applications
                WHERE job_id = ? AND applicant_username = ?
            """, (job_id, username))
            result = c.fetchone()

            if result:
                resume_data, resume_filename = result
                st.download_button(
                    label=f"📥 Download Submitted Resume: {resume_filename}",
                    data=resume_data,
                    file_name=resume_filename,
                    mime="application/octet-stream",
                    key=f"download_{job_id}"
                )
else:
    st.info("You haven’t applied to any jobs yet.")

conn.close()
