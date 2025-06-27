from uuid import uuid4
import streamlit as st
import tempfile
import os
import PyPDF2
import docx
import re
import numpy as np
from joblib import load
from utils.database import connect_db

# --- Page Config ---
st.set_page_config(page_title="Recruiter Dashboard", page_icon="🧑‍💼", layout="wide")

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

# --- Authentication Check ---
if 'username' not in st.session_state or st.session_state['role'] != 'recruiter':
    st.error("Unauthorized access")
    st.stop()

username = st.session_state['username']
st.markdown("<h2 style='text-align: center;'>🧑‍💼 Recruiter Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Post a Job ---
st.subheader("➕ Post a New Job")
with st.form("post_job_form"):
    job_title = st.text_input("Job Title")
    job_description = st.text_area("Job Description")
    submit = st.form_submit_button("Post Job")

    if submit:
        if job_title and job_description:
            conn = connect_db()
            c = conn.cursor()
            c.execute("""
                INSERT INTO jobs (job_title, job_description, recruiter_username)
                VALUES (?, ?, ?)
            """, (job_title, job_description, username))
            conn.commit()
            conn.close()
            st.success("✅ Job posted successfully!")
            st.rerun()
        else:
            st.warning("Please fill in both job title and description.")

# --- Load Models ---
try:
    bert_model = load("models/bert_encoder.joblib")
    scaler = load("models/scaler.joblib")
    model = load("models/resume_matcher_Logistic_Regression.joblib")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Resume Text Extraction ---
def extract_text(file_bytes, filename):
    ext = filename.lower().split('.')[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    temp_file.write(file_bytes)
    temp_file.close()
    text = ""
    try:
        if ext == "pdf":
            reader = PyPDF2.PdfReader(temp_file.name)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext in ["docx", "doc"]:
            docx_file = docx.Document(temp_file.name)
            text = "\n".join([p.text for p in docx_file.paragraphs])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    finally:
        os.remove(temp_file.name)
    return text

# --- Resume Score Cache ---
if "resume_score_cache" not in st.session_state:
    st.session_state.resume_score_cache = {}

# --- Posted Jobs ---
st.subheader("📋 Your Posted Jobs & Applications")
conn = connect_db()
c = conn.cursor()
c.execute("SELECT id, job_title, job_description FROM jobs WHERE recruiter_username = ?", (username,))
jobs = c.fetchall()

if not jobs:
    st.info("You haven't posted any jobs yet.")
else:
    for job_id, title, desc in jobs:
        with st.expander(f"💼 {title}"):
            st.write(desc)

            c.execute("""
                SELECT applicant_username, resume, resume_filename
                FROM applications WHERE job_id = ?
            """, (job_id,))
            applications = c.fetchall()

            if not applications:
                st.warning("No applications yet.")
                continue

            st.markdown("**📄 Applications:**")
            resume_scores = []
            cache = st.session_state.resume_score_cache.get(job_id, {})

            for applicant_username, resume_blob, resume_filename in applications:
                key = f"{applicant_username}_{resume_filename}"
                if key in cache:
                    score = cache[key]["score"]
                else:
                    resume_text = extract_text(resume_blob, resume_filename)
                    if not resume_text:
                        continue

                    # --- BERT Encode ---
                    resume_emb = bert_model.encode([resume_text])[0]
                    jd_emb = bert_model.encode([desc])[0]

                    # --- Feature Vector ---
                    sim = np.dot(resume_emb, jd_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb))
                    dot = np.dot(resume_emb, jd_emb)
                    features = np.hstack([resume_emb, jd_emb, [sim, dot]])
                    scaled = scaler.transform([features])
                    score = model.predict_proba(scaled)[0][1]

                    cache[key] = {
                        "score": score,
                        "resume_blob": resume_blob,
                        "filename": resume_filename
                    }

                resume_scores.append((
                    applicant_username,
                    cache[key]["score"],
                    cache[key]["filename"],
                    cache[key]["resume_blob"]
                ))

            st.session_state.resume_score_cache[job_id] = cache

            # --- Filter Slider ---
            threshold = st.slider(
                f"🎯 Minimum Match Score to Show (Job: {title})",
                0.0, 1.0, 0.0, 0.01,
                key=f"slider_{job_id}"
            )

            filtered_scores = [r for r in resume_scores if r[1] >= threshold]
            filtered_scores.sort(key=lambda x: x[1], reverse=True)

            for idx, (applicant_username, score, filename, resume_blob) in enumerate(filtered_scores, 1):
                st.markdown(f"**{idx}. 👤 {applicant_username} — Match Score: `{score:.2f}`**")
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=resume_blob,
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{uuid4()}"
                )
                st.markdown("---")

            # --- Resume Analysis ---
            if st.button(f"📊 Analyze Resumes for '{title}'", key=f"analyze_{job_id}"):
                st.session_state['selected_job_id'] = job_id
                st.switch_page("pages/analyze_resumes.py")

conn.close()
