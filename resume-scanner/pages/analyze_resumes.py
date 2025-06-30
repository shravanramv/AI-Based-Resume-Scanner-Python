from uuid import uuid4
import streamlit as st
import pandas as pd
from utils.database import connect_db
from joblib import load
import tempfile
import os
import PyPDF2
import docx
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Soft Skill Matching ---
SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "adaptability", "problem-solving",
    "creativity", "time management", "empathy", "collaboration", "emotional intelligence"
]

def count_soft_skills_match(resume_text, jd_text):
    resume_skills = [s for s in SOFT_SKILLS if s in resume_text.lower()]
    jd_skills = [s for s in SOFT_SKILLS if s in jd_text.lower()]
    if not jd_skills:
        return 0, []
    matched = set(resume_skills) & set(jd_skills)
    return len(matched) / len(jd_skills), list(matched)

# --- Page Config ---
st.set_page_config(page_title="Resume Analysis", page_icon="📊", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .card {
            border-radius: 12px;
            padding: 25px;
            background-color: #fff;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        }
        .section-title {
            font-weight: 600;
            font-size: 22px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper: Resume Text Extraction ---
def extract_text(file_bytes, filename):
    ext = filename.lower().split('.')[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    temp_file.write(file_bytes)
    temp_file.close()

    text = ""
    try:
        if ext == "pdf":
            reader = PyPDF2.PdfReader(temp_file.name)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ext in ["docx", "doc"]:
            docx_file = docx.Document(temp_file.name)
            text = "\n".join([p.text for p in docx_file.paragraphs])
    finally:
        os.remove(temp_file.name)

    return text

# --- Helper: Section Splitting ---
def split_sections(text):
    sections = {"Education": "", "Experience": "", "Skills": ""}
    lines = text.splitlines()
    current = None
    for line in lines:
        lower = line.lower()
        if "education" in lower:
            current = "Education"
        elif "experience" in lower:
            current = "Experience"
        elif "skill" in lower:
            current = "Skills"
        elif current:
            sections[current] += " " + line
    return sections

# --- Session Check ---
if 'username' not in st.session_state or st.session_state['role'] != 'recruiter':
    st.error("Unauthorized access")
    st.stop()

if 'selected_job_id' not in st.session_state:
    st.warning("No job selected.")
    st.stop()

# --- Load Models ---
try:
    bert_model = load("models/bert_encoder.joblib")
    scaler = load("models/scaler.joblib")
    model = load("models/resume_matcher_Logistic_Regression.joblib")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

job_id = st.session_state['selected_job_id']

# --- Fetch Job Info ---
conn = connect_db()
c = conn.cursor()
c.execute("SELECT job_title, job_description FROM jobs WHERE id = ?", (job_id,))
job_data = c.fetchone()

if not job_data:
    st.error("Job not found.")
    st.stop()

job_title, job_description = job_data
st.title(f"📊 Resume Analysis for: {job_title}")

# --- Fetch Applications ---
c.execute("""
    SELECT applicant_username, resume, resume_filename
    FROM applications
    WHERE job_id = ?
""", (job_id,))
applications = c.fetchall()

if not applications:
    st.warning("No resumes submitted yet.")
    st.stop()

# --- Processing ---
scores = []
section_scores_map = {}

for username, resume_blob, filename in applications:
    resume_text = extract_text(resume_blob, filename)

    # Encode full resume + JD
    resume_emb = bert_model.encode([resume_text])[0]
    jd_emb = bert_model.encode([job_description])[0]

    # Soft skill score
    soft_score, matched_soft_skills = count_soft_skills_match(resume_text, job_description)

    cos_sim = np.dot(resume_emb, jd_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb))
    dot = np.dot(resume_emb, jd_emb)
    full_features = np.hstack([resume_emb, jd_emb, [cos_sim, dot, soft_score]])
    full_scaled = scaler.transform([full_features])
    overall_score = round(model.predict_proba(full_scaled)[0][1], 2)

    # Encode each section
    sections = split_sections(resume_text)
    section_scores = {}
    for sec, sec_text in sections.items():
        sec_emb = bert_model.encode([sec_text])[0]
        sec_sim = np.dot(sec_emb, jd_emb) / (np.linalg.norm(sec_emb) * np.linalg.norm(jd_emb))
        sec_dot = np.dot(sec_emb, jd_emb)
        sec_features = np.hstack([sec_emb, jd_emb, [sec_sim, sec_dot, soft_score]])
        sec_scaled = scaler.transform([sec_features])
        sec_score = round(model.predict_proba(sec_scaled)[0][1], 2)
        section_scores[sec] = sec_score

    scores.append({
        "Applicant": username,
        "Filename": filename,
        "Match Score": overall_score,
        "ResumeBlob": resume_blob,
        "Matched Soft Skills": ", ".join(matched_soft_skills) if matched_soft_skills else "-"
    })
    section_scores_map[username] = section_scores

# --- Create DataFrame ---
df = pd.DataFrame(scores).sort_values(by="Match Score", ascending=False)

# --- Filter by Score ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🎯 Filter Applicants by Score")
min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.01)
filtered_df = df[df["Match Score"] >= min_score]
st.dataframe(filtered_df.drop(columns=["ResumeBlob"]), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Summary Metrics ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.metric("📈 Average Match Score", f"{filtered_df['Match Score'].mean():.2f}" if not filtered_df.empty else "0.00")
with col2:
    st.metric("🧑 Applicants Shown", len(filtered_df))
st.markdown("</div>", unsafe_allow_html=True)

# --- Histogram ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📉 Match Score Distribution")
fig, ax = plt.subplots()
ax.hist(df["Match Score"], bins=5, color="#4da6ff", edgecolor="black")
ax.set_xlabel("Match Score")
ax.set_ylabel("Number of Applicants")
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# --- Section Scores ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🔍 Per-Section Resume Match Scores")
for row in filtered_df.itertuples():
    username = row.Applicant
    st.markdown(f"**👤 {username}**")
    st.table(pd.DataFrame([section_scores_map[username]]))

    if row._asdict().get("Matched Soft Skills") and row._asdict()["Matched Soft Skills"] != "-":
        st.markdown(f"✅ **Matched Soft Skills**: {row._asdict()['Matched Soft Skills']}")

    st.download_button(
        label=f"📥 Download {row.Filename}",
        data=row.ResumeBlob,
        file_name=row.Filename,
        mime="application/octet-stream",
        key=f"download_{username}_{uuid4()}"
    )
st.markdown("</div>", unsafe_allow_html=True)

conn.close()
