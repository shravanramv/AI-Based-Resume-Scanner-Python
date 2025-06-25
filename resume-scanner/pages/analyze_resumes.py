from uuid import uuid4
import streamlit as st
import pandas as pd
from database import connect_db
from joblib import load
import tempfile
import os
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Resume Analysis", page_icon="📊", layout="wide")

# --- NLTK Setup ---
if not hasattr(st.session_state, "nltk_loaded"):
    nltk.download("stopwords")
    st.session_state.nltk_loaded = True

STOPWORDS = set(stopwords.words("english"))

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

# --- Helpers ---
def extract_text(file_bytes, filename):
    ext = filename.lower().split('.')[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    temp_file.write(file_bytes)
    temp_file.close()

    text = ""
    if ext == "pdf":
        reader = PyPDF2.PdfReader(temp_file.name)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext in ["docx", "doc"]:
        doc = docx.Document(temp_file.name)
        text = "\n".join([p.text for p in doc.paragraphs])

    os.remove(temp_file.name)
    return text

def basic_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

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

# --- Session & Role Check ---
if 'username' not in st.session_state or st.session_state['role'] != 'recruiter':
    st.error("Unauthorized access")
    st.stop()

if 'selected_job_id' not in st.session_state:
    st.warning("No job selected.")
    st.stop()

# --- Load ML Model ---
model = load("models/resume_matcher.joblib")
vectorizer = load("models/tfidf_vectorizer.joblib")
job_id = st.session_state['selected_job_id']

# --- DB Fetch ---
conn = connect_db()
c = conn.cursor()
c.execute("SELECT job_title, job_description FROM jobs WHERE id = ?", (job_id,))
job_data = c.fetchone()

if not job_data:
    st.error("Job not found.")
    st.stop()

job_title, job_description = job_data
st.title(f"📊 Resume Analysis for: {job_title}")

c.execute("""
    SELECT applicant_username, resume, resume_filename
    FROM applications
    WHERE job_id = ?
""", (job_id,))
applications = c.fetchall()

if not applications:
    st.warning("No resumes submitted yet.")
    st.stop()

# --- Score Processing ---
scores = []
section_scores_map = {}

for username, resume_blob, filename in applications:
    resume_text = extract_text(resume_blob, filename)
    combined = basic_preprocess(resume_text + " " + job_description)
    vectorized = vectorizer.transform([combined])
    overall_score = model.predict_proba(vectorized)[0][1]

    # Section-level matching
    sections = split_sections(resume_text)
    section_scores = {}
    for sec, sec_text in sections.items():
        combined_sec = basic_preprocess(sec_text + " " + job_description)
        vec = vectorizer.transform([combined_sec])
        section_scores[sec] = round(model.predict_proba(vec)[0][1], 2)

    scores.append({
        "Applicant": username,
        "Filename": filename,
        "Match Score": round(overall_score, 2),
        "ResumeBlob": resume_blob
    })
    section_scores_map[username] = section_scores

# --- DataFrame ---
df = pd.DataFrame(scores).sort_values(by="Match Score", ascending=False)

# --- Filter UI ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🎯 Filter Applicants by Score")
min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.01)
filtered_df = df[df["Match Score"] >= min_score]
st.dataframe(filtered_df.drop(columns=["ResumeBlob"]), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Summary Metrics ---
avg_score = filtered_df["Match Score"].mean() if not filtered_df.empty else 0.0

st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.metric("📈 Average Match Score", f"{avg_score:.2f}")
with col2:
    st.metric("🧑 Applicants Shown", len(filtered_df))
st.markdown("</div>", unsafe_allow_html=True)

# --- Score Distribution ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📉 Match Score Distribution")
fig, ax = plt.subplots()
ax.hist(df["Match Score"], bins=5, color="#77c3ec", edgecolor="black")
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
    st.download_button(
        label=f"📥 Download {row.Filename}",
        data=row.ResumeBlob,
        file_name=row.Filename,
        mime="application/octet-stream",
        key=f"download_{username}_{uuid4()}"
    )
st.markdown("</div>", unsafe_allow_html=True)

conn.close()
