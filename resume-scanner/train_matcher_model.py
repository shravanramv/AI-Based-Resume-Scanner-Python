import os
import time
import numpy as np
import pandas as pd
import PyPDF2
import docx
import re
from tqdm import tqdm
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sentence_transformers import SentenceTransformer

# ⏱️ Utility function for timing
def log_time(message):
    print(f"\n🕒 {message} at {time.strftime('%H:%M:%S')}")

# 📄 Extract text from documents
def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# 🔤 Basic text cleaning
def basic_clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())

# 💬 Extract soft skills + sentiment
def extract_soft_skills(text):
    soft_skills = [
        "teamwork", "communication", "adaptability", "leadership", "creativity",
        "critical thinking", "time management", "collaboration", "empathy"
    ]
    found = [skill for skill in soft_skills if skill in text.lower()]
    sentiment = TextBlob(text).sentiment.polarity
    return found, sentiment

# 📊 Load and process dataset
log_time("Loading labeled_pairs.csv")
df = pd.read_csv("labeled_pairs.csv")

features, labels = [], []
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

log_time("Extracting features from resumes and JDs")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Processing"):
    # Resume text
    res_txt = extract_text_from_pdf(row.resume_path) \
        if row.resume_path.lower().endswith(".pdf") else extract_text_from_docx(row.resume_path)
    
    # JD text
    jd_txt = extract_text_from_pdf(row.jd_path) \
        if row.jd_path.lower().endswith(".pdf") else extract_text_from_docx(row.jd_path)
    
    # Clean and combine
    res_clean = basic_clean(res_txt)
    jd_clean = basic_clean(jd_txt)
    combined_text = res_clean + " " + jd_clean

    # 🔠 BERT embedding
    embedding = bert_model.encode(combined_text)

    # 💡 Soft skill count + sentiment score
    soft_skills_found, sentiment = extract_soft_skills(res_txt)
    soft_skill_count = len(soft_skills_found)

    # Combine all features
    feature_vector = np.append(embedding, [soft_skill_count, sentiment])
    features.append(feature_vector)
    labels.append(row.label)

X = np.vstack(features)
y = np.array(labels)

# 📈 Train/test split
log_time("Splitting dataset")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ⚖️ Normalize for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🧠 Train model
log_time("Training Logistic Regression")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ✅ Evaluation
log_time("Evaluating model")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("📈 ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 🔁 Cross-validation
log_time("Running 5-fold cross-validation")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print("🔁 5-fold F1 scores:", cv_scores)
print("📊 Mean F1:", cv_scores.mean())

# 💾 Save artifacts
log_time("Saving model artifacts")
os.makedirs("models", exist_ok=True)
dump(model, "models/resume_matcher_bert.joblib")
dump(scaler, "models/scaler.joblib")
dump(bert_model, "models/bert_encoder.joblib")

print("✅ Model and vectorizer saved in /models")
log_time("All tasks complete")
