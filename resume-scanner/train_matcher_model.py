import os
import time
import numpy as np
import pandas as pd
import PyPDF2
import docx
import re
from tqdm import tqdm
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# --- Soft skills list for boosting
SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "adaptability", "creativity",
    "critical thinking", "problem solving", "time management", "collaboration",
    "empathy", "resilience", "flexibility", "initiative", "work ethic"
]

# --- PDF / DOCX text extraction
def extract_text(path):
    ext = path.lower().split(".")[-1]
    try:
        if ext == "pdf":
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == "docx":
            doc = docx.Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return ""
    except Exception as e:
        print(f"❌ Failed to extract {path}: {e}")
        return ""

# --- Basic text cleaner + soft skill booster
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    tokens = text.split()
    boosted = tokens[:]
    for skill in SOFT_SKILLS:
        if skill in tokens:
            boosted.extend([skill] * 2)  # boost weight
    return " ".join(boosted)

# --- Section extractor for scoring
def split_sections(text):
    sections = {"education": "", "experience": "", "skills": ""}
    current = ""
    for line in text.splitlines():
        line_lower = line.lower()
        if "education" in line_lower:
            current = "education"
        elif "experience" in line_lower:
            current = "experience"
        elif "skill" in line_lower:
            current = "skills"
        elif current:
            sections[current] += " " + line
    return sections

# --- START: Training process
def log(msg): print(f"\n🟩 {msg} @ {time.strftime('%H:%M:%S')}")

log("Loading labeled_pairs.csv")
df = pd.read_csv("labeled_pairs.csv")

texts, labels = [], []

log("Extracting and preprocessing text")
for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Processing"):
    resume = extract_text(row.resume_path)
    jd = extract_text(row.jd_path)

    # Combine all + soft skills
    combined = preprocess(resume + " " + jd)
    texts.append(combined)
    labels.append(row.label)

log("Vectorizing with TF-IDF")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(texts)
y = np.array(labels)

log("Splitting train/test set")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

log("Training Logistic Regression")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

log("Evaluating")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"✅ ROC-AUC:  {roc_auc_score(y_test, y_proba):.2f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

log("5-Fold Cross-Validation (F1 Score)")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f"📈 Avg F1 Score: {cv_scores.mean():.2f}")

log("Saving model and vectorizer")
os.makedirs("models", exist_ok=True)
dump(model, "models/resume_matcher.joblib")
dump(tfidf, "models/tfidf_vectorizer.joblib")

log("✅ All Done!")
