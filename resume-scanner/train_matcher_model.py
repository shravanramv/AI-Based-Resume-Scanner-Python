import os
import random
import re
import numpy as np
import PyPDF2
import docx
from tqdm import tqdm
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "adaptability", "creativity",
    "critical thinking", "problem solving", "time management", "collaboration",
    "empathy", "resilience", "flexibility", "initiative", "work ethic"
]

# --- Extractor
def extract_text(path):
    ext = path.lower().split(".")[-1]
    try:
        if ext == "pdf":
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ext == "docx":
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""
    return ""

# --- Preprocessing with soft skill boosting
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    tokens = text.split()
    boosted = tokens[:]
    for skill in SOFT_SKILLS:
        if skill in tokens:
            boosted.extend([skill] * 2)
    return " ".join(boosted)

def log(msg): print(f"\n🟢 {msg}")

# --- Load dataset structure
base_dir = "data/resumes"
categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

pairs, labels = [], []

log("Generating synthetic job descriptions and training pairs...")

for category in tqdm(categories, desc="📁 Processing categories"):
    cat_path = os.path.join(base_dir, category)
    resumes = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.endswith((".pdf", ".docx"))]

    if len(resumes) < 2:
        continue  # skip categories with too few samples

    jd_text = extract_text(resumes[0])  # use first resume as JD base
    jd_text = preprocess(jd_text)

    # --- Positive examples
    for res in resumes[1:]:
        res_text = extract_text(res)
        if res_text.strip():
            combined = preprocess(res_text + " " + jd_text)
            pairs.append(combined)
            labels.append(1)

    # --- Negative examples (pick from other random categories)
    other_resumes = []
    for other_cat in categories:
        if other_cat != category:
            other_cat_path = os.path.join(base_dir, other_cat)
            files = [os.path.join(other_cat_path, f) for f in os.listdir(other_cat_path) if f.endswith((".pdf", ".docx"))]
            other_resumes.extend(files)

    random.shuffle(other_resumes)
    for neg_res in other_resumes[:len(resumes) - 1]:
        res_text = extract_text(neg_res)
        if res_text.strip():
            combined = preprocess(res_text + " " + jd_text)
            pairs.append(combined)
            labels.append(0)

log(f"Total training pairs: {len(pairs)}")

log("Vectorizing...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(pairs)
y = np.array(labels)

log("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

log("Training Logistic Regression...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

log("Running 5-fold CV...")
f1 = cross_val_score(model, X, y, cv=5, scoring="f1")
print("📈 Avg F1:", f1.mean())

log("Saving model + vectorizer")
os.makedirs("models", exist_ok=True)
dump(model, "models/resume_matcher.joblib")
dump(tfidf, "models/tfidf_vectorizer.joblib")

log("✅ All Done")
