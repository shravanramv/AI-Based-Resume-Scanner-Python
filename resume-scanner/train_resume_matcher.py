import os
import random
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import PyPDF2
import docx
import re
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ==== Soft Skills List ====
SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "adaptability", "problem-solving",
    "creativity", "time management", "empathy", "collaboration", "emotional intelligence"
]

def count_soft_skills_match(resume_text, jd_text):
    resume_skills = [s for s in SOFT_SKILLS if s in resume_text.lower()]
    jd_skills = [s for s in SOFT_SKILLS if s in jd_text.lower()]
    if not jd_skills:
        return 0
    matched = set(resume_skills) & set(jd_skills)
    return len(matched) / len(jd_skills)

# ==== Timing Helper ====
def log_time(msg):
    print(f"\n🕒 {msg} at {time.strftime('%H:%M:%S')}")

# ==== Paths ====
RESUME_DIR = "data/resumes"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==== Load BERT ====
log_time("Loading Sentence-BERT")
bert = SentenceTransformer("all-MiniLM-L6-v2")

# ==== Utility Functions ====
def extract_text(path):
    try:
        if path.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(path)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif path.lower().endswith(".docx"):
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif path.lower().endswith(".txt"):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""
    return ""

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def generate_jd_fallback(category):
    return f"Seeking a {category.replace('_', ' ').lower()} with relevant experience and skills."

# ==== Generate Labeled Pairs ====
log_time("Generating labeled pairs")
resume_texts, jd_texts, labels, soft_skill_scores = [], [], [], []
categories = sorted([d for d in os.listdir(RESUME_DIR) if os.path.isdir(os.path.join(RESUME_DIR, d))])

random.seed(42)
split_ratio = 0.3
jd_building_sets, pair_sets = {}, {}

for cat in categories:
    resumes = [f for f in os.listdir(os.path.join(RESUME_DIR, cat))
               if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if len(resumes) < 2:
        jd_building_sets[cat] = []
        pair_sets[cat] = []
        continue
    random.shuffle(resumes)
    split = int(len(resumes) * split_ratio)
    jd_building_sets[cat] = resumes[:split]
    pair_sets[cat] = resumes[split:]

# TF-IDF JD generation
category_docs = {}
for cat in categories:
    docs = []
    for f in jd_building_sets[cat]:
        txt = clean(extract_text(os.path.join(RESUME_DIR, cat, f)))
        if len(txt) >= 100:
            docs.append(txt)
    category_docs[cat] = " ".join(docs) if docs else ""

jd_map = {}
valid_cats = [c for c in categories if category_docs[c]]

if valid_cats:
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)
    tfidf_matrix = tfidf.fit_transform([category_docs[c] for c in valid_cats])
    feature_names = tfidf.get_feature_names_out()
    for i, cat in enumerate(valid_cats):
        weights = tfidf_matrix[i].toarray().flatten()
        top_terms = [feature_names[j] for j in np.argsort(weights)[-10:][::-1]]
        jd_map[cat] = f"Seeking a {cat.replace('_', ' ').lower()} skilled in {', '.join(top_terms)}."
else:
    print("⚠️ TF-IDF skipped. Using fallback.")

for cat in categories:
    if cat not in jd_map:
        jd_map[cat] = generate_jd_fallback(cat)

# Labeled pair creation
for cat in tqdm(categories, desc="📁 Categories"):
    for f in pair_sets[cat]:
        res_path = os.path.join(RESUME_DIR, cat, f)
        res_text = clean(extract_text(res_path))
        if len(res_text) < 100:
            continue
        jd_text = jd_map[cat]
        score = count_soft_skills_match(res_text, jd_text)

        # Positive
        resume_texts.append(res_text)
        jd_texts.append(jd_text)
        soft_skill_scores.append(score)
        labels.append(1)

        # Negatives
        neg_cats = [c for c in categories if c != cat]
        for neg_cat in random.sample(neg_cats, min(3, len(neg_cats))):
            jd_text_neg = jd_map[neg_cat]
            score_neg = count_soft_skills_match(res_text, jd_text_neg)
            resume_texts.append(res_text)
            jd_texts.append(jd_text_neg)
            soft_skill_scores.append(score_neg)
            labels.append(0)

print(f"✅ {len(labels)} pairs created ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

# ==== Encode with BERT ====
log_time("Encoding with BERT")
res_emb = bert.encode(resume_texts, batch_size=32, show_progress_bar=True)
jd_emb = bert.encode(jd_texts, batch_size=32, show_progress_bar=True)

# ==== Feature Engineering ====
log_time("Creating similarity features")
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

similarity = []
for i in tqdm(range(len(res_emb)), desc="🔗 Similarity"):
    sim = cosine_sim(res_emb[i], jd_emb[i])
    dot = np.dot(res_emb[i], jd_emb[i])
    similarity.append([sim, dot])

# ==== Combine Features ====
X = np.hstack([
    res_emb,
    jd_emb,
    np.array(similarity),
    np.array(soft_skill_scores).reshape(-1, 1)  # Add soft skill match score
])
y = np.array(labels)

# ==== Train/Test Split & Scaling ====
log_time("Train/Test split and scaling")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==== Balance with SMOTE ====
log_time("Applying SMOTE")
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ==== Train Models ====
log_time("Training models")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", max_depth=7, learning_rate=0.1)
}

best_model, best_score = None, 0
for name, model in models.items():
    print(f"\n🧠 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ ROC-AUC: {roc:.4f}")
    print(f"📋 Report:\n{classification_report(y_test, y_pred)}")
    if roc > best_score:
        best_score = roc
        best_model = model
        best_name = name

# ==== Save Model Artifacts ====
log_time(f"Saving best model: {best_name}")
dump(best_model, f"{MODEL_DIR}/resume_matcher_{best_name.replace(' ', '_')}.joblib")
dump(bert, f"{MODEL_DIR}/bert_encoder.joblib")
dump(scaler, f"{MODEL_DIR}/scaler.joblib")
print(f"✅ All model artifacts saved to {MODEL_DIR}")

log_time("✅ All tasks complete")
