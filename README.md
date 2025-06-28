# AI-Based Resume Scanner for Job Applications

AI Resume Scanner is a Python-based web application that enables recruiters to post job listings and automatically match submitted resumes using an AI-powered scoring system. Applicants can browse and apply for jobs by uploading their resumes directly through the portal. The app uses BERT embeddings and a logistic regression model to evaluate resume-job relevance. It features separate dashboards for recruiters and applicants, with built-in resume analysis and secure authentication.

## 📦 Library Versions

This project uses Python 3.10 and the following package versions:

```plaintext
numpy==1.24.4
scikit-learn==1.3.2
sentence-transformers==2.2.2
transformers==4.29.2
huggingface-hub==0.14.1
tokenizers==0.13.3
spacy==3.6.1
PyPDF2==3.0.1
python-docx==1.1.0
streamlit==1.35.0
xgboost==2.0.3
imbalanced-learn==0.11.0
pandas==2.1.4
tqdm==4.66.4
pillow==10.3.0
matplotlib
nltk
```

## 🛠️ Running Locally

1. Clone the project:

```bash
git clone https://github.com/shravanramv/AI-Based-Resume-Scanner-Python.git
```

2. Create and activate virtual environment (Python 3.10):

```bash
python3.10 -m venv resume-env
source resume-env/bin/activate  # macOS/Linux

# OR

resume-env\Scripts\activate     # Windows
```

3. Upgrade Pip & Install All Dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Or install manually using:

```bash
pip install numpy==1.24.4 
pip install scikit-learn==1.3.2 
pip install sentence-transformers==2.2.2 
pip install transformers==4.29.2
pip install huggingface-hub==0.14.1 
pip install tokenizers==0.13.3 
pip install spacy==3.6.1 
pip install PyPDF2==3.0.1 
pip install python-docx==1.1.0 
pip install streamlit==1.35.0 
pip install xgboost==2.0.3 
pip install imbalanced-learn==0.11.0 pandas==2.1.4 
pip install tqdm==4.66.4
pip install pillow==10.3.0 
pip install matplotlib 
pip install nltk
```

4. Make sure to also download NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```

5. Run the app:

```bash
streamlit run app.py
```

## ✅ Features

- User registration/login (applicants & recruiters)
- Resume upload and job applications
- AI-based match scoring using BERT and logistic regression
- Section-based insights (skills, education, experience)
- Resume filtering based on match score
- Resume downloads and job posting management

## 🧠 Model Info

The ML model uses Sentence-BERT for text embeddings and logistic regression for classification. It was trained on labeled pairs of resumes and job descriptions using cosine similarity and dot product features. The best-performing model is saved in the ```models/``` folder.

## 🧑‍💻 Author
Built by **Shravan Ram Venkateswaran** for **Multimedia University**. 

**Final Year Project (FYP) T2510 2025**
