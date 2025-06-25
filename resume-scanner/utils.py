import PyPDF2
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def compute_similarity(jd_text, resume_text):
    texts = [jd_text, resume_text]
    cv = CountVectorizer().fit_transform(texts)
    similarity = cosine_similarity(cv[0:1], cv[1:2])
    return similarity[0][0]  # A score between 0 and 1
