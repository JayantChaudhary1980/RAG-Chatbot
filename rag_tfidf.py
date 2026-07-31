"""
RAG Core — chunking, TF-IDF retrieval, no heavy model
Works within Render free tier 512MB RAM limit
"""

import os
import gc
import pickle

import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

CHUNKS_STORE     = "vector_store/chunks.pkl"
VECTORIZER_STORE = "vector_store/vectorizer.pkl"
MATRIX_STORE     = "vector_store/matrix.pkl"

os.makedirs("vector_store", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

_chunks     = None
_vectorizer = None
_matrix     = None


def clear_cache():
    global _chunks, _vectorizer, _matrix
    _chunks = _vectorizer = _matrix = None
    gc.collect()


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def chunk_text(text):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index(chunks):
    global _chunks, _vectorizer, _matrix

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix     = vectorizer.fit_transform(chunks)

    with open(CHUNKS_STORE, "wb") as f:
        pickle.dump(chunks, f)
    with open(VECTORIZER_STORE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MATRIX_STORE, "wb") as f:
        pickle.dump(matrix, f)

    _chunks     = chunks
    _vectorizer = vectorizer
    _matrix     = matrix

    return len(chunks)


def load_index():
    global _chunks, _vectorizer, _matrix
    if _chunks is None:
        with open(CHUNKS_STORE, "rb") as f:
            _chunks = pickle.load(f)
    if _vectorizer is None:
        with open(VECTORIZER_STORE, "rb") as f:
            _vectorizer = pickle.load(f)
    if _matrix is None:
        with open(MATRIX_STORE, "rb") as f:
            _matrix = pickle.load(f)


def retrieve(query, top_k=4):
    if not os.path.exists(CHUNKS_STORE):
        return []

    load_index()

    q_vec   = _vectorizer.transform([query])
    scores  = cosine_similarity(q_vec, _matrix).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]

    return [
        {"chunk": _chunks[i], "score": round(float(scores[i]), 3)}
        for i in top_idx if scores[i] > 0
    ]


def process_pdf(pdf_path):
    clear_cache()
    text   = extract_text(pdf_path)
    chunks = chunk_text(text)
    return build_index(chunks)