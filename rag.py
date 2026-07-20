"""
RAG Core — chunking, embedding, FAISS, retrieval
"""

import os
import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pdfplumber

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50

VECTOR_STORE = "vector_store/index.faiss"
CHUNKS_STORE = "vector_store/chunks.pkl"

os.makedirs("vector_store", exist_ok=True)
os.makedirs("uploads", exist_ok=True)


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
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index(chunks):
    embeddings = EMBED_MODEL.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, VECTOR_STORE)
    with open(CHUNKS_STORE, "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def retrieve(query, top_k=4):
    if not os.path.exists(VECTOR_STORE):
        return []

    index = faiss.read_index(VECTOR_STORE)
    with open(CHUNKS_STORE, "rb") as f:
        chunks = pickle.load(f)

    q_embed = EMBED_MODEL.encode([query]).astype("float32")
    faiss.normalize_L2(q_embed)

    scores, indices = index.search(q_embed, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({"chunk": chunks[idx], "score": round(float(score), 3)})
    return results


def process_pdf(pdf_path):
    text   = extract_text(pdf_path)
    chunks = chunk_text(text)
    n      = build_index(chunks)
    return n
