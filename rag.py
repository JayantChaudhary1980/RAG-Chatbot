import os
import faiss
import pickle
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

VECTOR_STORE = "vector_store/index.faiss"
CHUNKS_STORE = "vector_store/chunks.pkl"

os.makedirs("vector_store", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

model = None


def get_model():
    global model

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


def extract_text(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def chunk_text(text):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def build_index(chunks):
    embeddings = get_model().encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    faiss.write_index(index, VECTOR_STORE)

    with open(CHUNKS_STORE, "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def process_pdf(pdf_path):
    text = extract_text(pdf_path)

    if not text.strip():
        raise ValueError("No readable text found in this PDF.")

    chunks = chunk_text(text)

    if len(chunks) > 300:
        chunks = chunks[:300]

    total_chunks = build_index(chunks)

    return total_chunks


def retrieve(question, top_k=4):
    if not os.path.exists(VECTOR_STORE):
        return []

    index = faiss.read_index(VECTOR_STORE)

    with open(CHUNKS_STORE, "rb") as f:
        chunks = pickle.load(f)

    query_embedding = get_model().encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "score": round(float(score), 3)
            })

    return results