"""
RAG Chatbot — PDF upload + FAISS + Groq LLM
"""

import os
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from groq import Groq
from rag import process_pdf, retrieve

app = Flask(__name__)
app.secret_key  = "rag_chatbot_secret_2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

UPLOAD_FOLDER   = "uploads"
ALLOWED_EXT     = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def ask_groq(question, context_chunks):
    context = "\n\n---\n\n".join([c["chunk"] for c in context_chunks])
    prompt  = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    n_chunks = process_pdf(filepath)
    return jsonify({"message": f"PDF processed successfully. {n_chunks} chunks indexed.", "filename": filename})


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question"}), 400

    if not os.path.exists("vector_store/index.faiss"):
        return jsonify({"error": "Please upload a PDF document first"}), 400

    chunks = retrieve(question, top_k=4)
    if not chunks:
        return jsonify({"answer": "No relevant content found in the document.", "sources": []})

    answer  = ask_groq(question, chunks)
    sources = [{"text": c["chunk"][:200] + "...", "score": c["score"]} for c in chunks]

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
