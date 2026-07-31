import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from groq import Groq
from rag_tfidf import process_pdf, retrieve

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ask_groq(question, context):

    context_text = "\n\n".join([item["chunk"] for item in context])

    prompt = f"""
You are a helpful AI assistant.

Answer the question only from the context below.

If the answer is not present, reply:
"I couldn't find the answer in the uploaded document."

Context:

{context_text}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return jsonify({
            "error": "No file selected."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Please choose a PDF file."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only PDF files are allowed."
        }), 400

    filename = secure_filename(file.filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    try:
        total_chunks = process_pdf(filepath)

        return jsonify({
            "message": f"PDF uploaded successfully. {total_chunks} chunks indexed."
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("question", "").strip()

    if question == "":
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if not os.path.exists("vector_store/chunks.pkl"):
        return jsonify({
            "error": "Please upload a PDF first."
        }), 400

    context = retrieve(question)

    if len(context) == 0:
        return jsonify({
            "answer": "No relevant information found.",
            "sources": []
        })

    answer = ask_groq(question, context)

    sources = []

    for item in context:
        sources.append({
            "text": item["chunk"][:200] + "...",
            "score": item["score"]
        })

    return jsonify({
        "answer": answer,
        "sources": sources
    })


if __name__ == "__main__":
    app.run(debug=True)