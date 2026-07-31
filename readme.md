# RAG-Powered Intelligent Document Assistant Chatbot (Capstone Project)

## Student Details

- **Name:** Jayant Chaudhary
- **Application No.:** IN26011386
- **University Registration No.:** 23BCE10085
- **University:** VIT Bhopal University
- **Internship:** Artificial Intelligence and Machine Learning Internship – Batch 2(B) (6:00 PM – 8:00 PM)
- **Submitted To:** Prof. Nishant Shrivastava

---

# Capstone Project

This project was developed as the **Capstone Project** for the Artificial Intelligence and Machine Learning Internship. It demonstrates the practical implementation of **Retrieval-Augmented Generation (RAG)** by combining document retrieval techniques with a Large Language Model to build an intelligent document question-answering system.

Unlike a conventional chatbot that relies solely on the knowledge of a language model, this application first retrieves the most relevant sections from an uploaded PDF document and then uses the retrieved information as context for answer generation. This approach improves response accuracy while ensuring that answers remain grounded in the uploaded document.

---

# Project Overview

The RAG Chatbot enables users to upload a PDF document and ask natural language questions about its contents. The system automatically extracts text from the document, divides it into manageable chunks, builds a searchable TF-IDF index, retrieves the most relevant passages using cosine similarity, and generates answers using Groq's Llama 3.1 model.

The application is suitable for querying:

- Student Handbooks
- Research Papers
- Technical Documentation
- User Manuals
- Company Policies
- Standard Operating Procedures
- Reports
- Government Documents
- Academic Notes

---

# Objectives

- Build an intelligent document question-answering system.
- Implement Retrieval-Augmented Generation (RAG).
- Extract and process information from PDF documents.
- Retrieve relevant document sections using information retrieval techniques.
- Generate context-aware answers using a Large Language Model.
- Display supporting document passages along with generated responses.

---

# Features

- Upload PDF documents up to 16 MB.
- Automatic text extraction using PDFPlumber.
- Intelligent document chunking.
- TF-IDF based document indexing.
- Fast cosine similarity search for relevant passages.
- AI-generated answers using Groq Llama 3.1.
- Displays retrieved source passages with similarity scores.
- Responsive web interface built with HTML, CSS and JavaScript.
- RESTful Flask backend.
- Optimized to run efficiently on cloud platforms such as Render.

---

# System Architecture

```
                 Upload PDF
                      │
                      ▼
             PDF Text Extraction
                (PDFPlumber)
                      │
                      ▼
             Document Chunking
                      │
                      ▼
            TF-IDF Vectorization
                      │
                      ▼
              Store Search Index
          (Pickled TF-IDF Objects)
                      │
                      ▼
              User Question
                      │
                      ▼
         Cosine Similarity Retrieval
                      │
                      ▼
          Top Relevant Document Chunks
                      │
                      ▼
       Groq Llama 3.1 Large Language Model
                      │
                      ▼
        Context-Aware Generated Response
                      │
                      ▼
     Answer + Supporting Source Passages
```

---

# Technology Stack

## Backend

- Python
- Flask
- Groq API

## Information Retrieval

- TF-IDF Vectorizer
- Cosine Similarity
- Scikit-learn

## Document Processing

- PDFPlumber

## Frontend

- HTML
- CSS
- JavaScript

## Deployment

- Gunicorn
- Render

---

# Project Workflow

1. User uploads a PDF document.
2. The document text is extracted using PDFPlumber.
3. The extracted text is divided into overlapping chunks.
4. A TF-IDF vector representation is created for all chunks.
5. The generated index is stored locally for efficient retrieval.
6. The user submits a question.
7. The system converts the question into a TF-IDF vector.
8. Cosine similarity identifies the most relevant document chunks.
9. Retrieved chunks are supplied as context to the Groq Llama 3.1 model.
10. The generated answer and supporting document excerpts are returned to the user.

---

# Project Structure

```
RAG-Chatbot/
│
├── app.py
├── rag_tfidf.py
├── requirements.txt
├── Procfile
├── gunicorn.conf.py
├── .python-version
├── .env.example
├── templates/
│   └── index.html
├── uploads/
├── vector_store/
├── docs/
│   └── student_handbook.pdf
└── README.md
```

---

# API Endpoints

## Home Page

```
GET /
```

Displays the chatbot interface.

---

## Upload PDF

```
POST /upload
```

Uploads and processes a PDF document.

### Response

```json
{
    "message": "PDF uploaded successfully. 45 chunks indexed."
}
```

---

## Ask a Question

```
POST /chat
```

### Request

```json
{
    "question": "What is the attendance policy?"
}
```

### Response

```json
{
    "answer": "...",
    "sources": [
        {
            "text": "...",
            "score": 0.82
        }
    ]
}
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create an environment file.

```
GROQ_API_KEY=your_groq_api_key
```

Run the application.

```bash
python app.py
```

The application will be available at:

```
http://localhost:5000
```

---

# Deployment

The project is deployed on Render using Gunicorn.

```
gunicorn app:app
```

Since the application is hosted on Render's free tier, the first request after inactivity may take a few seconds while the service starts.

---

# Sample Questions

- What are the attendance requirements?
- Summarize the uploaded document.
- What are the important rules mentioned?
- Explain Chapter 2.
- What are the eligibility criteria?
- List the responsibilities described in the document.
- What deadlines are mentioned?
- What precautions should be followed?

---

# Advantages of the Approach

The chatbot follows a Retrieval-Augmented Generation workflow, ensuring that responses are generated using information retrieved from the uploaded document instead of relying solely on the language model's internal knowledge.

The current implementation uses a TF-IDF based retrieval pipeline with cosine similarity, providing an efficient indexing and retrieval mechanism while keeping resource usage suitable for deployment on platforms with limited memory.

---

# Future Enhancements

- Multi-document support
- Conversation history
- User authentication
- OCR support for scanned PDFs
- Hybrid keyword and embedding-based retrieval
- Persistent document management
- Support for DOCX and TXT files
- Citation highlighting within PDF pages
- Streaming responses
- Multi-user document sessions

---

# Conclusion

This project demonstrates the practical implementation of a Retrieval-Augmented Generation (RAG) system for document question answering. By combining PDF text extraction, TF-IDF based retrieval, cosine similarity search, and Groq's Llama 3.1 language model, the application provides accurate and context-aware responses grounded in the uploaded document.

The modular architecture makes the system easy to understand, extend, and deploy, while its efficient retrieval pipeline enables reliable performance for document-based question answering across a wide range of real-world use cases.

---

# Author

**Jayant Chaudhary**

B.Tech Computer Science and Engineering

VIT Bhopal University

Artificial Intelligence and Machine Learning Internship – Batch 2(B)