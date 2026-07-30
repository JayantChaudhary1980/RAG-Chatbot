# RAG-Powered Intelligent Document Assistant Chatbot (Capstone Project)

## Student Details

* **Name:** Jayant Chaudhary
* **Application No.:** IN26011386
* **University Registration No.:** 23BCE10085
* **University:** VIT Bhopal University
* **Internship:** Artificial Intelligence and Machine Learning Internship – Batch 2(B) (6:00 PM – 8:00 PM)

---

# Capstone Project

This project was developed as the **Capstone Project** for the Artificial Intelligence and Machine Learning Internship. It demonstrates the practical implementation of **Retrieval-Augmented Generation (RAG)** by combining semantic document retrieval with a Large Language Model to build an intelligent document question-answering system.

---

# Project Overview

The **RAG Chatbot** is an AI-powered document assistant capable of answering natural language questions from any uploaded PDF document. Instead of relying solely on a language model's internal knowledge, the system retrieves the most relevant information from the uploaded document using semantic vector search and generates accurate, context-aware responses based only on the retrieved content.

The chatbot can be used with a wide variety of documents, including academic handbooks, technical documentation, company manuals, research papers, policy documents, user guides, and other PDF-based knowledge sources.

---

# Objectives

* Build an intelligent document question-answering system.
* Implement Retrieval-Augmented Generation (RAG).
* Generate semantic embeddings for document retrieval.
* Store document vectors using FAISS.
* Retrieve context relevant to user queries.
* Generate accurate answers using Groq Llama.
* Provide source passages supporting every response.

---

# Features

* Upload any PDF document.
* Automatic text extraction from PDFs.
* Intelligent document chunking.
* Semantic embeddings using Sentence Transformers.
* Fast similarity search using FAISS.
* AI-powered responses using Groq Llama.
* Displays retrieved source passages.
* Clean and responsive web interface.
* Works with multiple document domains.

---

# Technologies Used

* Python
* Flask
* Groq API
* FAISS
* Sentence Transformers
* PDFPlumber
* NumPy
* HTML
* CSS
* JavaScript

---

# Project Workflow

1. Upload a PDF document.
2. Extract text from the document.
3. Split the text into manageable chunks.
4. Generate vector embeddings for each chunk.
5. Store embeddings in a FAISS vector database.
6. Accept a natural language question from the user.
7. Retrieve the most relevant document chunks using semantic similarity search.
8. Provide the retrieved context to the Groq LLM.
9. Generate a context-aware answer.
10. Display the answer along with the retrieved source passages.

---

# Project Structure

```text
RAG-Chatbot/
│
├── app.py
├── rag.py
├── requirements.txt
├── templates/
│   └── index.html
├── uploads/
├── vector_store/
├── .env.example
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd RAG-Chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your Groq API Key:

```text
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
python app.py
```

Open in your browser:

```text
http://localhost:5002
```

## Live Demo

The application is deployed on Render and can be accessed here:

**Live Application:** https://rag-chatbot-k2vi.onrender.com/

> **Note:** Since the application is hosted on Render's free tier, the initial request may take around 30–60 seconds while the server wakes up from inactivity.

---

# Supported Documents

The chatbot can answer questions from documents such as:

* Student Handbooks
* Academic Regulations
* Company Documentation
* Product Manuals
* Technical Documentation
* Research Papers
* User Guides
* Policy Documents
* Standard Operating Procedures (SOPs)
* Government Publications
* Reports and Manuals

---

# Sample Questions

* What are the attendance requirements?
* Explain the installation procedure.
* What are the eligibility criteria?
* Summarize Chapter 3.
* What are the key responsibilities mentioned in the document?
* What are the important deadlines?
* What safety precautions are specified?
* What are the main features of this product?

---

# Future Enhancements

* Multi-document knowledge base
* OCR support for scanned PDFs
* Conversation history
* User authentication
* Hybrid keyword + semantic search
* Document management dashboard
* Support for DOCX, TXT, and Markdown files
* Cloud deployment with scalable vector database

---

# Conclusion

This Capstone Project demonstrates how Retrieval-Augmented Generation (RAG) can be used to build an intelligent document question-answering system. By combining semantic embeddings, FAISS vector search, and the Groq Llama large language model, the chatbot retrieves relevant information from uploaded PDF documents and generates accurate, context-aware responses. The architecture is scalable, efficient, and adaptable to a wide range of document types, making it suitable for applications in education, enterprise knowledge management, technical documentation, research, and customer support.

---

# Author

**Jayant Chaudhary**

VIT Bhopal University

Artificial Intelligence and Machine Learning Internship – Batch 2(B)
