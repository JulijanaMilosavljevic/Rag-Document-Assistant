# Rag‑Document‑Assistant
### Rag‑Document‑Assistant (TF‑IDF version)
This repository is a simple example of a Retrieval‑Augmented‑Generation (RAG) assistant that uses classical TF‑IDF‑based information retrieval. The goal is to show how traditional information‑retrieval techniques can be integrated with a generative model to answer user questions over a corpus of documents.

## Features

* **Document loading and indexing** – Text documents are loaded and split into smaller segments. A TF‑IDF vectorizer generates vector representations of the segments.
* **Search** – For each user query a TF‑IDF vector is computed and the most relevant segments are found via cosine similarity.
* **Generative model** – After retrieval the relevant segments are combined into a prompt that is passed to a generative model (e.g., T5) to formulate an answer.
* **Streamlit user interface** – The project uses Streamlit to provide a simple web UI through which users can upload documents and ask questions.
* **Tests and modular architecture** – The code is divided into layers (`app/` for the UI, `backend/` for logic) with unit tests in `tests/`.

## Project structure
Rag‑Document‑Assistant/ <br/>
├── app/ – Streamlit application (UI, document upload, question input) <br/>
├── backend/ – Functions for loading documents, vectorization and search <br/>
├── assets/ – Static assets (styles, icons) <br/>
├── deployment/ – Hosting configuration (Docker, Heroku, etc.) <br/>
├── tests/ – Unit tests for search logic <br/>
├── .streamlit/ – Streamlit environment configuration <br/>
├── requirements.txt – Python dependencies <br/>
└── keep_alive.py – Script for keeping the service alive on free hosts <br/>


## Running the project

1. **Clone the repository and install dependencies**  
   ```bash
   git clone https://github.com/JulijanaMilosavljevic/Rag-Document-Assistant.git
   cd Rag-Document-Assistant
   pip install -r requirements.txt
   ```
2. Launch the Streamlit app
   ```bash
   streamlit run app/main.py
   ```
3. **Upload documents and ask questions** – In the web interface upload text files and enter a question. The system will return a generated answer based on the most relevant segments.
## Live Demo 
[RAG Document Assistant](https://rag-document-assistant-gt.onrender.com/)
## Notes

This version uses a simple TF‑IDF approach that is suitable for small corpora and quick demonstrations. For larger corpora and more complex queries consider moving to a dense‑vector RAG implementation with FAISS (see the **Rag‑Document‑Assistant‑V2** project).
