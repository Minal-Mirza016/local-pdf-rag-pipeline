# Local PDF RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions about PDF documents using semantic search and a locally running large language model.

Built with Python, LangChain, Hugging Face embeddings, ChromaDB, and Llama 3.2 via Ollama.

## Overview

This project demonstrates how to build a basic RAG system that retrieves relevant information from a PDF and uses that information to generate a contextual answer.

Instead of asking an LLM to answer from its general knowledge alone, the pipeline supplies retrieved document passages as context.

## Features

- **PDF loading:** Extracts page-level content using `PyPDFLoader`.
- **Text chunking:** Splits content into overlapping chunks using `RecursiveCharacterTextSplitter`.
- **Text embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` to create vector representations.
- **Vector database:** Stores document embeddings in ChromaDB.
- **Semantic retrieval:** Retrieves the three closest matching chunks for each question.
- **Local LLM:** Generates answers using Llama 3.2 3B through Ollama.
- **Page metadata:** Preserves source metadata and page numbers for retrieved passages.
- **Context-grounded prompting:** Instructs the model to answer from the retrieved context and acknowledge when the answer is not found.

## How It Works

The pipeline consists of three main stages.

### 1. Indexing

The PDF is loaded and split into smaller, overlapping text chunks. Each chunk is converted into an embedding and stored in a Chroma vector database.

### 2. Retrieval

When a user enters a question, the system searches the vector database and retrieves the three closest matching chunks.

### 3. Generation

The retrieved chunks and the question are inserted into a prompt. Llama 3.2 3B then generates an answer based on the supplied context.

## Architecture

```text
PDF Document
     |
     v
Load PDF Pages
     |
     v
Split into Text Chunks
     |
     v
Generate Embeddings
     |
     v
Chroma Vector Database
     |
User Question
     |
     v
Semantic Retrieval (Top 3)
     |
     v
Retrieved Context + Question
     |
     v
Llama 3.2 via Ollama
     |
     v
Generated Answer
```

## Tech Stack

- Python
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2 3B
- PyPDF

## Project Structure

```text
local-pdf-rag-pipeline/
├── doc/                 # Local PDF files (not committed)
├── chroma_db/           # Generated vector database (not committed)
├── main.py              # RAG pipeline
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- Git
- Ollama installed locally
- Enough available memory and storage to run the embedding model and LLM

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/local-pdf-rag-pipeline.git
cd local-pdf-rag-pipeline
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download the local LLM

Install Ollama from [ollama.com](https://ollama.com/) if necessary, then run:

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the pipeline.

### 5. Add a PDF

Create a `doc` folder in the project root and place a PDF you are authorized to use inside it.

The current `main.py` expects this filename:

```text
doc/AI-Agents-Your-New-Smart-Colleagues.pdf
```

If you use another PDF, update `PDF_PATH` in `main.py`.

The source PDF is not included in this repository. Users must supply their own document.

### 6. Run the pipeline

```powershell
python main.py
```

Enter a question when prompted. The terminal displays the retrieved passages, page numbers, distance scores, and final generated answer.

## Limitations

- Only the top three retrieved chunks are passed to the model.
- Retrieval quality depends on chunk size, embeddings, and the document's content.
- The generated response may be incomplete or incorrect, even when the retrieved context appears relevant.
- A distance score is not a percentage of answer correctness.
- The current script creates the vector database when run; database reuse and document-update handling can be improved.

## Future Improvements

- Reuse the persisted Chroma database rather than rebuilding it each run.
- Add source citations to the generated answer.
- Evaluate retrieval with a set of known-answer questions.
- Experiment with chunk sizes, overlap, and embedding models.
- Add a FastAPI endpoint or Streamlit chat interface.
- Evaluate answer faithfulness and retrieval relevance separately.

## Learning Outcomes

This project is a hands-on exploration of document ingestion, text chunking, vector embeddings, semantic retrieval, vector databases, prompt engineering, and local LLM integration.

---

**Purpose:** Learning and experimentation with Retrieval-Augmented Generation (RAG).
