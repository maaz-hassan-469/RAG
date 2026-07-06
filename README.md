# Local RAG (Retrieval-Augmented Generation) Pipeline

This project implements a local Retrieval-Augmented Generation (RAG) system using LangChain. It allows you to search and ask questions over private text documents by converting them into vectors, storing them locally, and using an LLM to generate answers based strictly on that data.

---

## Technical Stack

* **Framework:** LangChain
* **Vector Store:** ChromaDB (Local storage)
* **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (Runs 100% locally)
* **LLM Engine:** ollama llama3
* **Environment Configuration:** Python Dotenv

---

## Project Workflow

### 1. Data Ingestion (`ingestion_pipeline.py`)
This script processes raw documents and builds the local vector database:

* **Document Loading:** Uses `DirectoryLoader` and `TextLoader` to read all `.txt` files inside the `docs/` folder using `utf-8` encoding.
* **Text Chunking:** Splitting raw text files into smaller segments using `CharacterTextSplitter` with a `chunk_size` of 400 characters and a `chunk_overlap` of 50 characters, separating text by newlines (`\n`).
* **Vector Storage:** Passes text chunks to the HuggingFace embedding model, converting text into 384-dimensional vectors. These vectors are saved permanently on disk inside the `db/chroma_db/` folder using Cosine Similarity for distance calculation.

### 2. Retrieval and Generation (`retrieval_pipeline.py`)
This script handles the search query and answer generation:

* **Database Connection:** Connects to the existing `db/chroma_db/` folder and loads the same HuggingFace model to translate user queries into vectors.
* **Semantic Retrieval:** Uses `.as_retriever()` with `search_kwargs={"k": 5}` to find the top 5 text chunks closest to the user's query meaning.
* **Context Assembly:** Extracts raw text content from the retrieved document objects using a list comprehension and joins them with newlines. 
* **LLM Grounding:** Embeds the text chunks into a strict prompt template instructing the LLM to answer using *only* the provided context. If the context does not contain the answer, the LLM is forced to say *"I don't have enough information"*, preventing AI hallucinations.

---

## Installation and Execution Guide

### 1. Installation
Run the following commands inside your virtual environment (`venv`):
```bash
pip install python-dotenv langchain langchain-openai langchain-huggingface
pip uninstall langchain-chroma chromadb -y
pip install chromadb langchain-chroma --prefer-binary
pip install sentence-transformers --no-cache-dir --default-timeout=1000
