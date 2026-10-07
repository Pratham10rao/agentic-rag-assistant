# Agentic RAG Assistant

A lightweight agentic Retrieval-Augmented Generation (RAG) assistant that answers questions from a local knowledge base.

## What it does

The workflow:
1. Loads documents from `data/`.
2. Splits documents into overlapping chunks.
3. Creates embeddings and stores them in a local Chroma vector database.
4. Uses an agentic router to decide whether retrieval is needed.
5. Retrieves relevant chunks when required.
6. Generates a grounded answer using an LLM.
7. Returns the answer together with source filenames.

The project uses **LangGraph**, **ChromaDB**, **OpenAI embeddings/LLM**, and Python.

## Architecture

```text
User Question
      |
      v
+-------------+
| Agent Router|
+-------------+
   |       |
  no      yes
   |       |
   v       v
 Answer   Retriever
            |
            v
        Relevant Chunks
            |
            v
       Answer Generator
            |
            v
      Answer + Sources
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

Add `.txt` or `.md` documents to `data/`.

Build the vector database:

```bash
python ingest.py
```

Run the assistant:

```bash
python main.py
```

Example:

```text
You: What is the main purpose of this document?

Assistant: ...
Sources: example.txt
```

## Hardest problem

The hardest part was retrieval quality. Early versions sometimes returned related but incomplete chunks, which caused weak or partially grounded answers. I improved this by using overlapping chunks, semantic embeddings, top-k retrieval, and an explicit instruction to ground the final response in retrieved context.

## Project structure

```text
rag-agent/
├── data/
├── chroma_db/
├── .env.example
├── .gitignore
├── ingest.py
├── main.py
├── requirements.txt
└── README.md
```

## Note

Do not commit API keys or `.env` files.
