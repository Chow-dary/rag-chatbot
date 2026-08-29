# RAG Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** application built with Python, Sentence Transformers, ChromaDB, Qwen, and Streamlit.

The application reads a PDF document, splits the text into chunks, converts those chunks into embeddings, stores them in a vector database, retrieves relevant information based on a user's question, and uses an instruction-tuned language model to generate an answer grounded in the retrieved document context.

## Architecture

```text
PDF Document
     ↓
Text Extraction
     ↓
Chunking
     ↓
Embeddings
     ↓
ChromaDB Vector Store
     ↓
User Question
     ↓
Query Embedding
     ↓
Retriever / Top-K Search
     ↓
Relevant Context
     ↓
Qwen LLM
     ↓
Grounded Answer
     ↓
Streamlit Web UI
```

## Features

* PDF text extraction
* Recursive text chunking with overlap
* Semantic embeddings
* Persistent vector storage with ChromaDB
* Semantic similarity search
* Top-K document retrieval
* Source metadata tracking
* LLM-ready context construction
* Grounded answer generation
* Hallucination fallback behavior
* Streamlit web interface
* Source/chunk display with generated answers

## Technology Stack

**Language**

* Python

**Document Processing**

* PyPDF

**Chunking**

* LangChain RecursiveCharacterTextSplitter

**Embedding Model**

* Sentence Transformers
* `all-MiniLM-L6-v2`

**Vector Database**

* ChromaDB

**Generation Model**

* `Qwen/Qwen2.5-0.5B-Instruct`

**LLM Framework**

* Hugging Face Transformers
* PyTorch

**Frontend**

* Streamlit

## Project Structure

```text
rag-chatbot/
│
├── app.py
│
├── README.md
├── .gitignore
│
├── data/
│   └── sample.pdf
│
├── chroma_db/
│
└── src/
    ├── ingestion.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    └── generator.py
```

## How the RAG Pipeline Works

### 1. PDF Ingestion

The application loads the PDF document using PyPDF and extracts the text.

```text
PDF
 ↓
PyPDF
 ↓
Extracted Text
```

### 2. Text Chunking

The extracted document is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

Example configuration:

```python
chunk_size = 200
chunk_overlap = 50
```

Overlap helps preserve context when important information falls near a chunk boundary.

### 3. Embeddings

Each document chunk is converted into a dense vector using:

```text
all-MiniLM-L6-v2
```

The model produces a **384-dimensional embedding** for each chunk.

Embeddings allow the system to compare text based on semantic similarity rather than relying only on exact keyword matches.

### 4. ChromaDB Vector Store

The document chunks, embeddings, IDs, and metadata are stored in a persistent ChromaDB collection.

Example metadata:

```python
{
    "source": "sample.pdf",
    "chunk_number": 3
}
```

Persistent storage allows the vector database to survive application restarts.

### 5. Retriever

When the user asks a question, the application:

1. Converts the question into an embedding.
2. Searches ChromaDB.
3. Retrieves the Top-K most relevant chunks.
4. Returns document text, metadata, and distance values.
5. Combines the retrieved chunks into an LLM-ready context block.

```text
Question
   ↓
Query Embedding
   ↓
ChromaDB
   ↓
Top-K Relevant Chunks
   ↓
Context
```

### 6. Grounded LLM Generation

The retrieved context and user question are passed to:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model is instructed to answer using only the provided document context.

If the answer cannot be found in the retrieved information, the application returns:

```text
I could not find the answer in the provided document.
```

This helps reduce unsupported answers and hallucinations.

### 7. Streamlit Interface

Streamlit provides a simple browser-based interface where users can enter questions and view:

* Generated answers
* Source document
* Retrieved chunk numbers

## Example

### Question

```text
How many vacation days do employees receive?
```

### Retrieved Information

```text
BrightTech employees receive 20 paid vacation days each year.
```

### Answer

```text
20 vacation days.
```

### Sources

```text
Source: sample.pdf | Chunk: 3
Source: sample.pdf | Chunk: 4
```

## Hallucination Test

The application was also tested with a question whose answer was not available in the document.

### Question

```text
What is the CEO's name?
```

### Result

```text
I could not find the answer in the provided document.
```

This demonstrates the basic grounding behavior of the RAG pipeline.

## Installation

Clone the repository:

```bash
git clone https://github.com/Chow-dary/rag-chatbot.git
cd rag-chatbot
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install pypdf
pip install langchain-text-splitters
pip install sentence-transformers
pip install chromadb
pip install transformers accelerate
pip install streamlit
```

## Build the Vector Database

Before starting the chatbot for the first time, create/populate the vector database:

```bash
python src/vector_store.py
```

This processes the PDF, generates embeddings, and stores the chunks in ChromaDB.

## Test the Retriever

```bash
python src/retriever.py
```

The retriever should return relevant chunks and source metadata.

## Test LLM Generation

```bash
python src/generator.py
```

This tests the complete retrieval and generation pipeline from the command line.

## Run the Web Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the URL displayed by Streamlit in your browser.

The application normally runs on port:

```text
8501
```

## Development Journey

The project was developed incrementally:

```text
Build 1 — PDF Ingestion
Build 2 — Text Chunking
Build 3 — Embeddings & Semantic Similarity
Build 4 — ChromaDB Vector Store
Build 5 — Retriever & Context Construction
Build 6 — Grounded LLM Generation
Build 7 — Streamlit Web Interface
```

This incremental approach made it possible to validate every component independently before combining them into the complete RAG pipeline.

## Retrieval Design

The retriever uses a reusable function:

```python
retrieve(question, top_k=2)
```

It returns:

```python
{
    "question": question,
    "context": context,
    "documents": documents,
    "metadatas": metadatas,
    "distances": distances
}
```

This keeps retrieval logic separate from generation and UI logic.

## Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Retrieval-Augmented Generation
* Document ingestion
* Text preprocessing
* Chunking strategies
* Chunk overlap
* Dense embeddings
* Semantic similarity
* Vector databases
* Persistent vector storage
* Top-K retrieval
* Metadata management
* Context construction
* Prompt grounding
* LLM generation
* Basic hallucination control
* Separation of retrieval and generation
* Streamlit application development

## Potential Improvements

Future versions could add:

* PDF upload through the UI
* Multiple-document support
* Chat-style message interface
* Conversation history
* Page-level citations
* Retrieval score thresholds
* Metadata filtering
* Hybrid search
* Reranking
* Retrieval evaluation
* Context deduplication
* Improved embedding models
* Larger instruction-tuned LLMs
* FastAPI backend
* Docker containerization
* Automated testing
* Cloud deployment
* Authentication and access control

## Important Note

This project is a learning and prototype implementation. Retrieval quality, generation quality, security, evaluation, observability, and deployment architecture would require additional work before using the application as a production system.

