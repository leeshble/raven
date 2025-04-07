# Raven

**Raven** is an intelligent Retrieval-Augmented Generation (RAG) system that enhances LLM performance by grounding answers in domain-specific or private knowledge. Inspired by the raven's symbolic wisdom and precision, this system delivers accurate, context-aware responses to complex queries.

## Features

- **Contextual Search**: Hybrid retrieval combining dense & sparse embeddings (e.g., FAISS + BM25)
- **Custom Knowledge Base**: Easily ingest files, URLs, databases, or APIs
- **LLM Integration**: Compatible with OpenAI, Anthropic, or open-source models like LLaMA, Mistral
- **Safe & Filtered Responses**: Configurable output filtering and prompt-guardrails
- **Modular Design**: Plug-and-play architecture for retrievers, chunkers, and vector stores
- **Ollama Integration**: Use Ollama models with vector-based retrieval for accurate RAG responses
- **Multilingual Support**: Enhanced support for Korean language queries for financial information

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally (for LLM integration)
- [Qdrant](https://qdrant.tech/) for vector storage (optional)

### Step 1: Clone the repository

```bash
git clone https://github.com/leeshble/raven.git
cd raven
```

### Step 2: Create a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install the following packages:

```bash
pip install fastapi uvicorn pydantic qdrant-client sentence-transformers faiss-cpu pandas ollama
```

### Step 4: Run the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API server will now be running at http://localhost:8000.

## Usage

### Creating Embeddings

Before using the RAG capabilities, you need to create embeddings for your data:

```bash
# For Qdrant vector store
curl -X POST "http://localhost:8000/create-embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/data.csv",
    "collection_name": "qa_dataset",
    "vector_dimension": 768,
    "distance": "Cosine"
  }'

# For FAISS vector store
curl -X POST "http://localhost:8000/create-faiss-embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/data.csv",
    "model_name": "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "device": "cpu",
    "save_dir": "faiss_data"
  }'
```

### Simple Vector Search

To perform a simple search without LLM generation:

```bash
# For Qdrant search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return rate of SOXL?"
  }'

# For FAISS search
curl -X POST "http://localhost:8000/faiss-search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return rate of SOXL?",
    "limit": 3,
    "index_dir": "faiss_data",
    "model_name": "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "device": "cpu"
  }'
```

### Ollama Integration

The system supports Ollama for generating responses based on vector-based retrieval:

```bash
# Example request to the Ollama Ask API
curl -X POST "http://localhost:8000/ask-ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "model": "llama3",
    "limit": 3,
    "system_prompt": "Optional system prompt",
    "ollama_base_url": "http://localhost:11434"
  }'
```

For Korean language financial queries, you can use:

```bash
curl -X POST "http://localhost:8000/ask-ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SOXL의 수익률을 얼마인가요?",
    "model": "exaone3.5:2.4b",
    "limit": 3
  }'
```

You can also use FAISS with Ollama for improved retrieval:

```bash
curl -X POST "http://localhost:8000/faiss-ask-ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SOXL의 수익률을 얼마인가요?",
    "model": "exaone3.5:2.4b",
    "limit": 3,
    "index_dir": "faiss_data",
    "system_prompt": "당신은 금융 정보를 제공하는 도우미입니다. 주어진 맥락에 있는 정보를 기반으로만 답변하세요.",
    "ollama_base_url": "http://localhost:11434"
  }'
```

### How It Works

This endpoint will:
1. Search the vector database for relevant information matching your query
2. Retrieve similar content from the database
3. Send the query along with the retrieved context to Ollama
4. Return Ollama's response along with the sources of information

### Parameters

- `query`: Your question (supports Korean and English)
- `model`: The Ollama model to use (default: "exaone3.5:2.4b")
- `limit`: Number of similar documents to retrieve (default: 3)
- `system_prompt`: Optional system prompt to guide the model behavior
- `ollama_base_url`: URL of your Ollama server (default: "http://localhost:11434")
- `index_dir`: Directory where FAISS index is stored (for FAISS endpoints)

### Response Format

```json
{
  "answer": "The model's answer to your question",
  "sources": ["Source 1", "Source 2", "Source 3"],
  "raw_contexts": ["Context 1", "Context 2", "Context 3"]
}
```

## Data Format

The CSV files for embeddings should have the following columns:
- `esti_question`: The question
- `esti_answer`: The answer
- `esti_rel_link`: The source link or reference

Example:
```
esti_question,esti_answer,esti_rel_link
"What is SOXL?","SOXL is a leveraged ETF that aims to provide 3x the daily return of the PHLX Semiconductor Sector Index.","https://www.example.com/soxl"
```

## Troubleshooting

- **FAISS Index Not Found**: Make sure you have created embeddings with the `/create-faiss-embeddings` endpoint before querying.
- **Ollama Connection Issues**: Ensure Ollama is running locally with `ollama serve` or update the `ollama_base_url` parameter.
- **Out of Memory Errors**: Try using a smaller model or reducing the batch size when creating embeddings.
