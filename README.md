# Raven

**Raven** is an intelligent Retrieval-Augmented Generation (RAG) system that enhances LLM performance by grounding answers in domain-specific or private knowledge. Inspired by the raven's symbolic wisdom and precision, this system delivers accurate, context-aware responses to complex queries.

## Features

-  **Contextual Search**: Hybrid retrieval combining dense & sparse embeddings (e.g., FAISS + BM25)
-  **Custom Knowledge Base**: Easily ingest files, URLs, databases, or APIs
-  **LLM Integration**: Compatible with OpenAI, Anthropic, or open-source models like LLaMA, Mistral
-  **Safe & Filtered Responses**: Configurable output filtering and prompt-guardrails
-  **Modular Design**: Plug-and-play architecture for retrievers, chunkers, and vector stores
-  **Ollama Integration**: Use Ollama models with vector-based retrieval for accurate RAG responses
-  **Multilingual Support**: Enhanced support for Korean language queries for financial information

## Usage

### Ollama Integration

The system now supports Ollama for generating responses based on vector-based retrieval:

```bash
# Example request to the Ollama Ask API
curl -X POST "http://localhost:8000/ask-ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "model": "llama3",
    "limit": 3,
    "system_prompt": "Optional system prompt"
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

### Response Format

```json
{
  "answer": "The model's answer to your question",
  "sources": ["Source 1", "Source 2", "Source 3"],
  "raw_contexts": ["Context 1", "Context 2", "Context 3"]
}
```
