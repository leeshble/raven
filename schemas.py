from pydantic import BaseModel, validator
from typing import Literal

# Define allowed models
ALLOWED_MODELS = ["deepseek-r1:14b", "exaone3.5:2.4b", "exaone3.5:7.8b"]

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class EmbeddingRequest(BaseModel):
    csv_path: str
    model_name: str = "sentence-transformers/static-similarity-mrl-multilingual-v1"
    device: str = "cpu"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "qa_dataset"
    vector_size: int = 768

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    num_points: int = 0

class FaissQueryRequest(BaseModel):
    query: str
    limit: int = 3
    model_name: str = "sentence-transformers/static-similarity-mrl-multilingual-v1"
    device: str = "cpu"
    index_dir: str = "faiss_data"

class FaissEmbeddingRequest(BaseModel):
    excel_path: str
    model_name: str = "sentence-transformers/static-similarity-mrl-multilingual-v1"
    device: str = "cpu"
    save_dir: str = "faiss_data"

class FaissEmbeddingResponse(BaseModel):
    success: bool
    message: str
    num_points: int = 0

class OllamaAskRequest(BaseModel):
    query: str
    model: str = "exaone3.5:2.4b"
    limit: int = 3
    system_prompt: str = "당신은 M-ITSM HelpChat입니다. 사용자의 질문에 정확한 정보를 제공해야 합니다. 문제 상황에 답변할 때는 제공된 컨텍스트에 기반하여 사실적으로 답변하세요. 확실하지 않은 정보는 추측하지 말고, 모르는 경우 솔직하게 모른다고 말하세요."
    ollama_base_url: str = "http://localhost:11434"

    @validator('model')
    def validate_model(cls, v):
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Model must be one of {ALLOWED_MODELS}")
        return v

class OllamaAskResponse(BaseModel):
    answer: str
    sources: list[str]
    raw_contexts: list[str] = []

class FaissOllamaRequest(BaseModel):
    query: str
    model: str = "exaone3.5:2.4b"
    limit: int = 3
    index_dir: str = "faiss_data"
    system_prompt: str = "당신은 M-ITSM HelpChat입니다. 사용자의 질문에 정확한 정보를 제공해야 합니다. 문제 상황에 답변할 때는 제공된 컨텍스트에 기반하여 사실적으로 답변하세요. 확실하지 않은 정보는 추측하지 말고, 모르는 경우 솔직하게 모른다고 말하세요."
    ollama_base_url: str = "http://localhost:11434"

    @validator('model')
    def validate_model(cls, v):
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Model must be one of {ALLOWED_MODELS}")
        return v