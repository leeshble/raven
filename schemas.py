from pydantic import BaseModel

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
    csv_path: str
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
    system_prompt: str = "당신은 한국어와 영어에 능숙한 금융 전문가 AI 비서입니다. 사용자의 금융 관련 질문에 정확하고 유용한 정보를 제공해야 합니다. 주식, ETF, 펀드 등에 관한 질문에 답변할 때는 제공된 컨텍스트에 기반하여 사실적으로 답변하세요. 확실하지 않은 정보는 추측하지 말고, 모르는 경우 솔직하게 모른다고 말하세요."
    ollama_base_url: str = "http://localhost:11434"

class OllamaAskResponse(BaseModel):
    answer: str
    sources: list[str]
    raw_contexts: list[str] = []

class FaissOllamaRequest(BaseModel):
    query: str
    model: str = "exaone3.5:2.4b"
    limit: int = 3
    index_dir: str = "faiss_data"
    system_prompt: str = "당신은 한국어와 영어에 능숙한 금융 전문가 AI 비서입니다. 사용자의 금융 관련 질문에 정확하고 유용한 정보를 제공해야 합니다. 주식, ETF, 펀드 등에 관한 질문에 답변할 때는 제공된 컨텍스트에 기반하여 사실적으로 답변하세요. 확실하지 않은 정보는 추측하지 말고, 모르는 경우 솔직하게 모른다고 말하세요."
    ollama_base_url: str = "http://localhost:11434"