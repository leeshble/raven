from typing import Union
from fastapi import FastAPI
from schemas import QueryRequest, QueryResponse, OllamaAskRequest, OllamaAskResponse
from schemas import FaissEmbeddingRequest, FaissEmbeddingResponse, FaissQueryRequest, FaissOllamaRequest
from vector_store import VectorStore
import uvicorn
from csv_embedding import create_embeddings, EmbeddingRequest, EmbeddingResponse
from ollama_service import OllamaService
from faiss_embedding import create_faiss_embeddings
from faiss_vector_store import FaissVectorStore
import os
import math

app = FastAPI()

# Initialize vector store
vector_store = VectorStore(host="localhost", port=6333, collection_name="qa_dataset")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    # Search for similar questions
    search_results = vector_store.search(request.query, limit=3)
    
    # Extract answers and sources from search results
    answers = []
    sources = []
    
    for result in search_results:
        answers.append(result.payload["esti_answer"])
        sources.append(result.payload["esti_rel_link"])
    
    # Combine answers if multiple results
    combined_answer = " ".join(answers)
    
    return {"answer": combined_answer, "sources": sources}

@app.post("/ask-ollama", response_model=OllamaAskResponse)
def ask_ollama(request: OllamaAskRequest):
    # Initialize Ollama service
    ollama_service = OllamaService(base_url=request.ollama_base_url)
    
    # Search for similar questions in vector store
    search_results = vector_store.search(request.query, limit=request.limit)
    
    # Extract answers and sources from search results
    contexts = []
    sources = []
    
    for result in search_results:
        context = f"Question: {result.payload.get('esti_question', '')}\nAnswer: {result.payload.get('esti_answer', '')}"
        contexts.append(context)
        # Ensure source is always a string, use empty string if None
        source = result.payload.get("esti_rel_link", "")
        sources.append(source if source is not None else "")
    
    # Generate answer from Ollama with retrieved contexts
    answer = ollama_service.generate_response(
        prompt=request.query,
        model=request.model,
        context=contexts,
        system_prompt=request.system_prompt
    )
    
    return {
        "answer": answer,
        "sources": sources,
        "raw_contexts": contexts
    }

@app.post("/create-embeddings", response_model=EmbeddingResponse)
def create_embeddings_endpoint(request: EmbeddingRequest):
    success, message, num_points = create_embeddings(
        request
    )
    
    return EmbeddingResponse(
        success=success,
        message=message,
        num_points=num_points
    )

@app.post("/search")
def simple_search(request: QueryRequest):
    """단순 검색: 쿼리와 가장 유사한 하나의 결과만 반환"""
    # 가장 유사한 결과 하나만 검색
    search_results = vector_store.search(request.query, limit=1)
    
    if not search_results:
        return {"result": None}
    
    # 결과가 있으면 첫 번째 결과 반환
    top_result = search_results[0]
    return {
        "question": top_result.payload["esti_question"],
        "answer": top_result.payload["esti_answer"],
        "source": top_result.payload["esti_rel_link"],
        "similarity": top_result.score
    }

@app.post("/create-faiss-embeddings", response_model=FaissEmbeddingResponse)
def create_faiss_embeddings_endpoint(request: FaissEmbeddingRequest):
    """Faiss를 사용하여 임베딩 생성"""
    success, message, num_points = create_faiss_embeddings(
        excel_path=request.excel_path,
        model_name=request.model_name,
        device=request.device,
        save_dir=request.save_dir
    )
    
    return FaissEmbeddingResponse(
        success=success,
        message=message,
        num_points=num_points
    )

@app.post("/faiss-search")
def faiss_search(request: FaissQueryRequest):
    """Faiss를 사용한 벡터 검색"""
    # Faiss 인덱스 파일 경로 확인
    index_path = os.path.join(request.index_dir, "faiss_index.bin")
    metadata_path = os.path.join(request.index_dir, "metadata.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return {"error": "Faiss 인덱스가 존재하지 않습니다. 먼저 임베딩을 생성해주세요."}
    
    try:
        # Faiss 벡터 저장소 초기화 및 인덱스 로드
        vector_store = FaissVectorStore(index_path=index_path, metadata_path=metadata_path)
        
        # 검색 수행
        search_results = vector_store.search(
            query_text=request.query,
            limit=request.limit,
            model_name=request.model_name,
            device=request.device
        )
        
        if not search_results:
            return {"results": []}
        
        # 결과 포맷팅
        formatted_results = []
        for result in search_results:
            # 유효한 JSON 값 확인
            similarity_score = result["score"]
            if isinstance(similarity_score, float) and (math.isnan(similarity_score) or math.isinf(similarity_score)):
                similarity_score = 0.0
                
            formatted_results.append({
                "id": result["id"],
                "question": result["payload"]["esti_question"],
                "answer": result["payload"]["esti_answer"],
                "source": result["payload"]["esti_rel_link"],
                "similarity": similarity_score
            })
        
        return {"results": formatted_results}
    except Exception as e:
        return {"error": f"검색 중 오류 발생: {str(e)}"}

@app.post("/faiss-ask-ollama", response_model=OllamaAskResponse)
def faiss_ask_ollama(request: FaissOllamaRequest):
    """Faiss 벡터 검색과 Ollama LLM을 조합한 질의응답 API"""
    try:
        # 1. Ollama 서비스 초기화
        ollama_service = OllamaService(base_url=request.ollama_base_url)
        
        # 2. Faiss 인덱스 파일 경로 확인
        index_dir = request.index_dir  # 사용자가 지정한 인덱스 디렉토리
        index_path = os.path.join(index_dir, "faiss_index.bin")
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return {"answer": f"Faiss 인덱스가 존재하지 않습니다. 먼저 임베딩을 생성해주세요.", "sources": [], "raw_contexts": []}
        
        # 3. Faiss 벡터 저장소 초기화 및 인덱스 로드
        vector_store = FaissVectorStore(index_path=index_path, metadata_path=metadata_path)
        
        # 4. Faiss 검색 수행
        search_results = vector_store.search(
            query_text=request.query,
            limit=request.limit,
            model_name="sentence-transformers/static-similarity-mrl-multilingual-v1",
            device="cpu"
        )
        
        # 5. 검색 결과에서 컨텍스트와 출처 추출
        contexts = []
        sources = []
        
        for result in search_results:
            # 질문과 답변을 컨텍스트로 조합
            question = result["payload"].get("esti_question", "")
            answer = result["payload"].get("esti_answer", "")
            context = f"질문: {question}\n답변: {answer}"
            contexts.append(context)
            
            # 출처 정보 저장
            source = result["payload"].get("esti_rel_link", "")
            # 출처가 None이면 빈 문자열로 대체
            sources.append(source if source is not None else "")
        
        # 컨텍스트가 없으면 알림
        if not contexts:
            return {"answer": "검색 결과가 없습니다.", "sources": [], "raw_contexts": []}
        
        # 6. Ollama LLM으로 최종 답변 생성
        answer = ollama_service.generate_response(
            prompt=request.query,
            model=request.model,
            context=contexts,
            system_prompt=request.system_prompt
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "raw_contexts": contexts
        }
    except Exception as e:
        return {"answer": f"오류가 발생했습니다: {str(e)}", "sources": [], "raw_contexts": []}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
