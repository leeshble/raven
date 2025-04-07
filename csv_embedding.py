from vector_store import VectorStore
from schemas import EmbeddingRequest, EmbeddingResponse

def create_embeddings(request: EmbeddingRequest):
    """Create embeddings from CSV and store in vector database"""
    try:
        # Initialize vector store
        vector_store = VectorStore(host=request.host, port=request.port, collection_name=request.collection_name)
        
        # Create collection with specified vector size
        vector_store.create_collection(vector_size=request.vector_size)
        
        # Load embeddings from CSV
        num_points = vector_store.load_embeddings_from_csv(
            csv_path=request.csv_path,
            model_name=request.model_name,
            device=request.device
        )
        
        return True, f"총 {num_points}개 데이터 저장 완료", num_points
    except Exception as e:
        return False, f"임베딩 생성 중 오류 발생: {str(e)}", 0
