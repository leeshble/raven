from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import pandas as pd

class VectorStore:
    def __init__(self, host="localhost", port=6333, collection_name="qa_dataset"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
    def create_collection(self, vector_size=768):
        """Create or recreate the collection with specified vector size"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
    def load_embeddings_from_csv(self, csv_path, model_name="sentence-transformers/static-similarity-mrl-multilingual-v1", device="cpu"):
        """Load data from CSV and create embeddings"""
        # Load CSV data
        try:
            # 먼저 cp949로 시도
            df = pd.read_csv(csv_path, names=["esti_question", "esti_answer", "esti_rel_link"], encoding='cp949')
        except UnicodeDecodeError:
            try:
                # cp949 실패시 euc-kr 시도
                df = pd.read_csv(csv_path, names=["esti_question", "esti_answer", "esti_rel_link"], encoding='euc-kr')
            except UnicodeDecodeError:
                try:
                    # euc-kr 실패시 utf-8-sig 시도 (BOM이 있는 UTF-8)
                    df = pd.read_csv(csv_path, names=["esti_question", "esti_answer", "esti_rel_link"], encoding='utf-8-sig')
                except UnicodeDecodeError:
                    # 마지막으로 기본 utf-8 시도
                    df = pd.read_csv(csv_path, names=["esti_question", "esti_answer", "esti_rel_link"], encoding='utf-8')
        
        # Initialize embedding model
        model = SentenceTransformer(model_name, device=device)
        
        # Create points for Qdrant
        points = []
        for idx, row in df.iterrows():
            embedding = model.encode(row["esti_question"]).tolist()
            
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "esti_question": row["esti_question"],
                        "esti_answer": row["esti_answer"],
                        "esti_rel_link": row["esti_rel_link"]
                    }
                )
            )
        
        # Upload points in batches
        for i in range(0, len(points), 100):
            batch = points[i:i+100]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            
        return len(points)
    
    def search(self, query_text, limit=1, model_name="sentence-transformers/static-similarity-mrl-multilingual-v1", device="cpu"):
        """Search for similar questions using text query"""
        # 텍스트 쿼리를 임베딩 벡터로 변환
        model = SentenceTransformer(model_name, device=device)
        query_vector = model.encode(query_text).tolist()
        
        # 벡터로 검색 수행
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return search_results 