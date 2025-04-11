import numpy as np
import pandas as pd
import faiss
import pickle
import os
import math
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional

class FaissVectorStore:
    def __init__(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Faiss 벡터 저장소 초기화
        
        Args:
            index_path: Faiss 인덱스 파일 경로 (이미 존재하는 경우)
            metadata_path: 메타데이터 파일 경로 (이미 존재하는 경우)
        """
        self.index = None
        self.metadata = []
        
        # 기존 인덱스 불러오기 (있는 경우)
        if index_path and os.path.exists(index_path) and metadata_path and os.path.exists(metadata_path):
            self.load_index(index_path, metadata_path)
    
    def create_index(self, vector_size: int = 768):
        """
        Faiss 벡터 인덱스 생성
        
        Args:
            vector_size: 벡터 크기
        """
        # L2 거리를 사용하는 인덱스 생성 (cosine 유사도를 위해서는 정규화 필요)
        self.index = faiss.IndexFlatIP(vector_size)  # 내적(Inner Product)으로 cosine 유사도 사용
        return self.index
    
    def load_embeddings_from_excel(self, 
                                 excel_path: str, 
                                 model_name: str = "sentence-transformers/static-similarity-mrl-multilingual-v1", 
                                 device: str = "cpu",
                                 save_path: Optional[str] = None):
        """
        Excel 파일에서 데이터를 로드하고 임베딩 생성
        
        Args:
            excel_path: Excel 파일 경로
            model_name: 임베딩 모델 이름
            device: 모델 실행 디바이스 (cpu 또는 cuda)
            save_path: 생성된 인덱스와 메타데이터 저장 경로 (디렉토리)
        
        Returns:
            생성된 포인트(임베딩) 개수
        """
        try:
            # Excel 파일 로드
            df = pd.read_excel(excel_path)
            
            # 필요한 컬럼이 있는지 확인
            required_columns = ['esti_question', 'esti_answer', 'esti_rel_link']
            
            # 컬럼 이름이 다른 경우 이름 변경
            if len(df.columns) >= 3 and not all(col in df.columns for col in required_columns):
                # 첫 번째 컬럼을 질문으로, 두 번째 컬럼을 답변으로, 세 번째 컬럼을 링크로 간주
                df.columns = required_columns[:len(df.columns)]
            
            # 컬럼이 3개 미만인 경우 오류 발생
            if len(df.columns) < 3:
                raise ValueError(f"Excel 파일에는 최소 3개의 컬럼이 필요합니다. 현재 컬럼 수: {len(df.columns)}")
            
            # 데이터프레임에서 필요한 컬럼만 선택
            df = df[required_columns[:len(df.columns)]]
            
            # NaN 값 처리
            df = df.fillna("")
            
            if len(df) == 0:
                raise ValueError("Excel 파일에서 데이터를 추출할 수 없습니다.")
            
            # 임베딩 모델 초기화
            model = SentenceTransformer(model_name, device=device)
            
            # 벡터 크기 확인하고 인덱스 생성
            if self.index is None:
                sample_embedding = model.encode(df["esti_question"].iloc[0])
                vector_size = len(sample_embedding)
                self.create_index(vector_size=vector_size)
            
            # 임베딩 생성 및 인덱스에 추가
            embeddings = []
            self.metadata = []
            
            for idx, row in df.iterrows():
                # 질문 임베딩 생성
                embedding = model.encode(row["esti_question"])
                embeddings.append(embedding)
                
                # JSON 직렬화 가능한 값만 메타데이터에 저장
                try:
                    # 직렬화 가능한지 테스트
                    metadata_item = {
                        "esti_question": str(row["esti_question"]) if pd.notna(row["esti_question"]) else "",
                        "esti_answer": str(row["esti_answer"]) if pd.notna(row["esti_answer"]) else "",
                        "esti_rel_link": str(row["esti_rel_link"]) if pd.notna(row["esti_rel_link"]) else ""
                    }
                    json.dumps(metadata_item)  # 직렬화 테스트
                    self.metadata.append(metadata_item)
                except (TypeError, ValueError, OverflowError):
                    # 직렬화할 수 없는 경우 기본값 사용
                    self.metadata.append({
                        "esti_question": str(row["esti_question"])[:1000] if pd.notna(row["esti_question"]) else "",
                        "esti_answer": str(row["esti_answer"])[:1000] if pd.notna(row["esti_answer"]) else "",
                        "esti_rel_link": str(row["esti_rel_link"])[:1000] if pd.notna(row["esti_rel_link"]) else ""
                    })
            
            # 임베딩 정규화 (cosine 유사도를 위해)
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # 인덱스에 임베딩 추가
            self.index.add(embeddings_array)
            
            # 저장 경로가 제공된 경우 인덱스와 메타데이터 저장
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                index_path = os.path.join(save_path, "faiss_index.bin")
                metadata_path = os.path.join(save_path, "metadata.pkl")
                
                self.save_index(index_path, metadata_path)
            
            return len(embeddings)
        except Exception as e:
            raise ValueError(f"Excel 파일 로드 중 오류가 발생했습니다. 오류: {str(e)}")
    
    def save_index(self, index_path: str, metadata_path: str):
        """인덱스와 메타데이터를 파일로 저장"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self, index_path: str, metadata_path: str):
        """인덱스와 메타데이터를 파일에서 로드"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
    
    def search(self, 
              query_text: str, 
              limit: int = 3, 
              model_name: str = "sentence-transformers/static-similarity-mrl-multilingual-v1", 
              device: str = "cpu"):
        """
        텍스트 쿼리로 유사한 질문 검색
        
        Args:
            query_text: 검색할 텍스트 쿼리
            limit: 반환할 검색 결과 수
            model_name: 임베딩 모델 이름
            device: 모델 실행 디바이스
        
        Returns:
            검색 결과 (인덱스, 점수, 메타데이터)의 리스트
        """
        if self.index is None:
            raise ValueError("인덱스가 생성되지 않았습니다. 먼저 create_index를 호출하세요.")
        
        # 임베딩 모델 초기화
        model = SentenceTransformer(model_name, device=device)
        
        # 쿼리 텍스트 임베딩
        query_vector = model.encode(query_text).astype('float32').reshape(1, -1)
        
        # 벡터 정규화 (cosine 유사도를 위해)
        faiss.normalize_L2(query_vector)
        
        # 검색 수행
        scores, indices = self.index.search(query_vector, limit)
        
        # 결과 형식화
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata) and idx >= 0:  # 유효한 인덱스인지 확인
                try:
                    # JSON 직렬화를 위해 무한대/NaN 값 처리
                    score_value = float(score)
                    if math.isnan(score_value) or math.isinf(score_value):
                        score_value = 0.0
                    
                    # 메타데이터 복사 및 확인
                    payload = {
                        "esti_question": str(self.metadata[idx].get("esti_question", ""))[:1000],
                        "esti_answer": str(self.metadata[idx].get("esti_answer", ""))[:1000],
                        "esti_rel_link": str(self.metadata[idx].get("esti_rel_link", ""))[:1000]
                    }
                    
                    # 결과 생성
                    result = {
                        "id": int(idx),
                        "score": score_value,
                        "payload": payload
                    }
                    
                    # 직렬화 테스트
                    json.dumps(result)
                    
                    results.append(result)
                except (TypeError, ValueError, OverflowError):
                    # 직렬화 실패 시 해당 결과 건너뜀
                    continue
        
        return results 