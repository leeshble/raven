from faiss_vector_store import FaissVectorStore
from typing import Tuple, Optional
import os

def create_faiss_embeddings(
    csv_path: str,
    model_name: str = "sentence-transformers/static-similarity-mrl-multilingual-v1",
    device: str = "cpu",
    save_dir: str = "faiss_data"
) -> Tuple[bool, str, int]:
    """
    CSV에서 임베딩을 생성하고 Faiss 인덱스로 저장
    
    Args:
        csv_path: CSV 파일 경로
        model_name: 임베딩 모델 이름
        device: 모델 실행 디바이스 (cpu 또는 cuda)
        save_dir: 인덱스와 메타데이터를 저장할 디렉토리
    
    Returns:
        성공 여부, 메시지, 생성된 임베딩 개수
    """
    try:
        # 저장 디렉토리가 없으면 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Faiss 벡터 저장소 초기화
        vector_store = FaissVectorStore()
        
        # CSV에서 임베딩 생성 및 인덱스에 추가
        num_points = vector_store.load_embeddings_from_csv(
            csv_path=csv_path,
            model_name=model_name,
            device=device,
            save_path=save_dir
        )
        
        return True, f"총 {num_points}개 데이터가 Faiss 인덱스로 저장되었습니다.", num_points
    except Exception as e:
        return False, f"Faiss 임베딩 생성 중 오류 발생: {str(e)}", 0 