import os
import faiss
import numpy as np
import pickle
from dataclasses import asdict
from src.rag.embedder import Chunk

class VectorStore:
    def __init__(self, index_path: str = "data/processed/faiss.index",
                 meta_path: str = "data/processed/chunks_meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.chunks_meta = []
        
        self._load_if_exists()
        
    def _load_if_exists(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.chunks_meta = pickle.load(f)
                print(f"Loaded existing index with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"Failed to load index or meta: {e}. Starting fresh.")
                self.index = None
                self.chunks_meta = []

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        FAISS IndexFlatIP 인덱스 생성
        """
        if embeddings.shape[0] == 0:
            print("No embeddings to build.")
            return
            
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        self.chunks_meta = [asdict(c) for c in chunks]
        
        # 디스크에 저장
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks_meta, f)
            
        print(f"Built and saved index with {self.index.ntotal} vectors.")

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[dict]:
        """
        쿼리 벡터로 유사도 상위 k개 검색.
        반환: [{"text": str, "year": int, "section": str, "score": float}, ...]
        """
        if not self.is_built():
            print("Warning: Index is not built yet.")
            return []
            
        # 2D 배열로 맞춤
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        # 임베딩 모델이 바뀐 상태에서 예전 FAISS 인덱스를 재사용하면 차원이 달라질 수 있음.
        # FAISS 내부 assert 대신 명확한 예외 메시지를 던져 상위 레이어에서 재빌드하게 한다.
        query_dim = int(query_vec.shape[1])
        index_dim = int(self.index.d)
        if query_dim != index_dim:
            raise ValueError(
                f"Embedding dimension mismatch: query={query_dim}, index={index_dim}. "
                "Please rebuild the vector index."
            )
            
        k_search = min(k, self.index.ntotal)
        if k_search == 0:
            return []

        distances, indices = self.index.search(query_vec, k_search)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_meta):
                continue
            meta = self.chunks_meta[idx]
            result = {
                "text": meta["text"],
                "year": meta["year"],
                "section": meta["section"],
                "score": float(dist)
            }
            results.append(result)
            
        return results

    def is_built(self) -> bool:
        """인덱스가 빌드되어 있는지 여부"""
        return self.index is not None and self.index.ntotal > 0

    def get_index_dim(self) -> int | None:
        """빌드된 인덱스의 임베딩 차원. 미빌드 상태면 None."""
        if not self.is_built():
            return None
        return int(self.index.d)
