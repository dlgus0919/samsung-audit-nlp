import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "upskyy/bge-m3-korean"
CHUNK_SIZE = 400

FINANCIAL_SECTIONS_REPEAT = {"포괄손익", "재무상태표", "현금흐름"}
REPEAT_COUNT = 3  # 재무제표 청크 반복 횟수

@dataclass
class Chunk:
    text: str
    year: int
    section: str
    chunk_id: int

def load_chunks_from_csv(csv_path: str) -> list[Chunk]:
    """
    sections.csv를 읽어 섹션별 텍스트를 CHUNK_SIZE 단위로 분할하여 Chunk 리스트 반환.
    - 길이 30자 미만인 청크는 의미가 부족하여 건너뜀.
    - 빈 content는 건너뜀.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"year", "section", "content"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"sections.csv에 필수 컬럼 누락: {missing}")

    chunks = []
    chunk_idx = 0
    skipped = 0

    for _, row in df.iterrows():
        if pd.isna(row.get("content")) or not str(row.get("content", "")).strip():
            skipped += 1
            continue

        content = str(row["content"])
        year = int(row["year"])
        section = str(row["section"])

        # 단순 문자 단위 슬라이싱 (CHUNK_SIZE 반영)
        for i in range(0, len(content), CHUNK_SIZE):
            prefix = f"[{year}년 {section}] "
            chunk_text = prefix + content[i:i+CHUNK_SIZE].strip()
            # 의미 있는 문장이 되도록 최소 30자 이상만 임베딩 후보로 채택
            if len(chunk_text) >= 30:
                repeat = REPEAT_COUNT if section in FINANCIAL_SECTIONS_REPEAT else 1
                for _ in range(repeat):
                    chunks.append(Chunk(
                        text=chunk_text,
                        year=year,
                        section=section,
                        chunk_id=chunk_idx
                    ))
                    chunk_idx += 1

    if skipped:
        print(f"Warning: {skipped}개 행이 빈 content로 스킵되었습니다.")

    return chunks

@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    """싱글톤 패턴으로 모델을 한 번만 로드"""
    print(f"Loading embed model: {EMBED_MODEL} ...")
    try:
        return SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"임베딩 모델 로드 실패 ({EMBED_MODEL}). "
            f"인터넷 연결 및 디스크 여유공간(~5GB)을 확인하세요.\n원인: {e}"
        ) from e

def build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """
    SentenceTransformer로 청크 텍스트를 임베딩.
    반환: shape (N, D) float32 배열, L2 정규화 적용 (코사인 유사도를 inner product로 계산하기 위함)
    """
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    model = get_embed_model()
    texts = [c.text for c in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)
