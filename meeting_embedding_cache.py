import hashlib
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore


class CachedMeetingEmbeddings(Embeddings):
    """모델 설정별로 청크와 질문 벡터를 로컬 파일에 재사용한다."""

    def __init__(self, underlying, cache_directory, *, model_identity: str):
        if not isinstance(model_identity, str) or not model_identity.strip():
            raise ValueError('임베딩 캐시를 구분할 모델 설정이 필요합니다.')
        self.dimensions = underlying.dimensions
        namespace = hashlib.sha256(model_identity.encode('utf-8')).hexdigest() + '-'
        root = Path(cache_directory)
        # E5는 같은 텍스트도 문서와 질문의 접두사가 다르므로 저장소를 분리한다.
        self.cached = CacheBackedEmbeddings.from_bytes_store(
            underlying,
            LocalFileStore(root / 'documents'),
            namespace=namespace,
            query_embedding_cache=LocalFileStore(root / 'queries'),
            key_encoder='sha256',
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.cached.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.cached.embed_query(text)
