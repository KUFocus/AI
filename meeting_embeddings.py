from pathlib import Path

import torch
import torch.nn.functional as functional
from langchain_core.embeddings import Embeddings


class LocalMeetingEmbeddings(Embeddings):
    """E5 모델로 회의 원문 청크와 질문을 임베딩한다. 분할과 저장은 별도로 수행한다."""

    dimensions = 384
    max_tokens = 512
    document_prefix = 'passage: '

    def __init__(self, tokenizer, model, *, batch_size: int = 8):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('임베딩 배치 크기는 1 이상의 정수여야 합니다.')
        self.tokenizer = tokenizer
        self.model = model.to('cpu').eval()
        self.batch_size = batch_size

    @classmethod
    def from_local_directory(cls, directory: str, *, batch_size: int = 8):
        path = Path(directory)
        if not path.is_dir():
            raise ValueError('임베딩 모델을 저장한 로컬 폴더가 없습니다.')
        from transformers import AutoModel, AutoTokenizer

        # 요청 처리 중 모델 다운로드나 외부 모델 코드 실행을 하지 않는다.
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
        if model.config.hidden_size != cls.dimensions:
            raise ValueError('임베딩 모델의 출력 차원이 multilingual-e5-small과 다릅니다.')
        return cls(tokenizer, model, batch_size=batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, prefix=self.document_prefix)

    def count_document_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise ValueError('토큰 수를 계산할 원문은 문자열이어야 합니다.')
        encoded = self.tokenizer(
            self.document_prefix + text, add_special_tokens=True, truncation=False,
        )
        return len(encoded['input_ids'])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], prefix='query: ')[0]

    def _embed(self, texts: list[str], *, prefix: str) -> list[list[float]]:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError('임베딩 입력은 비어 있지 않은 문자열 목록이어야 합니다.')
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                [prefix + text for text in texts[start:start + self.batch_size]],
                padding=True, truncation=False, return_tensors='pt',
            )
            # 접두사와 특수 토큰까지 포함하여 원문이 조용히 잘리는 것을 방지한다.
            if batch['input_ids'].shape[1] > self.max_tokens:
                raise ValueError('임베딩 입력은 접두사와 특수 토큰을 포함해 512토큰 이하여야 합니다. 원문을 먼저 분할해 주세요.')
            with torch.inference_mode():
                output = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].bool().unsqueeze(-1)
                pooled = output.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
                normalized = functional.normalize(pooled, p=2, dim=1)
            if normalized.shape[1] != self.dimensions or not torch.isfinite(normalized).all():
                raise ValueError('임베딩 결과의 차원 또는 숫자 값이 올바르지 않습니다.')
            if not torch.allclose(normalized.norm(dim=1), torch.ones(len(normalized)), atol=1e-5):
                raise ValueError('임베딩 결과를 단위 벡터로 정규화하지 못했습니다.')
            vectors.extend(normalized.tolist())
        return vectors
