import json
import re
from importlib.metadata import version
from pathlib import Path
from threading import Lock


def create_local_index_provider(model_directory, database_path, model_revision):
    """설정이 있을 때 첫 요청에서만 모델을 읽고 이후 요청에 재사용한다."""
    instance = None
    lock = Lock()

    def provide():
        nonlocal instance
        if not all(isinstance(value, str) and value.strip()
                   for value in (model_directory, database_path, model_revision)):
            return None
        with lock:
            if instance is not None:
                return instance
            if not re.fullmatch(r'[0-9a-f]{40}', model_revision):
                raise ValueError('임베딩 모델 리비전은 40자리 커밋 해시여야 합니다.')
            # 서비스 시작 시 OCR과 함께 임베딩 모델을 미리 적재하지 않는다.
            from meeting_embeddings import LocalMeetingEmbeddings
            from meeting_chunks import MeetingChunker
            from meeting_index import LocalMeetingIndex
            from meeting_embedding_cache import CachedMeetingEmbeddings

            model_path = Path(model_directory)
            manifest = json.loads((model_path / 'model-info.json').read_text())
            if manifest.get('model') != 'intfloat/multilingual-e5-small' or manifest.get('revision') != model_revision:
                raise ValueError('로컬 모델 정보와 설정한 임베딩 모델 리비전이 다릅니다.')
            pipeline_version = json.dumps({
                'model': manifest['model'], 'revision': model_revision,
                'embedding': 'masked-mean-l2-passage-v1',
                'splitter': 'RecursiveCharacterTextSplitter',
                'splitter_package': version('langchain-text-splitters'),
                'chunk_mapping': 'source-spans-v1', 'max_tokens': 512, 'overlap_tokens': 64,
            }, sort_keys=True)
            embeddings = LocalMeetingEmbeddings.from_local_directory(model_directory)
            chunker = MeetingChunker(embeddings.count_document_tokens)
            # 경로는 서버 설정에서만 받으며 요청 본문으로 지정할 수 없다.
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
            model_identity = json.dumps({
                'model': manifest['model'], 'revision': model_revision,
                'pooling': 'masked-mean-l2-v1', 'dimensions': embeddings.dimensions,
                'document_prefix': 'passage: ', 'query_prefix': 'query: ',
            }, sort_keys=True)
            cached_embeddings = CachedMeetingEmbeddings(
                embeddings, str(database_path) + '.embeddings', model_identity=model_identity,
            )
            instance = LocalMeetingIndex(database_path, chunker, cached_embeddings, pipeline_version=pipeline_version)
            return instance

    return provide
