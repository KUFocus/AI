import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexResult:
    changed: bool
    chunk_count: int
    source_hash: str


class LocalMeetingIndex:
    """회의별 원문과 벡터를 로컬에 저장한다. 검색과 접근 권한 검사는 별도 계층의 역할이다."""

    def __init__(self, database_path, chunker, embeddings, *, pipeline_version: str):
        if not isinstance(pipeline_version, str) or not pipeline_version.strip():
            raise ValueError('모델 리비전과 청크 설정을 구분할 파이프라인 버전이 필요합니다.')
        self.database_path = str(Path(database_path))
        if self.database_path == ':memory:':
            raise ValueError('회의록 인덱스에는 영구 저장할 파일 경로가 필요합니다.')
        self.chunker = chunker
        self.embeddings = embeddings
        self.pipeline_version = pipeline_version
        with closing(self._connect()) as connection, connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS meeting_index (
                project_id INTEGER NOT NULL,
                minutes_id INTEGER NOT NULL,
                source_hash TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                source_text TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                chunks_json TEXT NOT NULL,
                PRIMARY KEY (project_id, minutes_id)
            )''')

    def _connect(self):
        return sqlite3.connect(self.database_path, timeout=30)

    @staticmethod
    def _validate_ids(project_id, minutes_id):
        if any(type(value) is not int or not 0 < value <= 2**63 - 1
               for value in (project_id, minutes_id)):
            raise ValueError('프로젝트 ID와 회의록 ID는 양의 64비트 정수여야 합니다.')

    def index(self, project_id: int, minutes_id: int, text: str) -> IndexResult:
        self._validate_ids(project_id, minutes_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('저장할 회의 원문은 비어 있지 않은 문자열이어야 합니다.')
        source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        with closing(self._connect()) as connection, connection:
            # 로컬 프로토타입에서는 저장 작업을 직렬화해 동시 중복 요청도 재계산하지 않는다.
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute(
                'SELECT source_hash, pipeline_version, dimensions, chunks_json FROM meeting_index '
                'WHERE project_id = ? AND minutes_id = ?', (project_id, minutes_id),
            ).fetchone()
            if previous and previous[:3] == (source_hash, self.pipeline_version, self.embeddings.dimensions):
                return IndexResult(False, len(json.loads(previous[3])), source_hash)
            chunks = self.chunker.split(text)
            if not chunks:
                raise ValueError('저장할 원문 청크가 없습니다.')
            vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise ValueError('원문 청크 수와 임베딩 벡터 수가 다릅니다.')
            records = []
            for chunk, vector in zip(chunks, vectors):
                if not 0 <= chunk.start < chunk.end <= len(text) or text[chunk.start:chunk.end] != chunk.text:
                    raise ValueError('청크의 위치와 회의 원문이 일치하지 않습니다.')
                if len(vector) != self.embeddings.dimensions or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in vector
                ):
                    raise ValueError('저장할 임베딩 벡터의 차원 또는 숫자 값이 올바르지 않습니다.')
                if not math.isclose(sum(value * value for value in vector), 1, abs_tol=1e-4):
                    raise ValueError('저장할 임베딩 벡터는 단위 벡터여야 합니다.')
                records.append({'index': chunk.index, 'start': chunk.start, 'end': chunk.end,
                                'text': chunk.text, 'token_count': chunk.token_count, 'vector': vector})
            connection.execute('''INSERT INTO meeting_index VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, minutes_id) DO UPDATE SET
                source_hash=excluded.source_hash, pipeline_version=excluded.pipeline_version,
                source_text=excluded.source_text, dimensions=excluded.dimensions,
                chunks_json=excluded.chunks_json''',
                (project_id, minutes_id, source_hash, self.pipeline_version, text,
                 self.embeddings.dimensions, json.dumps(records, ensure_ascii=False, allow_nan=False)))
            return IndexResult(True, len(records), source_hash)

    def get(self, project_id: int, minutes_id: int) -> dict | None:
        self._validate_ids(project_id, minutes_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                'SELECT source_hash, pipeline_version, source_text, dimensions, chunks_json '
                'FROM meeting_index WHERE project_id = ? AND minutes_id = ?',
                (project_id, minutes_id),
            ).fetchone()
        if row is None:
            return None
        return {'project_id': project_id, 'minutes_id': minutes_id, 'source_hash': row[0],
                'pipeline_version': row[1], 'source_text': row[2],
                'dimensions': row[3], 'chunks': json.loads(row[4])}
