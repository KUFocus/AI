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


@dataclass(frozen=True)
class SearchHit:
    project_id: int
    minutes_id: int
    chunk_index: int
    text: str
    start: int
    end: int
    score: float
    source_hash: str


class DeletedMeetingError(ValueError):
    """이미 삭제된 회의록의 지연 색인 요청을 구분한다."""


class LocalMeetingIndex:
    """회의별 원문과 벡터를 저장하고 프로젝트 안에서 검색한다. 접근 권한 검사는 별도 계층에서 수행한다."""

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
            self._initialize_schema(connection)

    @staticmethod
    def _initialize_schema(connection):
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
        connection.execute('''CREATE TABLE IF NOT EXISTS meeting_index_deletions (
            project_id INTEGER NOT NULL,
            minutes_id INTEGER NOT NULL,
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
            deleted = connection.execute(
                'SELECT 1 FROM meeting_index_deletions WHERE project_id = ? AND minutes_id = ?',
                (project_id, minutes_id),
            ).fetchone()
            if deleted:
                raise DeletedMeetingError('삭제된 회의록은 다시 색인할 수 없습니다.')
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
                self._validate_vector(vector)
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

    @classmethod
    def delete_saved(cls, database_path, project_id: int, minutes_id: int) -> bool:
        """모델을 적재하지 않고 해당 프로젝트의 회의 원문과 벡터를 함께 삭제한다."""
        cls._validate_ids(project_id, minutes_id)
        path = Path(database_path).resolve()
        # 첫 색인보다 삭제 요청이 먼저 와도 삭제 기록을 보존한다.
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=30)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            cls._initialize_schema(connection)
            connection.execute(
                'INSERT OR IGNORE INTO meeting_index_deletions VALUES (?, ?)',
                (project_id, minutes_id),
            )
            cursor = connection.execute(
                'DELETE FROM meeting_index WHERE project_id = ? AND minutes_id = ?',
                (project_id, minutes_id),
            )
            return cursor.rowcount > 0

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

    def _validate_vector(self, vector):
        if not isinstance(vector, list) or len(vector) != self.embeddings.dimensions or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in vector
        ):
            raise ValueError('임베딩 벡터의 차원 또는 숫자 값이 올바르지 않습니다.')
        if not math.isclose(sum(value * value for value in vector), 1, abs_tol=1e-4):
            raise ValueError('임베딩 벡터는 단위 벡터여야 합니다.')

    def search(self, project_id: int, question: str, *, minutes_id: int | None = None,
               top_k: int = 5) -> list[SearchHit]:
        """유사한 근거 후보를 반환한다. 유사도는 답변 가능성이나 정확도 확률이 아니다."""
        self._validate_ids(project_id, 1 if minutes_id is None else minutes_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError('검색 질문은 비어 있지 않은 문자열이어야 합니다.')
        if type(top_k) is not int or not 1 <= top_k <= 50:
            raise ValueError('검색 결과 수는 1 이상 50 이하의 정수여야 합니다.')
        sql = ('SELECT minutes_id, source_hash, chunks_json FROM meeting_index '
               'WHERE project_id = ? AND pipeline_version = ? AND dimensions = ?')
        parameters = [project_id, self.pipeline_version, self.embeddings.dimensions]
        if minutes_id is not None:
            sql += ' AND minutes_id = ?'
            parameters.append(minutes_id)
        # 프로젝트와 임베딩 버전이 맞는 데이터만 읽어 다른 벡터 공간과 섞지 않는다.
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        if not rows:
            return []
        query = self.embeddings.embed_query(question)
        self._validate_vector(query)
        hits = []
        for stored_minutes_id, source_hash, chunks_json in rows:
            for chunk in json.loads(chunks_json):
                vector = chunk['vector']
                self._validate_vector(vector)
                # 저장과 질문 양쪽이 단위 벡터이므로 내적이 코사인 유사도다.
                score = max(-1.0, min(1.0, sum(a * b for a, b in zip(query, vector))))
                hits.append(SearchHit(project_id, stored_minutes_id, chunk['index'],
                                      chunk['text'], chunk['start'], chunk['end'], score, source_hash))
        return sorted(hits, key=lambda hit: (-hit.score, hit.minutes_id, hit.chunk_index))[:top_k]
