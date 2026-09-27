import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings
from meeting_embedding_cache import CachedMeetingEmbeddings
from meeting_chunks import MeetingChunk
from meeting_index import LocalMeetingIndex


class CountingEmbeddings(Embeddings):
    dimensions = 3

    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[1., 0., 0.] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [0., 1., 0.]


class CachedMeetingEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = CountingEmbeddings()
        self.cached = self.wrap(self.model)

    def wrap(self, model, identity='e5:첫 버전:정규화:접두사'):
        return CachedMeetingEmbeddings(model, self.root / 'cache', model_identity=identity)

    def test_only_new_chunks_are_embedded(self):
        first = self.cached.embed_documents(['변경 없는 내용', '이전 일정'])
        second = self.cached.embed_documents(['변경 없는 내용', '수정된 일정'])
        self.assertEqual(self.model.documents, ['변경 없는 내용', '이전 일정', '수정된 일정'])
        self.assertEqual(first, second)

    def test_restart_reuses_persistent_vectors(self):
        self.cached.embed_documents(['회의 내용'])
        self.cached.embed_query('질문')
        next_model = CountingEmbeddings()
        reopened = self.wrap(next_model)
        self.assertEqual(reopened.embed_documents(['회의 내용']), [[1., 0., 0.]])
        self.assertEqual(reopened.embed_query('질문'), [0., 1., 0.])
        self.assertEqual(next_model.documents, [])
        self.assertEqual(next_model.queries, [])

    def test_same_text_has_separate_document_and_query_cache(self):
        self.assertEqual(self.cached.embed_documents(['동일한 문자열']), [[1., 0., 0.]])
        self.assertEqual(self.cached.embed_query('동일한 문자열'), [0., 1., 0.])
        self.cached.embed_query('동일한 문자열')
        self.assertEqual(self.model.queries, ['동일한 문자열'])

    def test_model_revision_or_settings_change_does_not_reuse_old_vectors(self):
        self.cached.embed_documents(['회의 내용'])
        other = CountingEmbeddings()
        self.wrap(other, 'e5:변경된 버전:정규화:접두사').embed_documents(['회의 내용'])
        self.assertEqual(other.documents, ['회의 내용'])

    def test_partial_meeting_update_reuses_unchanged_chunk(self):
        class TwoChunks:
            def split(self, text):
                left, right = text.split('|')
                return [MeetingChunk(0, 0, len(left), left, 1),
                        MeetingChunk(1, len(left) + 1, len(text), right, 1)]
        store = LocalMeetingIndex(self.root / 'index.db', TwoChunks(), self.cached, pipeline_version='분할 버전')
        store.index(1, 10, '변경 없는 내용|이전 일정')
        store.index(1, 10, '변경 없는 내용|수정된 일정')
        self.assertEqual(len(self.model.documents), 3)
        self.assertEqual(store.get(1, 10)['source_text'], '변경 없는 내용|수정된 일정')

    def test_empty_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            self.wrap(self.model, '')
