import tempfile
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from meeting_chunks import MeetingChunk
from meeting_index import LocalMeetingIndex


class MeetingIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'index.sqlite3'
        self.chunker = Mock()
        self.chunker.split.side_effect = lambda text: [MeetingChunk(0, 0, len(text), text, len(text))]
        self.embeddings = Mock(dimensions=3)
        self.embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for text in texts]
        self.store = self.make_store()

    def make_store(self, version='e5-revision-1:splitter-1:512:64'):
        return LocalMeetingIndex(self.path, self.chunker, self.embeddings, pipeline_version=version)

    def test_data_survives_reopening_and_other_project_cannot_read_it(self):
        text = '  원문을 확인합니다.\n'
        result = self.store.index(1, 10, text)
        self.assertTrue(result.changed)
        loaded = self.make_store().get(1, 10)
        self.assertEqual(loaded['source_text'], text)
        self.assertEqual(loaded['chunks'][0]['vector'], [1.0, 0.0, 0.0])
        self.assertEqual(loaded['chunks'][0]['end'], len(text))
        self.assertIsNone(self.store.get(2, 10))
        self.store.index(2, 10, '다른 프로젝트')
        self.assertEqual(self.store.get(1, 10)['source_text'], text)

    def test_identical_source_skips_chunking_and_embedding_after_restart(self):
        first = self.store.index(1, 10, '회의 원문')
        second = self.make_store().index(1, 10, '회의 원문')
        self.assertFalse(second.changed)
        self.assertEqual(second.source_hash, first.source_hash)
        self.assertEqual(self.chunker.split.call_count, 1)
        self.assertEqual(self.embeddings.embed_documents.call_count, 1)

    def test_updated_source_replaces_all_old_chunks(self):
        self.chunker.split.side_effect = None
        self.chunker.split.return_value = [MeetingChunk(0, 0, 1, '가', 1), MeetingChunk(1, 1, 2, '나', 1)]
        self.store.index(1, 10, '가나')
        self.chunker.split.return_value = [MeetingChunk(0, 0, 1, '다', 1)]
        self.assertTrue(self.store.index(1, 10, '다').changed)
        loaded = self.store.get(1, 10)
        self.assertEqual([c['text'] for c in loaded['chunks']], ['다'])
        self.assertEqual(loaded['source_text'], '다')

    def test_pipeline_version_change_reembeds_unchanged_source(self):
        self.store.index(1, 10, '원문')
        changed = self.make_store(version='e5-revision-2:splitter-1:512:64').index(1, 10, '원문')
        self.assertTrue(changed.changed)
        self.assertEqual(self.embeddings.embed_documents.call_count, 2)

    def test_embedding_failure_preserves_previous_document(self):
        self.store.index(1, 10, '기존 원문')
        before = self.store.get(1, 10)
        self.embeddings.embed_documents.side_effect = RuntimeError('임베딩 실패')
        with self.assertRaises(RuntimeError):
            self.store.index(1, 10, '수정 원문')
        self.assertEqual(self.store.get(1, 10), before)

    def test_invalid_vectors_preserve_previous_document(self):
        self.store.index(1, 10, '기존 원문')
        before = self.store.get(1, 10)
        self.embeddings.embed_documents.side_effect = None
        for output in [[], [[1.0]], [[float('nan'), 0, 0]], [[float('inf'), 0, 0]], [[0, 0, 0]], [[True, 0, 0]]]:
            with self.subTest(output=output):
                self.embeddings.embed_documents.return_value = output
                with self.assertRaises(ValueError):
                    self.store.index(1, 10, '새 원문')
                self.assertEqual(self.store.get(1, 10), before)

    def test_database_failure_preserves_previous_document(self):
        self.store.index(1, 10, '기존 원문')
        before = self.store.get(1, 10)
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER reject_write BEFORE INSERT ON meeting_index "
                               "BEGIN SELECT RAISE(ABORT, '저장 실패'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.index(1, 10, '수정 원문')
        self.assertEqual(self.store.get(1, 10), before)

    def test_bad_source_offsets_are_not_stored(self):
        self.chunker.split.side_effect = None
        self.chunker.split.return_value = [MeetingChunk(0, 1, 5, '다른 원문', 4)]
        with self.assertRaises(ValueError):
            self.store.index(1, 10, '새 원문')
        self.assertIsNone(self.store.get(1, 10))

    def test_concurrent_duplicate_requests_embed_only_once(self):
        other = self.make_store()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda store: store.index(1, 10, '동일 원문'), [self.store, other]))
        self.assertEqual(sorted(result.changed for result in results), [False, True])
        self.assertEqual(self.embeddings.embed_documents.call_count, 1)

    def test_invalid_identity_and_source_do_not_embed(self):
        for ids in [(0, 1), (1, -1), (True, 1), ('1', 1), (2**63, 1)]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.store.index(*ids, '원문')
        for text in ['', ' \n', None]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.store.index(1, 10, text)
        self.embeddings.embed_documents.assert_not_called()


if __name__ == '__main__':
    unittest.main()
