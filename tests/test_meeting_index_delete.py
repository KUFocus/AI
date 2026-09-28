import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from flask import Flask
from meeting_chunks import MeetingChunk
from meeting_index import LocalMeetingIndex, DeletedMeetingError
from meeting_index_routes import create_meeting_index_blueprint


class MeetingIndexDeleteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'index.sqlite3'
        chunker = Mock()
        chunker.split.side_effect = lambda text: [MeetingChunk(0, 0, len(text), text, len(text))]
        self.embeddings = Mock(dimensions=3)
        self.embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for text in texts]
        self.store = LocalMeetingIndex(self.path, chunker, self.embeddings, pipeline_version='이전 모델 버전')
        self.store.index(1, 10, '삭제할 원문')
        self.store.index(2, 10, '다른 프로젝트 원문')
        self.store.index(1, 11, '다른 회의록 원문')
        self.provider = Mock(side_effect=AssertionError('삭제할 때 모델을 초기화하면 안 됩니다.'))
        self.client = self.make_client(str(self.path))

    def make_client(self, path):
        app = Flask(__name__)
        app.register_blueprint(create_meeting_index_blueprint(self.provider, database_path=path))
        return app.test_client()

    def delete(self, client=None, payload=None):
        return (client or self.client).delete('/index_meeting', json=payload or {'projectId': 1, 'minutesId': 10})

    def test_deletion_removes_source_vectors_and_search_results_only_in_scope(self):
        result = self.delete()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, {'projectId': 1, 'minutesId': 10, 'deleted': True})
        self.assertIsNone(self.store.get(1, 10))
        self.assertEqual(self.store.search(1, '원문', minutes_id=10), [])
        self.assertIsNotNone(self.store.get(2, 10))
        self.assertIsNotNone(self.store.get(1, 11))
        self.provider.assert_not_called()
        self.embeddings.embed_query.assert_not_called()

    def test_repeated_deletion_is_successful_without_model_loading(self):
        self.delete()
        result = self.delete()
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json['deleted'])
        self.provider.assert_not_called()

    def test_delete_before_first_index_persists_deletion(self):
        path = self.path.parent / 'missing.sqlite3'
        result = self.delete(self.make_client(str(path)))
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json['deleted'])
        self.assertTrue(path.exists())
        store = LocalMeetingIndex(path, self.store.chunker, self.embeddings, pipeline_version='새 버전')
        with self.assertRaises(DeletedMeetingError):
            store.index(1, 10, '늦게 도착한 원문')

    def test_invalid_identifiers_do_not_delete_anything(self):
        for value in [True, '1', 0, -1, 2**63, None]:
            with self.subTest(value=value):
                self.assertEqual(self.delete(payload={'projectId': value, 'minutesId': 10}).status_code, 400)
                with self.assertRaises(ValueError):
                    LocalMeetingIndex.delete_saved(self.path, value, 10)
        self.assertEqual(self.delete(payload={'projectId': 1, 'minutesId': 10, 'text': '원문'}).status_code, 400)
        self.assertIsNotNone(self.store.get(1, 10))
        self.provider.assert_not_called()

    def test_missing_configuration_is_unavailable(self):
        self.assertEqual(self.delete(self.make_client(None)).status_code, 503)
        self.assertIsNotNone(self.store.get(1, 10))

    def test_failed_delete_preserves_data_and_hides_internal_error(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON meeting_index "
                               "BEGIN SELECT RAISE(ABORT, '내부 삭제 오류'); END")
        with self.assertLogs(self.client.application.logger, level='ERROR'):
            result = self.delete()
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.json, {'error': '회의록 색인을 삭제하지 못했습니다.'})
        self.assertIsNotNone(self.store.get(1, 10))

    def test_late_index_is_rejected_after_restart_without_embedding(self):
        self.delete()
        self.embeddings.reset_mock()
        store = LocalMeetingIndex(self.path, self.store.chunker, self.embeddings, pipeline_version='새 버전')
        with self.assertRaises(DeletedMeetingError):
            store.index(1, 10, '늦게 도착한 원문')
        self.embeddings.embed_documents.assert_not_called()
        self.assertIsNone(store.get(1, 10))
        self.assertIsNotNone(store.get(2, 10))

    def test_late_index_http_request_returns_conflict(self):
        self.delete()
        app = Flask(__name__)
        app.register_blueprint(create_meeting_index_blueprint(lambda: self.store, str(self.path)))
        result = app.test_client().post('/index_meeting', json={'projectId': 1, 'minutesId': 10, 'text': '늦은 원문'})
        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.json, {'error': '삭제된 회의록은 다시 색인할 수 없습니다.'})

    def test_failed_delete_rolls_back_deletion_record_too(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON meeting_index "
                               "BEGIN SELECT RAISE(ABORT, '삭제 실패'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            LocalMeetingIndex.delete_saved(self.path, 1, 10)
        self.assertTrue(self.store.index(1, 10, '정상 수정').changed)

    def test_delete_waits_for_in_progress_index_then_removes_it(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        entered, release, delete_started = Event(), Event(), Event()

        def embed(texts):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('테스트 임베딩 대기 시간이 초과됐습니다.')
            return [[1.0, 0.0, 0.0] for text in texts]

        def delete():
            delete_started.set()
            return LocalMeetingIndex.delete_saved(self.path, 1, 10)

        self.embeddings.embed_documents.side_effect = embed
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending_index = pool.submit(self.store.index, 1, 10, '진행 중인 수정')
            try:
                self.assertTrue(entered.wait(5))
                pending_delete = pool.submit(delete)
                self.assertTrue(delete_started.wait(5))
            finally:
                release.set()
            self.assertTrue(pending_index.result(timeout=5).changed)
            self.assertTrue(pending_delete.result(timeout=5))
        self.assertIsNone(self.store.get(1, 10))
        with self.assertRaises(DeletedMeetingError):
            self.store.index(1, 10, '다시 도착한 수정')
