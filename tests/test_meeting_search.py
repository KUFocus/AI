import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from meeting_chunks import MeetingChunk
from meeting_index import LocalMeetingIndex


class MeetingSearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'search.sqlite3'
        self.chunker = Mock()
        self.chunker.split.side_effect = lambda text: [MeetingChunk(0, 0, len(text), text, len(text))]
        self.embeddings = Mock(dimensions=3)
        vectors = {'서버 오류': [1., 0., 0.], '디자인 일정': [0., 1., 0.],
                   '관련 자료': [0.6, 0.8, 0.], '다른 프로젝트': [1., 0., 0.]}
        self.embeddings.embed_documents.side_effect = lambda texts: [vectors[text] for text in texts]
        self.embeddings.embed_query.return_value = [1., 0., 0.]
        self.store = LocalMeetingIndex(self.path, self.chunker, self.embeddings, pipeline_version='v1')
        self.store.index(1, 10, '디자인 일정')
        self.store.index(1, 11, '서버 오류')
        self.store.index(1, 12, '관련 자료')
        self.store.index(2, 20, '다른 프로젝트')
        self.embeddings.embed_documents.reset_mock()

    def test_ranked_results_include_source_identity_and_positions(self):
        results = self.store.search(1, '오류 원인은?', top_k=2)
        self.assertEqual([hit.minutes_id for hit in results], [11, 12])
        for hit, expected in zip(results, [1.0, 0.6]):
            self.assertAlmostEqual(hit.score, expected, places=6)
        for hit in results:
            document = self.store.get(1, hit.minutes_id)
            self.assertEqual(hit.text, document['source_text'][hit.start:hit.end])
            self.assertEqual(hit.source_hash, document['source_hash'])
            self.assertEqual(hit.project_id, 1)
        self.embeddings.embed_query.assert_called_once_with('오류 원인은?')
        self.embeddings.embed_documents.assert_not_called()

    def test_single_meeting_filter_and_project_filter_are_combined(self):
        results = self.store.search(1, '오류 원인은?', minutes_id=10)
        self.assertEqual([hit.minutes_id for hit in results], [10])
        self.assertEqual(self.store.search(1, '질문', minutes_id=20), [])
        self.assertEqual([hit.minutes_id for hit in self.store.search(2, '질문')], [20])

    def test_empty_or_incompatible_project_skips_query_embedding(self):
        self.assertEqual(self.store.search(3, '질문'), [])
        different_version = LocalMeetingIndex(self.path, self.chunker, self.embeddings, pipeline_version='v2')
        self.assertEqual(different_version.search(1, '질문'), [])
        self.embeddings.dimensions = 4
        self.assertEqual(self.store.search(1, '질문'), [])
        self.embeddings.embed_query.assert_not_called()

    def test_updated_document_is_used_after_reopening(self):
        self.store.index(1, 11, '디자인 일정')
        reopened = LocalMeetingIndex(self.path, self.chunker, self.embeddings, pipeline_version='v1')
        self.assertEqual(reopened.search(1, '오류 원인은?', top_k=1)[0].minutes_id, 12)
        self.assertEqual(reopened.search(1, '질문', minutes_id=11)[0].text, '디자인 일정')

    def test_ties_have_stable_order_and_top_k_may_exceed_count(self):
        self.store.index(1, 13, '서버 오류')
        hits = self.store.search(1, '질문', top_k=50)
        self.assertEqual([hit.minutes_id for hit in hits], [11, 13, 12, 10])

    def test_invalid_query_vectors_are_not_scored(self):
        for vector in [[1., 0.], [float('nan'), 0, 0], [0, 0, 0], [True, 0, 0]]:
            with self.subTest(vector=vector):
                self.embeddings.embed_query.return_value = vector
                with self.assertRaises(ValueError):
                    self.store.search(1, '질문')

    def test_corrupted_stored_vector_is_not_silently_scored(self):
        document = self.store.get(1, 11)
        document['chunks'][0]['vector'] = [1.0]
        with sqlite3.connect(self.path) as connection:
            connection.execute('UPDATE meeting_index SET chunks_json = ? WHERE project_id=1 AND minutes_id=11',
                               (json.dumps(document['chunks']),))
        with self.assertRaises(ValueError):
            self.store.search(1, '질문')

    def test_invalid_search_input_does_not_embed(self):
        for question in ['', '  ', None]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                self.store.search(1, question)
        for count in [0, 51, True, 1.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.store.search(1, '질문', top_k=count)
        for project, minutes in [(0, None), (True, None), (1, -1)]:
            with self.subTest(project=project, minutes=minutes), self.assertRaises(ValueError):
                self.store.search(project, '질문', minutes_id=minutes)
        self.embeddings.embed_query.assert_not_called()

    def test_ties_at_top_k_boundary_keep_smallest_identity(self):
        for minutes_id in [16, 14, 13, 15]:
            self.store.index(1, minutes_id, '서버 오류')
        self.assertEqual([hit.minutes_id for hit in self.store.search(1, '질문', top_k=2)], [11, 13])

    def test_faiss_matches_previous_dot_product_for_normalized_vectors(self):
        import random
        import math
        randomizer = random.Random(37)
        dimensions = 384
        def unit_vector():
            values = [randomizer.uniform(-1, 1) for _ in range(dimensions)]
            length = math.sqrt(sum(value * value for value in values))
            return [value / length for value in values]
        vectors = {str(i): unit_vector() for i in range(40)}
        query = unit_vector()
        model = Mock(dimensions=dimensions)
        model.embed_documents.side_effect = lambda texts: [vectors[text] for text in texts]
        model.embed_query.return_value = query
        store = LocalMeetingIndex(self.path, self.chunker, model, pipeline_version='비교 버전')
        for i in range(40):
            store.index(5, i + 1, str(i))
        expected = sorted([(sum(a * b for a, b in zip(vector, query)), int(text) + 1)
                           for text, vector in vectors.items()], key=lambda item: (-item[0], item[1]))[:10]
        actual = store.search(5, '비교 질문', top_k=10)
        self.assertEqual([hit.minutes_id for hit in actual], [item[1] for item in expected])
        for hit, (score, _) in zip(actual, expected):
            self.assertAlmostEqual(hit.score, score, places=6)

    def test_deleted_document_is_absent_from_next_faiss_search(self):
        self.assertEqual(self.store.search(1, '질문', top_k=1)[0].minutes_id, 11)
        self.store.delete_saved(self.path, 1, 11)
        self.assertEqual(self.store.search(1, '질문', top_k=1)[0].minutes_id, 12)


if __name__ == '__main__':
    unittest.main()
