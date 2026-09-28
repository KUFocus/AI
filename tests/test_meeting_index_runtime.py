import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, Mock

from meeting_index_runtime import create_local_index_provider


class MeetingIndexRuntimeTests(unittest.TestCase):
    def test_missing_configuration_does_not_load_model(self):
        with patch('meeting_embeddings.LocalMeetingEmbeddings.from_local_directory') as load:
            for config in [(None, 'db', 'a' * 40), ('model', '', 'a' * 40), ('model', 'db', None)]:
                self.assertIsNone(create_local_index_provider(*config)())
            load.assert_not_called()

    def test_first_concurrent_requests_share_one_model_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model-info.json').write_text(json.dumps({'model': 'intfloat/multilingual-e5-small', 'revision': 'a' * 40}))
            model = Mock(dimensions=384)
            model.count_document_tokens.return_value = 4
            with patch('meeting_embeddings.LocalMeetingEmbeddings.from_local_directory', return_value=model) as load:
                provider = create_local_index_provider(directory, str(root/'data/index.sqlite3'), 'a' * 40)
                load.assert_not_called()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    values = list(pool.map(lambda _: provider(), range(2)))
                self.assertIs(values[0], values[1])
                self.assertIs(values[0], provider())
                load.assert_called_once_with(directory)
                self.assertEqual(json.loads(values[0].pipeline_version)['revision'], 'a' * 40)
                self.assertTrue((root/'data/index.sqlite3').exists())

    def test_revision_mismatch_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'model-info.json').write_text(json.dumps({'model': 'intfloat/multilingual-e5-small', 'revision': 'b' * 40}))
            with patch('meeting_embeddings.LocalMeetingEmbeddings.from_local_directory') as load:
                for revision in ['main', 'a' * 40]:
                    with self.subTest(revision=revision), self.assertRaises(ValueError):
                        create_local_index_provider(directory, str(Path(directory)/'index.db'), revision)()
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
