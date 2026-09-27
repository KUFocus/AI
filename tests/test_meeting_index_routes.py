import unittest
from unittest.mock import Mock

from flask import Flask

from meeting_index import IndexResult
from meeting_index_routes import create_meeting_index_blueprint


class MeetingIndexRoutesTests(unittest.TestCase):
    def setUp(self):
        self.index = Mock()
        self.index.index.return_value = IndexResult(True, 2, 'hash')
        self.provider = Mock(return_value=self.index)
        app = Flask(__name__)
        app.register_blueprint(create_meeting_index_blueprint(self.provider))
        self.client = app.test_client()
        self.payload = {'projectId': 1, 'minutesId': 10, 'text': '  원문입니다.\n'}

    def test_success_preserves_raw_text_and_returns_index_state(self):
        result = self.client.post('/index_meeting', json=self.payload)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, {'projectId': 1, 'minutesId': 10, 'changed': True,
                                       'chunkCount': 2, 'sourceHash': 'hash'})
        self.index.index.assert_called_once_with(1, 10, '  원문입니다.\n')

    def test_unchanged_result_is_returned(self):
        self.index.index.return_value = IndexResult(False, 2, 'hash')
        result = self.client.post('/index_meeting', json=self.payload)
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json['changed'])

    def test_invalid_json_or_input_does_not_initialize_model(self):
        payloads = [None, [], {}, {**self.payload, 'projectId': True}, {**self.payload, 'projectId': '1'},
                    {**self.payload, 'minutesId': 0}, {**self.payload, 'minutesId': 2**63},
                    {**self.payload, 'text': '  '}, {**self.payload, 'text': None},
                    {**self.payload, 'text': '가' * 100001}, {**self.payload, 'databasePath': '/tmp/other'}]
        for payload in payloads:
            with self.subTest(payload_type=type(payload)):
                result = self.client.post('/index_meeting', json=payload)
                self.assertEqual(result.status_code, 400)
                self.assertIn('error', result.json)
        self.assertEqual(self.client.post('/index_meeting', data='{', content_type='application/json').status_code, 400)
        self.provider.assert_not_called()

    def test_missing_runtime_configuration_returns_503(self):
        self.provider.return_value = None
        result = self.client.post('/index_meeting', json=self.payload)
        self.assertEqual(result.status_code, 503)
        self.index.index.assert_not_called()

    def test_internal_failure_returns_generic_korean_error(self):
        for target in [self.provider, self.index.index]:
            target.side_effect = RuntimeError('내부 경로와 민감한 상세 정보')
            with self.assertLogs(self.client.application.logger, level='ERROR'):
                result = self.client.post('/index_meeting', json=self.payload)
            self.assertEqual(result.status_code, 500)
            self.assertEqual(result.json, {'error': '회의록을 색인하지 못했습니다.'})
            self.assertNotIn('민감한', result.get_data(as_text=True))
            target.side_effect = None


if __name__ == '__main__':
    unittest.main()
