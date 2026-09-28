import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flask import Flask
from meeting_chunks import MeetingChunk
from meeting_index import LocalMeetingIndex
from meeting_answer_routes import create_meeting_answer_blueprint
from meeting_answer_runtime import create_meeting_answer_completion


class MeetingAnswerRoutesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        chunker = Mock()
        chunker.split.side_effect = lambda text: [MeetingChunk(0, 0, len(text), text, len(text))]
        self.embeddings = Mock(dimensions=3)
        self.embeddings.embed_documents.side_effect = lambda texts: [[1., 0., 0.] for text in texts]
        self.embeddings.embed_query.return_value = [1., 0., 0.]
        self.index = LocalMeetingIndex(Path(directory.name) / 'index.sqlite3', chunker, self.embeddings, pipeline_version='검증용')
        self.text = '검토 회의는 금요일로 변경했습니다.'
        self.index.index(1, 10, self.text)
        self.index.index(2, 20, '다른 프로젝트의 비공개 내용')
        self.provider = Mock(return_value=self.index)
        self.complete = Mock(return_value=json.dumps({'status': 'answered', 'claims': [
            {'text': '검토 회의는 금요일입니다.', 'citations': [{'source_id': 'E1', 'quote': self.text}]}]}))
        app = Flask(__name__)
        app.register_blueprint(create_meeting_answer_blueprint(self.provider, self.complete))
        self.client = app.test_client()

    def ask(self, **kwargs):
        return self.client.post('/answer_meeting', json={'projectId': 1, 'question': '검토 회의는 언제인가요?', **kwargs})

    def test_answer_contains_exact_source_and_only_requested_project_context(self):
        result = self.ask()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['status'], 'answered')
        citation = result.json['claims'][0]['citations'][0]
        self.assertEqual((citation['project_id'], citation['minutes_id']), (1, 10))
        self.assertEqual(self.text[citation['start']:citation['end']], citation['quote'])
        self.complete.assert_called_once()
        messages = self.complete.call_args.args[0]
        self.assertNotIn('비공개', str(messages))

    def test_minutes_filter_excludes_other_meetings_without_generation(self):
        result = self.ask(minutesId=20)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['status'], 'insufficient_evidence')
        self.complete.assert_not_called()
        self.embeddings.embed_query.assert_not_called()

    def test_empty_project_does_not_call_paid_model(self):
        self.assertEqual(self.ask(projectId=3).json['status'], 'insufficient_evidence')
        self.complete.assert_not_called()

    def test_invalid_citation_abstains_without_retry(self):
        self.complete.return_value = json.dumps({'status': 'answered', 'claims': [
            {'text': '검토 회의는 월요일입니다.', 'citations': [{'source_id': 'E1', 'quote': '없는 근거'}]}]})
        with self.assertLogs('meeting_answers', level='WARNING'):
            result = self.ask()
        self.assertEqual(result.json['status'], 'insufficient_evidence')
        self.complete.assert_called_once()

    def test_invalid_request_does_not_initialize_runtime(self):
        for fields in [{'projectId': True}, {'minutesId': '10'}, {'minutesId': 0}, {'question': ' '},
                       {'question': '가' * 2001}, {'question': None}, {'sources': []}]:
            with self.subTest(fields=fields):
                self.assertEqual(self.ask(**fields).status_code, 400)
        self.assertEqual(self.client.post('/answer_meeting', data='{', content_type='application/json').status_code, 400)
        self.provider.assert_not_called()
        self.complete.assert_not_called()

    def test_missing_index_configuration_returns_unavailable(self):
        self.provider.return_value = None
        self.assertEqual(self.ask().status_code, 503)
        self.complete.assert_not_called()

    def test_generation_failure_is_not_disguised_as_lack_of_evidence(self):
        self.complete.side_effect = RuntimeError('민감한 내부 정보')
        with self.assertLogs(self.client.application.logger, level='ERROR'):
            result = self.ask()
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.json, {'error': '회의록 질문을 처리하지 못했습니다.'})
        self.complete.assert_called_once()

    def test_paid_client_is_lazy_reused_and_has_no_automatic_retries(self):
        with patch('openai.OpenAI') as factory, patch('meeting_answers.OpenAIMeetingAnswerModel') as model:
            complete = create_meeting_answer_completion('검증용 키')
            factory.assert_not_called()
            complete([])
            complete([])
            factory.assert_called_once_with(api_key='검증용 키', max_retries=0, timeout=30)
            self.assertEqual(model.return_value.call_count, 2)

    def test_missing_api_key_does_not_create_paid_client(self):
        with patch('openai.OpenAI') as factory:
            complete = create_meeting_answer_completion(None)
            with self.assertRaises(ValueError):
                complete([])
            factory.assert_not_called()
