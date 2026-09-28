import unittest
from unittest.mock import Mock

from meeting_answer_workflow import build_meeting_answer_workflow


class MeetingAnswerWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.search = Mock(return_value=['근거 후보'])
        self.context = Mock(return_value=[{'source_id': 'E1', 'text': '원문'}])
        self.generate = Mock(return_value='모델 응답')
        self.valid = {'status': 'answered', 'answer': '답변', 'claims': []}
        self.held = {'status': 'insufficient_evidence', 'answer': '근거 부족', 'claims': []}
        self.validate = Mock(return_value=self.valid)
        self.abstain = Mock(return_value=self.held)
        self.graph = build_meeting_answer_workflow(self.search, self.context, self.generate,
                                                   self.validate, self.abstain)
        self.request = {'project_id': 1, 'minutes_id': 10, 'question': '질문'}

    def run_nodes(self):
        updates = list(self.graph.stream(self.request, stream_mode='updates'))
        return [next(iter(update)) for update in updates], updates

    def test_answered_path_generates_once_and_preserves_scope(self):
        nodes, updates = self.run_nodes()
        self.assertEqual(nodes, ['retrieve', 'prepare_context', 'generate', 'validate_citations'])
        self.assertEqual(updates[-1]['validate_citations']['result'], self.valid)
        self.search.assert_called_once_with(1, '질문', minutes_id=10, top_k=3)
        self.context.assert_called_once_with(1, ['근거 후보'])
        self.generate.assert_called_once_with('질문', self.context.return_value)
        self.validate.assert_called_once_with(1, '모델 응답', self.context.return_value)
        self.abstain.assert_not_called()

    def test_no_hits_skips_context_and_generation(self):
        self.search.return_value = []
        nodes, updates = self.run_nodes()
        self.assertEqual(nodes, ['retrieve', 'abstain'])
        self.assertEqual(updates[0]['retrieve']['abstention_reason'], '검색 결과 없음')
        self.assertEqual(updates[-1]['abstain']['result'], self.held)
        self.context.assert_not_called()
        self.generate.assert_not_called()
        self.validate.assert_not_called()

    def test_empty_context_skips_generation(self):
        self.context.return_value = []
        nodes, _ = self.run_nodes()
        self.assertEqual(nodes, ['retrieve', 'prepare_context', 'abstain'])
        self.generate.assert_not_called()
        self.validate.assert_not_called()

    def test_insufficient_answer_routes_to_abstention_without_retry(self):
        self.validate.return_value = self.held
        nodes, updates = self.run_nodes()
        self.assertEqual(nodes, ['retrieve', 'prepare_context', 'generate', 'validate_citations', 'abstain'])
        self.assertEqual(updates[-1]['abstain']['result'], self.held)
        self.generate.assert_called_once()
        self.search.assert_called_once()

    def test_context_change_error_stops_before_generation(self):
        self.context.side_effect = ValueError('원문 변경')
        with self.assertRaisesRegex(ValueError, '원문 변경'):
            self.graph.invoke(self.request)
        self.generate.assert_not_called()
        self.abstain.assert_not_called()

    def test_generation_error_is_not_retried_or_reported_as_no_evidence(self):
        self.generate.side_effect = RuntimeError('모델 연결 실패')
        with self.assertRaisesRegex(RuntimeError, '연결 실패'):
            self.graph.invoke(self.request)
        self.generate.assert_called_once()
        self.validate.assert_not_called()
        self.abstain.assert_not_called()

    def test_malformed_response_error_is_not_retried(self):
        self.validate.side_effect = ValueError('응답 형식 오류')
        with self.assertRaisesRegex(ValueError, '형식 오류'):
            self.graph.invoke(self.request)
        self.generate.assert_called_once()
        self.abstain.assert_not_called()

    def test_next_request_does_not_reuse_previous_sources_or_answer(self):
        self.assertEqual(self.graph.invoke(self.request)['result'], self.valid)
        self.search.return_value = []
        result = self.graph.invoke({'project_id': 2, 'question': '다른 질문'})
        self.assertEqual(result['result'], self.held)
        self.assertNotIn('sources', result)
        self.assertNotIn('response_content', result)
        self.search.assert_called_with(2, '다른 질문', minutes_id=None, top_k=3)
        self.generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
