import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError

from meeting_answers import MeetingQuestionAnswerer, OpenAIMeetingAnswerModel
from meeting_index import SearchHit


class MeetingAnswerTests(unittest.TestCase):
    def setUp(self):
        pieces = ['일반 자료는 공유합니다. ', '민수가 고객 명단을 공유하기로 했습니다. ',
                  '다만 고객 명단 공유는 취소했고 익명 통계만 공유하기로 했습니다.']
        self.source = ''.join(pieces)
        start = 0
        chunks = []
        for i, text in enumerate(pieces):
            chunks.append({'index': i, 'start': start, 'end': start + len(text), 'text': text})
            start += len(text)
        self.index = Mock(pipeline_version='v1')
        self.index.search.return_value = [SearchHit(1, 10, 1, pieces[1], chunks[1]['start'], chunks[1]['end'], .9, 'hash1')]
        self.index.get.return_value = {'source_hash': 'hash1', 'pipeline_version': 'v1',
                                       'source_text': self.source, 'chunks': chunks}
        self.quote = pieces[-1]
        self.complete = Mock(return_value=self.response())
        self.answerer = MeetingQuestionAnswerer(self.index, self.complete)

    def response(self, *, source_id='E1', quote=None):
        return json.dumps({'status': 'answered', 'claims': [{'text': '고객 명단 공유는 취소됐고 익명 통계만 공유합니다.',
                           'citations': [{'source_id': source_id, 'quote': self.quote if quote is None else quote}]}]}, ensure_ascii=False)

    def test_neighbor_exception_is_included_and_quote_offsets_match(self):
        result = self.answerer.answer(1, '고객 명단도 공유하나요?', minutes_id=10)
        self.index.search.assert_called_once_with(1, '고객 명단도 공유하나요?', minutes_id=10, top_k=3)
        self.complete.assert_called_once()
        context = json.loads(self.complete.call_args.args[0][1]['content'])['sources']
        self.assertEqual(context[0]['text'], self.source)
        citation = result['claims'][0]['citations'][0]
        self.assertEqual(self.source[citation['start']:citation['end']], self.quote)
        self.assertEqual(citation['minutes_id'], 10)
        self.assertEqual(citation['source_hash'], 'hash1')

    def test_overlapping_context_is_merged_without_duplicate_text(self):
        self.index.search.return_value *= 2
        self.answerer.answer(1, '질문')
        context = json.loads(self.complete.call_args.args[0][1]['content'])['sources']
        self.assertEqual(len(context), 1)
        self.index.get.assert_called_once()
        self.assertEqual(context[0]['text'], self.source)

    def test_no_search_results_skips_generation(self):
        self.index.search.return_value = []
        self.assertEqual(self.answerer.answer(1, '질문')['status'], 'insufficient_evidence')
        self.complete.assert_not_called()

    def test_context_over_budget_is_not_cut_mid_sentence(self):
        answerer = MeetingQuestionAnswerer(self.index, self.complete, max_context_chars=5)
        self.assertEqual(answerer.answer(1, '질문')['status'], 'insufficient_evidence')
        self.complete.assert_not_called()

    def test_model_can_abstain_even_with_search_results(self):
        self.complete.return_value = json.dumps({'status': 'insufficient_evidence', 'claims': []})
        result = self.answerer.answer(1, '비용은 얼마인가요?')
        self.assertEqual(result['claims'], [])
        self.assertEqual(result['status'], 'insufficient_evidence')

    def test_unknown_source_and_invented_quote_are_not_returned(self):
        for source_id, quote in [('OTHER', self.quote), ('E1', '공유 비용은 10만 원입니다.'), ('E1', ' ')]:
            with self.subTest(source_id=source_id, quote=quote):
                self.complete.return_value = self.response(source_id=source_id, quote=quote)
                with self.assertLogs('meeting_answers', level='WARNING'):
                    result = self.answerer.answer(1, '질문')
                self.assertEqual(result['status'], 'insufficient_evidence')
                self.assertEqual(result['claims'], [])
        self.assertEqual(self.complete.call_count, 3)

    def test_wrong_project_and_changed_source_fail_before_generation(self):
        for project_id, source_hash, version in [(2, 'hash1', 'v1'), (1, 'hash2', 'v1'), (1, 'hash1', 'v2')]:
            with self.subTest(project_id=project_id, version=version):
                self.index.get.return_value['source_hash'] = source_hash
                self.index.get.return_value['pipeline_version'] = version
                with self.assertRaises(ValueError):
                    self.answerer.answer(project_id, '질문')
        self.complete.assert_not_called()

    def test_uncited_claim_and_inconsistent_status_are_rejected_without_retry(self):
        for payload in [{'status': 'answered', 'claims': []},
                        {'status': 'answered', 'claims': [{'text': '답변', 'citations': []}]},
                        {'status': 'insufficient_evidence', 'claims': json.loads(self.response())['claims']}]:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                self.complete.return_value = json.dumps(payload)
                self.answerer.answer(1, '질문')
        self.assertEqual(self.complete.call_count, 3)

    def test_invalid_question_is_rejected_before_search(self):
        for question in ['', ' ', None, '가' * 2001]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                self.answerer.answer(1, question)
        self.index.search.assert_not_called()

    def test_model_error_is_not_retried(self):
        self.complete.side_effect = RuntimeError('모델 호출 실패')
        with self.assertRaises(RuntimeError):
            self.answerer.answer(1, '질문')
        self.complete.assert_called_once()


class MeetingAnswerModelTests(unittest.TestCase):
    def test_single_bounded_structured_request(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(refusal=None, content='{"status":"insufficient_evidence","claims":[]}'))])
        result = OpenAIMeetingAnswerModel(client)([])
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['max_completion_tokens'], 700)
        self.assertEqual(request['temperature'], 0)
        self.assertTrue(request['response_format']['json_schema']['strict'])
        self.assertEqual(json.loads(result)['status'], 'insufficient_evidence')
        client.chat.completions.create.assert_called_once()

    def test_invalid_provider_responses_are_rejected(self):
        client = Mock()
        for choices in [[], [SimpleNamespace(finish_reason='length', message=SimpleNamespace(refusal=None, content='{}'))],
                        [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(refusal='거절', content=None))],
                        [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(refusal=None, content=' '))]]:
            client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                OpenAIMeetingAnswerModel(client)([])


if __name__ == '__main__':
    unittest.main()
