import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError

from summary_task_review import SummaryTaskReviewer


class SummaryTaskReviewerTests(unittest.TestCase):
    source = '민수: 제가 기록을 확인할게요.\n지연: 다른 업무는 나중에 정하죠.'
    task = {'assignees': ['민수'], 'task': '기록을 확인한다.', 'evidence': {'start': 1, 'end': 1}}

    def client(self, reviews=None, *, content=None, finish='stop', refusal=None):
        if content is None:
            content = json.dumps({'reviews': reviews if reviews is not None else [
                {'taskIndex': 0, 'verdict': 'supported', 'reason': '민수가 확인을 약속했다.'},
            ]})
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal),
        )])
        client = Mock()
        client.chat.completions.create.return_value = response
        return client

    def test_empty_tasks_skip_model_call(self):
        client = self.client()
        self.assertEqual(SummaryTaskReviewer(client).review('원인 미확인 이슈가 있다.', []), [])
        client.chat.completions.create.assert_not_called()

    def test_reviews_batch_once_and_keeps_full_source_context(self):
        client = self.client(reviews=[
            {'taskIndex': 1, 'verdict': 'uncertain', 'reason': '담당 범위가 불명확하다.'},
            {'taskIndex': 0, 'verdict': 'supported', 'reason': '민수가 확인을 약속했다.'},
        ])
        result = SummaryTaskReviewer(client).review(self.source, [self.task, self.task])
        self.assertEqual([r.taskIndex for r in result], [0, 1])
        self.assertEqual(result[1].verdict, 'uncertain')
        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        payload = json.loads(request['messages'][1]['content'])
        self.assertEqual(payload['source'], self.source)
        self.assertEqual(payload['tasks'][0]['sourceEvidence'], '민수: 제가 기록을 확인할게요.\n')
        self.assertTrue(request['response_format']['json_schema']['strict'])

    def test_invalid_source_reference_is_rejected_before_api_call(self):
        client = self.client()
        with self.assertRaises(ValidationError):
            SummaryTaskReviewer(client).review(self.source, [{**self.task, 'assignees': ['수빈']}])
        client.chat.completions.create.assert_not_called()

    def test_missing_duplicate_or_unknown_indexes_are_rejected(self):
        for indexes in [[], [0, 0], [1], [-1]]:
            with self.subTest(indexes=indexes):
                client = self.client(reviews=[{'taskIndex': i, 'verdict': 'supported', 'reason': '검토 사유'} for i in indexes])
                with self.assertRaisesRegex(ValueError, '업무 번호'):
                    SummaryTaskReviewer(client).review(self.source, [self.task])
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_invalid_verdict_or_blank_reason_is_rejected(self):
        for verdict, reason in [('maybe', '검토 사유'), ('supported', ' ')]:
            client = self.client(reviews=[{'taskIndex': 0, 'verdict': verdict, 'reason': reason}])
            with self.subTest(verdict=verdict, reason=reason), self.assertRaises(ValidationError):
                SummaryTaskReviewer(client).review(self.source, [self.task])

    def test_failed_provider_response_is_not_accepted_or_retried(self):
        for finish, refusal, content in [('length', None, '{}'), ('stop', '거절', '{}'), ('stop', None, '')]:
            client = self.client(content=content, finish=finish, refusal=refusal)
            with self.subTest(finish=finish, refusal=refusal), self.assertRaises(ValueError):
                SummaryTaskReviewer(client).review(self.source, [self.task])
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_transport_error_propagates_without_local_retry(self):
        client = self.client()
        client.chat.completions.create.side_effect = TimeoutError('연결 시간 초과')
        with self.assertRaises(TimeoutError):
            SummaryTaskReviewer(client).review(self.source, [self.task])
        self.assertEqual(client.chat.completions.create.call_count, 1)


if __name__ == '__main__':
    unittest.main()
