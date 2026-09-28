import json
import unittest
from datetime import date
from unittest.mock import Mock, patch

from pydantic import ValidationError

from schedule_history import ScheduleHistoryError
from summarization import MeetingSummarizer


def decision(status, quote, expression=None, event_id='review'):
    return {
        'eventId': event_id, 'status': status, 'timeExpression': '오전 10시' if status == 'confirmed' else None, 'evidence': quote,
        'dateExpression': expression,
        'extractedScheduleContent': '리뷰',
    }


def response(decisions):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': decisions}, ensure_ascii=False)


class ScheduleHistoryWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.first = decision('confirmed', '내일 오전 10시 리뷰를 확정합니다.', '내일')
        self.proposal = decision('tentative', '모레 오전 10시로 변경할까요?')
        self.cancel = decision('cancelled', '리뷰를 취소합니다.')
        self.changed = decision('confirmed', '모레 오전 10시 리뷰를 다시 확정합니다.', '모레')

    def test_transitions_reach_public_response_without_extra_generation(self):
        cases = [
            ([self.first, self.changed], '2026-09-27T10:00:00'),
            ([self.first, self.proposal], '2026-09-26T10:00:00'),
            ([self.first, self.cancel], None),
            ([self.first, self.cancel, self.proposal], None),
            ([self.first, self.cancel, self.changed], '2026-09-27T10:00:00'),
        ]
        for decisions, expected_date in cases:
            with self.subTest(decisions=decisions):
                model = Mock(return_value=response(list(reversed(decisions))))
                source = '\n'.join(item['evidence'] for item in decisions)
                result = MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))
                expected = [] if expected_date is None else [{
                    'extractedScheduleDate': expected_date, 'extractedScheduleContent': '리뷰',
                }]
                self.assertEqual(result, {'summarizedText': '회의 요약', 'schedules': expected})
                model.assert_called_once()

    def test_same_title_on_separate_events_does_not_merge(self):
        other = {**self.changed, 'eventId': 'second-review'}
        source = '\n'.join(item['evidence'] for item in [self.first, other, self.cancel])
        model = Mock(return_value=response([self.first, other, self.cancel]))
        result = MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))
        self.assertEqual(result['schedules'], [{
            'extractedScheduleDate': '2026-09-27T10:00:00', 'extractedScheduleContent': '리뷰',
        }])

    def test_cancelled_history_does_not_validate_obsolete_date(self):
        obsolete = decision('confirmed', '이번 주 일요일 오전 10시 리뷰를 확정합니다.', '이번 주 일요일')
        model = Mock(return_value=response([obsolete, self.cancel]))
        source = obsolete['evidence'] + '\n' + self.cancel['evidence']
        with patch('summary_workflow.resolve_schedule_date') as tool:
            result = MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))
        self.assertEqual(result['schedules'], [])
        tool.invoke.assert_not_called()

    def test_overlapping_history_is_repaired_through_graph(self):
        source = self.first['evidence'] + '\n' + self.changed['evidence']
        overlap = {**self.first, 'evidence': source}
        model = Mock(side_effect=[response([overlap, self.changed]), response([self.first, self.changed])])
        summarizer = MeetingSummarizer(model)
        with self.assertLogs('summary_workflow', level='WARNING'):
            updates = list(summarizer.workflow.stream({
                'messages': [{'role': 'user', 'content': source}],
                'input_text': source, 'meeting_date': '2026-09-25',
            }, stream_mode='updates'))
        self.assertEqual([next(iter(update)) for update in updates], [
            'generate', 'validate', 'resolve_histories', 'repair',
            'generate', 'validate', 'resolve_histories', 'normalize_dates', 'normalize_times', 'assemble_result',
        ])
        self.assertIsNone(updates[2]['resolve_histories']['result'])
        self.assertEqual(updates[-4]['resolve_histories']['schedules'][0]['dateExpression'], '모레')
        self.assertIn('일정 식별자 review', model.call_args.args[0][-1]['content'])
        self.assertEqual(model.call_count, 2)

    def test_history_errors_share_existing_retry_limit(self):
        source = self.first['evidence'] + '\n' + self.changed['evidence']
        bad = response([{**self.first, 'evidence': source}, self.changed])
        for responses, repair, expected_calls in [([bad, bad], True, 2), ([bad], False, 1)]:
            with self.subTest(repair=repair):
                model = Mock(side_effect=responses)
                with self.assertRaisesRegex(ScheduleHistoryError, '근거 구간이 겹쳐'):
                    MeetingSummarizer(model, repair_invalid_response=repair).summarize(
                        source, meeting_date=date(2026, 9, 25),
                    )
                self.assertEqual(model.call_count, expected_calls)

    def test_date_is_normalized_after_history_repair_without_another_retry(self):
        source = self.first['evidence'] + '\n' + self.changed['evidence']
        bad_history = response([{**self.first, 'evidence': source}, self.changed])
        changed = self.changed.copy()
        model = Mock(side_effect=[bad_history, response([self.first, changed])])
        with self.assertLogs('summary_workflow', level='WARNING'):
            result = MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))
        self.assertEqual(result['schedules'], [{
            'extractedScheduleDate': '2026-09-27T10:00:00', 'extractedScheduleContent': '리뷰',
        }])
        self.assertEqual(model.call_count, 2)

    def test_unsupported_date_after_history_repair_still_exhausts_budget(self):
        changed = decision('confirmed', '이번 주 일요일 오전 10시로 다시 확정합니다.', '이번 주 일요일')
        source = self.first['evidence'] + '\n' + changed['evidence']
        overlap = {**self.first, 'evidence': source}
        model = Mock(side_effect=[response([overlap, changed]), response([self.first, changed])])
        with self.assertLogs('summary_workflow', level='WARNING'):
            with self.assertRaisesRegex(ValueError, 'schedules.1.dateExpression: 현재 지원하지 않는 날짜 표현'):
                MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))
        self.assertEqual(model.call_count, 2)

    def test_missing_or_blank_event_id_is_rejected(self):
        for fields in [{}, {'eventId': ''}, {'eventId': '   '}]:
            with self.subTest(fields=fields):
                value = {key: item for key, item in self.first.items() if key != 'eventId'}
                value.update(fields)
                with self.assertRaises(ValidationError) as caught:
                    MeetingSummarizer.validate_response(response([value]), self.first['evidence'])
                self.assertEqual(caught.exception.errors()[0]['loc'], ('schedules', 0, 'eventId'))

    def test_unexpected_history_failure_is_not_retried(self):
        model = Mock(return_value=response([self.first]))
        with patch('summary_workflow.resolve_schedule_history', side_effect=ValueError('내부 처리 오류')):
            with self.assertRaisesRegex(ValueError, '내부 처리 오류'):
                MeetingSummarizer(model).summarize(self.first['evidence'], meeting_date=date(2026, 9, 25))
        model.assert_called_once()


if __name__ == '__main__':
    unittest.main()
