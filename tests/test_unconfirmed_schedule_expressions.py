import json
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from pydantic import ValidationError

from summarization import MeetingSummarizer


def decision(status, evidence, day=None, clock=None):
    return {'eventId': 'review', 'extractedScheduleContent': '검토', 'status': status,
            'evidence': evidence, 'dateExpression': day, 'timeExpression': clock}


def response(*decisions):
    return json.dumps({'summarizedText': '검토 일정을 논의했습니다.', 'schedules': decisions}, ensure_ascii=False)


class UnconfirmedScheduleExpressionsTest(unittest.TestCase):
    def test_proposal_with_quoted_date_and_time_returns_no_schedule_without_retry(self):
        source = '다음 주 화요일 오전 10시에 검토하면 어떨까요?'
        response = json.dumps({'summarizedText': '검토 일정을 제안했습니다.', 'schedules': [{
            'eventId': 'review', 'extractedScheduleContent': '검토',
            'dateExpression': '다음 주 화요일', 'timeExpression': '오전 10시',
            'status': 'tentative', 'evidence': source,
        }]}, ensure_ascii=False)
        model = Mock(return_value=response)
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        self.assertEqual(result['schedules'], [])
        model.assert_called_once()

    def test_dated_proposal_does_not_replace_an_existing_confirmation(self):
        confirmed = decision('confirmed', '내일 오전 10시 검토를 확정합니다.', '내일', '오전 10시')
        proposal = decision('tentative', '모레 오후 2시로 옮기면 어떨까요?', '모레', '오후 2시')
        model = Mock(return_value=response(confirmed, proposal))
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(
            confirmed['evidence'] + '\n' + proposal['evidence'])
        self.assertEqual(result['schedules'], [{'extractedScheduleContent': '검토',
                                                'extractedScheduleDate': '2026-09-27T10:00:00'}])
        model.assert_called_once()

    def test_dated_cancellation_removes_confirmation_before_date_calculation(self):
        confirmed = decision('confirmed', '내일 오전 10시 검토를 확정합니다.', '내일', '오전 10시')
        cancelled = decision('cancelled', '내일 오전 10시 검토는 취소합니다.', '내일', '오전 10시')
        proposal = decision('tentative', '모레 오후 2시에 다시 할까요?', '모레', '오후 2시')
        model = Mock(return_value=response(confirmed, cancelled, proposal))
        with patch('summary_workflow.resolve_schedule_date') as date_tool, patch('summary_workflow.resolve_schedule_time') as time_tool:
            result = MeetingSummarizer(model).summarize('\n'.join(d['evidence'] for d in [confirmed, cancelled, proposal]))
        self.assertEqual(result['schedules'], [])
        date_tool.invoke.assert_not_called()
        time_tool.invoke.assert_not_called()
        model.assert_called_once()

    def test_unconfirmed_ambiguous_expressions_are_preserved_without_becoming_a_schedule(self):
        for status in ['tentative', 'cancelled']:
            with self.subTest(status=status):
                source = '조만간 3시쯤 검토하는 안은 아직 보류 상태입니다.'
                raw = response(decision(status, source, '조만간', '3시쯤'))
                parsed = MeetingSummarizer.validate_response(raw, source)
                self.assertEqual(parsed['schedules'][0]['dateExpression'], '조만간')
                self.assertEqual(parsed['schedules'][0]['timeExpression'], '3시쯤')
                with patch('summary_workflow.resolve_schedule_date') as date_tool, patch('summary_workflow.resolve_schedule_time') as time_tool:
                    result = MeetingSummarizer(Mock(return_value=raw)).summarize(source)
                self.assertEqual(result['schedules'], [])
                date_tool.invoke.assert_not_called()
                time_tool.invoke.assert_not_called()

    def test_every_status_requires_expressions_to_belong_to_its_own_evidence(self):
        source = '내일 오전 10시 발표를 제안합니다. 검토는 취소합니다.'
        for status in ['confirmed', 'tentative', 'cancelled']:
            for day, clock, message in [('내일', None, '날짜 표현'), (None, '오전 10시', '시각 표현')]:
                with self.subTest(status=status, day=day, clock=clock):
                    raw = response(decision(status, '검토는 취소합니다.', day, clock))
                    with self.assertRaisesRegex(ValidationError, message + '이 근거 발언에 포함되어야 합니다'):
                        MeetingSummarizer.validate_response(raw, source)

    def test_invalid_proposal_still_fails_the_request_after_one_repair(self):
        source = '다음 주 화요일 오전 10시에 검토하면 어떨까요?'
        model = Mock(return_value=response(decision('tentative', '검토를 확정했습니다.', '다음 주 화요일', '오전 10시')))
        with self.assertRaisesRegex(ValidationError, '근거 발언이 회의 원문에 그대로 존재하지 않습니다'):
            MeetingSummarizer(model).summarize(source)
        self.assertEqual(model.call_count, 2)


if __name__ == '__main__':
    unittest.main()
