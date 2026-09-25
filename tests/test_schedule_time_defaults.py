import json
import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

from summarization import MeetingSummarizer
from schedule_tools import resolve_schedule_time, ScheduleTimeExpressionError


def decision(source, day=None, clock=None, **changes):
    return {
        'eventId': 'submission', 'status': 'confirmed', 'evidence': source,
        'dateExpression': day, 'timeExpression': clock,
        'extractedScheduleDate': None, 'extractedScheduleContent': '제출 마감',
        **changes,
    }


def response(*schedules):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': schedules}, ensure_ascii=False)


class ScheduleTimeDefaultsTest(unittest.TestCase):
    def test_date_only_deadlines_use_18_and_ignore_model_invented_clock(self):
        for source, day, expected in [
            ('내일까지 제출해주세요.', '내일', '2026-09-26'),
            ('다음 주 화요일까지 완료해주세요.', '다음 주 화요일', '2026-09-29'),
        ]:
            for invented in [None, expected, expected + 'T23:59:00']:
                with self.subTest(source=source, invented=invented):
                    model = Mock(return_value=response(decision(source, day, extractedScheduleDate=invented)))
                    with patch('summary_workflow.resolve_schedule_time') as tool:
                        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 25, 10)).summarize(source)
                    self.assertEqual(result['schedules'], [{
                        'extractedScheduleDate': expected + 'T18:00:00', 'extractedScheduleContent': '제출 마감',
                    }])
                    tool.invoke.assert_not_called()
                    model.assert_called_once()

    def test_time_only_uses_reference_date_and_keeps_explicit_period(self):
        for expression, expected in [('3시', '15:00:00'), ('8시', '08:00:00'),
                                     ('오전 3시', '03:00:00'), ('오후 8시', '20:00:00')]:
            with self.subTest(expression=expression):
                source = expression + '부터 시작해주세요.'
                model = Mock(return_value=response(decision(source, clock=expression)))
                summarizer = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 25, 10))
                with patch('summary_workflow.resolve_schedule_date') as tool:
                    result = summarizer.summarize(source)
                self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2026-09-25T' + expected)
                tool.invoke.assert_not_called()
                model.assert_called_once()

    def test_business_hour_boundaries_and_explicit_clock_override_defaults(self):
        examples = [('03시', '15:00:00'), ('1시', '13:00:00'), ('7시 반', '19:30:00'), ('8시', '08:00:00'),
                    ('11시 30분', '11:30:00'), ('12시', '12:00:00'), ('12시 반', '12:30:00'),
                    ('0시', '00:00:00'), ('23시', '23:00:00'), ('03:00', '03:00:00'),
                    ('오전 12시', '00:00:00'), ('오후 12시', '12:00:00')]
        for expression, expected in examples:
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_time.invoke({
                    'expression': expression, 'assume_business_hours': True,
                }), expected)
        for expression in ['3시쯤', '3시 또는 4시', '8시 이후', '24시', '12시 60분']:
            with self.subTest(expression=expression):
                with self.assertRaises(ScheduleTimeExpressionError):
                    resolve_schedule_time.invoke({'expression': expression, 'assume_business_hours': True})

    def test_missing_date_and_time_is_omitted_even_if_model_invents_timestamp(self):
        source = '보고서를 제출해주세요.'
        model = Mock(return_value=response(decision(source, extractedScheduleDate='2026-09-26T12:00:00')))
        with patch('summary_workflow.resolve_schedule_date') as date_tool, patch('summary_workflow.resolve_schedule_time') as time_tool:
            result = MeetingSummarizer(model).summarize(source)
        self.assertEqual(result['schedules'], [])
        date_tool.invoke.assert_not_called()
        time_tool.invoke.assert_not_called()
        model.assert_called_once()

    def test_vague_date_or_time_does_not_trigger_missing_value_defaults(self):
        for source, day, clock in [('조만간 제출해주세요.', '조만간', None),
                                   ('내일 3시쯤 시작해주세요.', '내일', '3시쯤')]:
            with self.subTest(source=source):
                model = Mock(return_value=response(decision(source, day, clock)))
                with self.assertRaisesRegex(ValueError, '지원하지 않는'):
                    MeetingSummarizer(model, repair_invalid_response=False).summarize(source)
                model.assert_called_once()

    def test_midnight_during_repair_does_not_change_request_reference(self):
        source = '3시까지 제출해주세요.'
        now = Mock(side_effect=[datetime(2026, 9, 25, 23, 59), datetime(2026, 9, 26, 0, 1)])
        good = response(decision(source, clock='3시'))
        model = Mock(side_effect=['{}', good, good])
        summarizer = MeetingSummarizer(model, now=now)
        with self.assertLogs('summary_workflow', level='WARNING'):
            first = summarizer.summarize(source)
        self.assertEqual(first['schedules'][0]['extractedScheduleDate'], '2026-09-25T15:00:00')
        now.assert_called_once()
        self.assertEqual(model.call_args_list[0].args[0][:2], model.call_args_list[1].args[0][:2])
        second = summarizer.summarize(source)
        self.assertEqual(second['schedules'][0]['extractedScheduleDate'], '2026-09-26T15:00:00')
        self.assertEqual(now.call_count, 2)

    def test_processing_timestamp_is_converted_to_korea_and_meeting_date_takes_precedence(self):
        source = '8시부터 시작해주세요.'
        model = Mock(return_value=response(decision(source, clock='8시')))
        summarizer = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 25, 16, tzinfo=timezone.utc))
        current = summarizer.summarize(source)
        supplied = summarizer.summarize(source, meeting_date=date(2026, 9, 24))
        self.assertEqual(current['schedules'][0]['extractedScheduleDate'], '2026-09-26T08:00:00')
        self.assertEqual(supplied['schedules'][0]['extractedScheduleDate'], '2026-09-24T08:00:00')

    def test_internal_checks_distinguish_defaults_and_record_past_times_without_rollover(self):
        source = '3시까지 제출해주세요. 내일까지 검토해주세요. 오늘 오후 5시에 공유해주세요.'
        model = Mock(return_value=response(
            decision('3시까지 제출해주세요.', clock='3시'),
            decision('내일까지 검토해주세요.', day='내일', eventId='review'),
            decision('오늘 오후 5시에 공유해주세요.', day='오늘', clock='오후 5시', eventId='share'),
        ))
        result = MeetingSummarizer(model).workflow.invoke({
            'messages': [], 'input_text': source, 'meeting_date': '2026-09-25',
            'request_datetime': '2026-09-25T16:00:00+09:00',
        })
        self.assertEqual(result['date_checks'][0]['policy'], 'reference_date')
        self.assertEqual(result['time_checks'][0]['policy'], 'business_hours')
        self.assertTrue(result['time_checks'][0]['before_request'])
        self.assertEqual(result['result']['schedules'][0]['extractedScheduleDate'], '2026-09-25T15:00:00')
        self.assertEqual(result['time_checks'][1]['policy'], 'missing_time_18')
        self.assertFalse(result['time_checks'][1]['before_request'])
        self.assertNotIn('policy', result['time_checks'][2])

    def test_cancellation_removes_defaulted_candidate_before_temporal_tools(self):
        first, last = '내일까지 제출해주세요.', '제출 일정을 취소합니다.'
        model = Mock(return_value=response(decision(first, day='내일'), decision(last, status='cancelled')))
        with patch('summary_workflow.resolve_schedule_date') as tool:
            result = MeetingSummarizer(model).summarize(first + ' ' + last)
        self.assertEqual(result['schedules'], [])
        tool.invoke.assert_not_called()


if __name__ == '__main__':
    unittest.main()
