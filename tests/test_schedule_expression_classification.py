import json
import unittest
from datetime import date, datetime
from unittest.mock import Mock, patch

from pydantic import ValidationError

from summarization import MeetingSummarizer


def response(source, day, clock):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': [{
        'eventId': 'review', 'status': 'confirmed', 'evidence': source,
        'dateExpression': day, 'timeExpression': clock,
        'extractedScheduleContent': '검토',
    }]}, ensure_ascii=False)


class ScheduleExpressionClassificationTest(unittest.TestCase):
    def test_clock_in_date_field_uses_reference_date_without_another_model_call(self):
        for hour in range(24):
            for suffix in ['', '까지', '부터는']:
                expression = f'{hour:02d}:23{suffix}'
                source = f'검토는 {expression} 확정합니다.'
                with self.subTest(expression=expression):
                    model = Mock(return_value=response(source, expression, None))
                    result = MeetingSummarizer(model, now=lambda: datetime(2028, 2, 29, 22)).summarize(source)
                    self.assertEqual(result['schedules'], [{
                        'extractedScheduleDate': f'2028-02-29T{hour:02d}:23:00',
                        'extractedScheduleContent': '검토',
                    }])
                    model.assert_called_once()

    def test_equivalent_expressions_are_compared_after_clock_calculation(self):
        for day, clock, expected in [
            ('2시 반까지', '오후 2시 30분', '14:30:00'),
            ('오전 12시', '00:00', '00:00:00'),
            ('8시부터', '08:00', '08:00:00'),
        ]:
            with self.subTest(day=day, clock=clock):
                source = f'검토는 {day}로 정했습니다. 즉 {clock}입니다.'
                model = Mock(return_value=response(source, day, clock))
                result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(
                    source, meeting_date=date(2027, 1, 3),
                )
                self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2027-01-03T' + expected)
                model.assert_called_once()

    def test_conflicting_clocks_request_repair_without_choosing_either(self):
        source = '검토는 오전 2시가 아니라 오후 2시에 확정합니다.'
        model = Mock(side_effect=[
            response(source, '오전 2시', '오후 2시'), response(source, None, '오후 2시'),
        ])
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        self.assertEqual(model.call_count, 2)
        self.assertIn('시각이 서로 달라', model.call_args.args[0][-1]['content'])
        self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2026-09-26T14:00:00')

    def test_conflict_cannot_bypass_the_shared_retry_limit(self):
        source = '검토는 03:00 또는 3시에 진행합니다.'
        model = Mock(return_value=response(source, '03:00', '3시'))
        with self.assertRaisesRegex(ValueError, '시각이 서로 달라'):
            MeetingSummarizer(model).summarize(source)
        self.assertEqual(model.call_count, 2)

    def test_ambiguous_or_invalid_date_field_is_not_reduced_to_a_clock(self):
        for expression in ['2시쯤', '2시 이후', '2시 또는 3시', '2시가 아니라',
                           '24시', '11시 60분', '조만간', '2027년 2월 29일 오후 2시',
                           '다음 주 중 오후 2시', '2시에 하기로']:
            with self.subTest(expression=expression):
                source = f'검토는 {expression} 결정했습니다.'
                model = Mock(return_value=response(source, expression, None))
                with self.assertRaises(ValueError):
                    MeetingSummarizer(model, repair_invalid_response=False).summarize(source)
                model.assert_called_once()

    def test_ambiguous_time_field_is_not_overwritten_by_a_clear_date_field_clock(self):
        source = '검토는 2시쯤 진행합니다.'
        model = Mock(return_value=response(source, '2시', '2시쯤'))
        with self.assertRaisesRegex(ValueError, '지원하지 않는 시각'):
            MeetingSummarizer(model, repair_invalid_response=False).summarize(source)
        model.assert_called_once()

    def test_original_fields_and_default_records_are_preserved(self):
        source = '검토는 2시 반까지 확정합니다.'
        model = Mock(return_value=response(source, '2시 반까지', None))
        result = MeetingSummarizer(model).workflow.invoke({
            'messages': [], 'input_text': source, 'meeting_date': '2026-09-26',
            'request_datetime': '2026-09-26T16:00:00+09:00',
        })
        schedule = result['result']['schedules'][0]
        self.assertEqual(schedule['dateExpression'], '2시 반까지')
        self.assertIsNone(schedule['timeExpression'])
        self.assertEqual(schedule['evidence'], source)
        self.assertEqual(schedule['extractedScheduleDate'], '2026-09-26T14:30:00')
        self.assertEqual(result['date_checks'][0]['policy'], 'reference_date')
        self.assertEqual(result['date_checks'][0]['reclassified_as'], 'timeExpression')
        self.assertEqual(result['time_checks'][0]['source_field'], 'dateExpression')
        self.assertEqual(result['time_checks'][0]['policy'], 'business_hours')
        self.assertTrue(result['time_checks'][0]['before_request'])

    def test_expression_from_another_evidence_is_still_rejected(self):
        quote = '검토를 확정합니다.'
        source = quote + ' 다른 작업은 2시 반까지입니다.'
        model = Mock(return_value=response(quote, '2시 반까지', None))
        with patch('summary_workflow.resolve_schedule_time') as tool:
            with self.assertRaises(ValidationError):
                MeetingSummarizer(model, repair_invalid_response=False).summarize(source)
        tool.invoke.assert_not_called()

    def test_unexpected_tool_errors_propagate_without_model_retry(self):
        source = '검토는 2시까지 확정합니다.'
        model = Mock(return_value=response(source, '2시', None))
        with patch('summary_workflow.resolve_schedule_time') as tool:
            tool.invoke.side_effect = RuntimeError('내부 계산 오류')
            with self.assertRaisesRegex(RuntimeError, '내부 계산 오류'):
                MeetingSummarizer(model).summarize(source)
        model.assert_called_once()


if __name__ == '__main__':
    unittest.main()
