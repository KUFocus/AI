import json
import unittest
from datetime import date
from unittest.mock import Mock, patch

from pydantic import ValidationError

from summarization import MeetingSummarizer


SOURCE = '내일 오후 3시 반에 리뷰를 확정합니다.'


def schedule(**changes):
    return {
        'eventId': 'review', 'status': 'confirmed', 'evidence': SOURCE,
        'dateExpression': '내일', 'timeExpression': '오후 3시 반',
        'extractedScheduleDate': '2026-09-27T03:00:42.123', 'extractedScheduleContent': '리뷰',
        **changes,
    }


def response(*schedules):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': schedules}, ensure_ascii=False)


class ScheduleTimeWorkflowTest(unittest.TestCase):
    def test_date_and_time_use_original_expressions_without_extra_model_call(self):
        model = Mock(return_value=response(schedule()))
        result = MeetingSummarizer(model).summarize(SOURCE, meeting_date=date(2026, 9, 25))
        self.assertEqual(result, {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleDate': '2026-09-26T15:30:00', 'extractedScheduleContent': '리뷰',
        }]})
        model.assert_called_once()

    def test_time_expression_is_required_and_must_quote_source_and_evidence(self):
        source = SOURCE + ' 다음 회의는 오전 10시에 확정합니다.'
        missing = schedule()
        del missing['timeExpression']
        for value, message in [
            (missing, 'timeExpression'),
            (schedule(timeExpression=''), '시각 표현은 비어'),
            (schedule(timeExpression='  '), '시각 표현은 비어'),
            (schedule(timeExpression='16:00'), '시각 표현이 회의 원문에'),
            (schedule(timeExpression='오전 10시'), '시각 표현이 근거 발언에'),
        ]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, message):
                    MeetingSummarizer.validate_response(response(value), source)

    def test_ambiguous_time_is_not_replaced_by_model_clock(self):
        for expression, source in [
            ('오후 3시쯤', '내일 오후 3시쯤 리뷰를 확정합니다.'),
        ]:
            with self.subTest(expression=expression):
                model = Mock(return_value=response(schedule(timeExpression=expression, evidence=source)))
                with self.assertLogs('summary_workflow', level='WARNING'):
                    with self.assertRaisesRegex(ValueError, 'schedules.0.timeExpression'):
                        MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))
                self.assertEqual(model.call_count, 2)

    def test_time_failure_uses_remaining_shared_repair_budget(self):
        source = '내일 3시쯤 리뷰를 확정합니다.'
        bad = response(schedule(timeExpression='3시쯤', evidence=source))
        for outputs, repair, calls in [(['{}', bad], True, 2), ([bad], False, 1)]:
            with self.subTest(repair=repair):
                model = Mock(side_effect=outputs)
                with self.assertRaisesRegex(ValueError, '지원하지 않는 시각 표현'):
                    MeetingSummarizer(model, repair_invalid_response=repair).summarize(
                        source, meeting_date=date(2026, 9, 25),
                    )
                self.assertEqual(model.call_count, calls)

    def test_source_time_can_be_repaired_once_without_changing_original_context(self):
        model = Mock(side_effect=[response(schedule(timeExpression='오후 3시쯤')), response(schedule())])
        with self.assertLogs('summary_workflow', level='WARNING'):
            result = MeetingSummarizer(model).summarize(SOURCE, meeting_date=date(2026, 9, 25))
        self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2026-09-26T15:30:00')
        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.call_args_list[0].args[0][:2], model.call_args_list[1].args[0][:2])
        self.assertIn('시각 표현이 회의 원문에 그대로 존재하지 않습니다', model.call_args.args[0][-1]['content'])

    def test_one_unresolved_time_blocks_partial_response_and_records_original_index(self):
        second_source = '모레 4시쯤 발표를 확정합니다.'
        source = SOURCE + ' ' + second_source
        second = schedule(eventId='presentation', evidence=second_source, dateExpression='모레', timeExpression='4시쯤')
        cancelled = schedule(eventId='cancelled', status='cancelled', evidence='점검 취소',
                             dateExpression=None, timeExpression=None, extractedScheduleDate=None)
        source += ' 점검 취소'
        model = Mock(return_value=response(cancelled, schedule(), second))
        workflow = MeetingSummarizer(model).workflow
        updates = []
        with self.assertLogs('summary_workflow', level='WARNING'):
            with self.assertRaisesRegex(ValueError, 'schedules.2.timeExpression'):
                for update in workflow.stream({
                    'messages': [{'role': 'user', 'content': source}], 'input_text': source,
                    'meeting_date': '2026-09-25',
                }, stream_mode='updates'):
                    updates.append(update)
        failure = next(update['normalize_times'] for update in updates if 'normalize_times' in update)
        self.assertIsNone(failure['result'])
        self.assertEqual(failure['time_checks'], [
            {'schedule_index': 1, 'status': 'normalized', 'expected_time': '15:30:00', 'original_time': '03:00:42.123'},
            {'schedule_index': 2, 'status': 'unresolved'},
        ])
        self.assertEqual(model.call_count, 2)

    def test_cancelled_schedule_does_not_invoke_time_tool(self):
        model = Mock(return_value=response(schedule(status='cancelled', extractedScheduleDate=None,
                                                    dateExpression=None, timeExpression=None, evidence='리뷰 취소')))
        with patch('summary_workflow.resolve_schedule_time') as tool:
            result = MeetingSummarizer(model).summarize('리뷰 취소')
        self.assertEqual(result['schedules'], [])
        tool.invoke.assert_not_called()

    def test_unconfirmed_schedule_cannot_keep_time_expression(self):
        for status in ['tentative', 'cancelled']:
            with self.subTest(status=status):
                value = schedule(status=status, extractedScheduleDate=None, dateExpression=None)
                with self.assertRaisesRegex(ValidationError, '미확정 또는 취소 일정의 시각 표현은 null'):
                    MeetingSummarizer.validate_response(response(value), SOURCE)

    def test_unexpected_time_tool_failure_is_not_retried(self):
        model = Mock(return_value=response(schedule()))
        with patch('summary_workflow.resolve_schedule_time') as tool:
            tool.invoke.side_effect = ValueError('시각 계산 내부 오류')
            with self.assertRaisesRegex(ValueError, '시각 계산 내부 오류'):
                MeetingSummarizer(model).summarize(SOURCE, meeting_date=date(2026, 9, 25))
        model.assert_called_once()


if __name__ == '__main__':
    unittest.main()
