import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from summarization import MeetingSummarizer
from summary_workflow import build_summary_workflow


class SummaryWorkflowTest(unittest.TestCase):
    def test_reused_workflow_keeps_each_request_separate(self):
        generate = Mock(side_effect=['첫 번째 응답', '두 번째 응답'])
        validate = Mock(side_effect=lambda content, input_text: {'summarizedText': content, 'schedules': []})
        workflow = build_summary_workflow(generate, validate)
        first_messages = [{'role': 'user', 'content': '첫 번째 회의'}]
        second_messages = [{'role': 'user', 'content': '두 번째 회의'}]

        first = workflow.invoke({'messages': first_messages, 'input_text': '첫 번째 회의'})
        second = workflow.invoke({'messages': second_messages, 'input_text': '두 번째 회의'})

        self.assertEqual(first['result']['summarizedText'], '첫 번째 응답')
        self.assertEqual(second['result']['summarizedText'], '두 번째 응답')
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args_list[0].args[0], first_messages)
        self.assertEqual(generate.call_args_list[1].args[0], second_messages)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(validate.call_args_list[0].args[0], '첫 번째 응답')
        self.assertEqual(validate.call_args_list[1].args[0], '두 번째 응답')
        self.assertEqual(first['attempts'], 1)
        self.assertEqual(second['attempts'], 1)

    def test_generation_failure_stops_before_validation_without_retry(self):
        for error in [TimeoutError('모델 응답 시간이 초과되었습니다.'), ValueError('모델 응답 거절 또는 출력 중단')]:
            with self.subTest(error=error):
                generate = Mock(side_effect=error)
                validate = Mock()
                workflow = build_summary_workflow(generate, validate)

                with self.assertRaises(type(error)) as caught:
                    workflow.invoke({'messages': [{'role': 'user', 'content': '회의 내용'}], 'input_text': '회의 내용'})

                self.assertIs(caught.exception, error)
                generate.assert_called_once()
                validate.assert_not_called()

    def test_repairs_json_or_expression_error_with_feedback_and_original_context(self):
        expected = {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleContent': '디자인 리뷰',
            'dateExpression': '다음 주 월요일',
            'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': '오전 10시', 'evidence': '다음 주 월요일 오전 10시',
        }]}
        invalid_expression = {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleContent': '디자인 리뷰',
            'dateExpression': '없는 날짜 표현',
            'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': '오전 10시', 'evidence': '다음 주 월요일 오전 10시',
        }]}
        messages = [{'role': 'system', 'content': '기준일: 2026-09-25'},
                    {'role': 'user', 'content': '다음 주 월요일 오전 10시에 디자인 리뷰를 한다.'}]
        for invalid, feedback in [
            ('잘못된 JSON', '올바른 JSON 형식'),
            (json.dumps(invalid_expression), 'schedules.0.dateExpression: 날짜 표현이 회의 원문에 그대로 존재하지 않습니다.'),
        ]:
            with self.subTest(invalid=invalid):
                generate = Mock(side_effect=[invalid, json.dumps(expected)])
                workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
                with self.assertLogs('summary_workflow', level='WARNING'):
                    updates = list(workflow.stream({
                        'messages': messages, 'input_text': messages[1]['content'], 'meeting_date': '2026-09-25',
                    }, stream_mode='updates'))

                self.assertEqual([next(iter(update)) for update in updates], ['generate', 'validate', 'repair', 'generate', 'validate', 'resolve_histories', 'normalize_dates', 'normalize_times', 'assemble_result'])
                self.assertEqual(updates[-5]['validate'], {
                    'summarized_text': expected['summarizedText'], 'schedules': expected['schedules'],
                    'result': None, 'validation_error': None, 'evidence_repair_required': False,
                })
                self.assertIsNone(updates[1]['validate']['result'])
                self.assertEqual(generate.call_count, 2)
                original = generate.call_args_list[0].args[0]
                repaired = generate.call_args_list[1].args[0]
                self.assertEqual(original, messages)
                self.assertEqual(len(original), 2)
                self.assertEqual(repaired[:2], original)
                self.assertEqual(repaired[-2], {'role': 'assistant', 'content': invalid})
                self.assertIn(feedback, repaired[-1]['content'])

    def test_stops_after_second_invalid_response(self):
        for content, error_type in [('{}', ValidationError), ('잘못된 JSON', json.JSONDecodeError)]:
            with self.subTest(content=content):
                generate = Mock(return_value=content)
                workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
                with self.assertLogs('summary_workflow', level='WARNING'):
                    with self.assertRaises(error_type):
                        workflow.invoke({'messages': [{'role': 'user', 'content': '회의 내용'}], 'input_text': '회의 내용'})
                self.assertEqual(generate.call_count, 2)

    def test_repair_can_be_disabled_for_single_call_evaluation(self):
        generate = Mock(return_value='{}')
        workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response, repair_invalid_response=False)
        with self.assertRaises(ValidationError):
            workflow.invoke({'messages': [{'role': 'user', 'content': '회의 내용'}], 'input_text': '회의 내용'})
        generate.assert_called_once()

    def test_failed_request_does_not_leak_state_into_next_request(self):
        expected = {'summarizedText': '두 번째 회의 요약', 'schedules': []}
        generate = Mock(side_effect=['{}', '{}', '{}', json.dumps(expected)])
        workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
        with self.assertLogs('summary_workflow', level='WARNING'):
            with self.assertRaises(ValidationError):
                workflow.invoke({'messages': [{'role': 'user', 'content': '첫 번째 회의'}], 'input_text': '첫 번째 회의'})
            second = workflow.invoke({'messages': [{'role': 'user', 'content': '두 번째 회의'}], 'input_text': '두 번째 회의'})

        self.assertEqual(second['result'], expected)
        self.assertEqual(second['attempts'], 2)
        self.assertIsNone(second['validation_error'])
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(generate.call_args_list[2].args[0], [{'role': 'user', 'content': '두 번째 회의'}])

    def test_unexpected_validation_error_is_not_repaired(self):
        error = TypeError('검증 함수 내부 오류')
        generate = Mock(return_value='{}')
        workflow = build_summary_workflow(generate, Mock(side_effect=error))
        with self.assertRaises(TypeError) as caught:
            workflow.invoke({'messages': [{'role': 'user', 'content': '회의 내용'}], 'input_text': '회의 내용'})
        self.assertIs(caught.exception, error)
        generate.assert_called_once()

    def test_final_datetime_is_created_only_after_date_and_time_resolution(self):
        response = json.dumps({'summarizedText': '회의 요약', 'schedules': [{
            'dateExpression': '다음 주 월요일 오후 3시',
            'extractedScheduleContent': '디자인 리뷰',
            'status': 'confirmed', 'eventId': '리뷰', 'timeExpression': '오후 3시', 'evidence': '다음 주 월요일 오후 3시',
        }]})
        generate = Mock(return_value=response)
        workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
        original = '다음 주 월요일 오후 3시에 디자인 리뷰를 한다.'
        updates = list(workflow.stream({
            'messages': [{'role': 'user', 'content': original}],
            'input_text': original, 'meeting_date': '2026-09-25',
        }, stream_mode='updates'))

        self.assertEqual([next(iter(update)) for update in updates], [
            'generate', 'validate', 'resolve_histories', 'normalize_dates', 'normalize_times', 'assemble_result',
        ])
        self.assertEqual(updates[-3]['normalize_dates']['date_checks'], [{
            'schedule_index': 0, 'status': 'resolved', 'expected_date': '2026-09-28',
        }])
        selected = updates[-4]['resolve_histories']['schedules'][0]
        dated = updates[-3]['normalize_dates']['schedules'][0]
        resolved = updates[-2]['normalize_times']['schedules'][0]
        self.assertNotIn('extractedScheduleDate', selected)
        self.assertNotIn('resolvedDate', selected)
        self.assertEqual(dated['resolvedDate'], '2026-09-28')
        self.assertNotIn('extractedScheduleDate', dated)
        self.assertEqual(resolved['extractedScheduleDate'], '2026-09-28T15:00:00')
        self.assertEqual(updates[-1]['assemble_result']['result'], {
            'summarizedText': '회의 요약', 'schedules': [resolved],
        })
        for update in updates[2:-1]:
            node_update = next(iter(update.values()))
            self.assertNotIn('summarized_text', node_update)
            self.assertNotIn('result', node_update)
        generate.assert_called_once()

    def test_date_normalization_does_not_require_remaining_repair_budget(self):
        response = json.dumps({'summarizedText': '회의 요약', 'schedules': [{
            'dateExpression': '내일',
            'extractedScheduleContent': '자료 제출',
            'status': 'confirmed', 'eventId': '자료 제출', 'timeExpression': '오전 10시', 'evidence': '내일 오전 10시',
        }]})
        for responses, repair_enabled, calls in [
            ([response], True, 1), (['{}', response], True, 2), ([response], False, 1),
        ]:
            with self.subTest(responses=responses, repair_enabled=repair_enabled):
                generate = Mock(side_effect=responses)
                workflow = build_summary_workflow(
                    generate, MeetingSummarizer.validate_response, repair_invalid_response=repair_enabled,
                )
                result = workflow.invoke({
                    'messages': [{'role': 'user', 'content': '내일 오전 10시 자료 제출'}],
                    'input_text': '내일 오전 10시 자료 제출', 'meeting_date': '2026-09-25',
                })
                self.assertEqual(result['result']['schedules'][0]['extractedScheduleDate'], '2026-09-26T10:00:00')
                self.assertEqual(generate.call_count, calls)

    def test_unresolved_date_blocks_partial_success_and_uses_one_repair(self):
        schedules = [{
            'dateExpression': expression,
            'extractedScheduleContent': content,
            'status': 'confirmed', 'eventId': content, 'timeExpression': '오전 10시', 'evidence': f'{expression} 오전 10시',
        } for expression, content in [
            ('내일', '자료 제출'),
            ('2026년 9월 28일', '리뷰'),
            ('이번 주 일요일', '점검'),
        ]]
        payload = {'summarizedText': '회의 요약', 'schedules': schedules}
        generate = Mock(return_value=json.dumps(payload))
        workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
        original = '내일 오전 10시 자료 제출, 2026년 9월 28일 오전 10시 리뷰, 이번 주 일요일 오전 10시 점검.'
        updates = []
        with self.assertLogs('summary_workflow', level='WARNING'):
            with self.assertRaisesRegex(ValueError, 'schedules.2.dateExpression: 현재 지원하지 않는 날짜 표현'):
                for update in workflow.stream({
                    'messages': [{'role': 'user', 'content': original}],
                    'input_text': original, 'meeting_date': '2026-09-25',
                }, stream_mode='updates'):
                    updates.append(update)
        self.assertIsNone(updates[3]['normalize_dates']['result'])
        self.assertEqual(updates[3]['normalize_dates']['date_checks'], [
            {'schedule_index': 0, 'status': 'resolved', 'expected_date': '2026-09-26'},
            {'schedule_index': 1, 'status': 'resolved', 'expected_date': '2026-09-28'},
            {'schedule_index': 2, 'status': 'unresolved'},
        ])
        self.assertEqual(generate.call_count, 2)

    def test_explicit_date_is_computed_when_repair_is_disabled(self):
        payload = {'summarizedText': '회의 요약', 'schedules': [{
            'dateExpression': '2026년 9월 28일 오전 10시',
            'extractedScheduleContent': '리뷰',
            'status': 'confirmed', 'eventId': '리뷰', 'timeExpression': '오전 10시', 'evidence': '2026년 9월 28일 오전 10시',
        }]}
        generate = Mock(return_value=json.dumps(payload))
        workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response, repair_invalid_response=False)
        original = '2026년 9월 28일 오전 10시에 리뷰를 한다.'
        result = workflow.invoke({
            'messages': [{'role': 'user', 'content': original}],
            'input_text': original, 'meeting_date': '2026-09-25',
        })
        self.assertEqual(result['result']['schedules'][0]['extractedScheduleDate'], '2026-09-28T10:00:00')
        generate.assert_called_once()

    def test_tool_failure_is_not_treated_as_unsupported_or_repaired(self):
        payload = {'summarizedText': '회의 요약', 'schedules': [{
            'dateExpression': '내일',
            'extractedScheduleContent': '제출',
            'status': 'confirmed', 'eventId': '제출', 'timeExpression': '오전 10시', 'evidence': '내일 오전 10시',
        }]}
        generate = Mock(return_value=json.dumps(payload))
        workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
        with patch('summary_workflow.resolve_schedule_date') as date_tool:
            date_tool.invoke.side_effect = ValueError('계산 오류')
            with self.assertRaisesRegex(ValueError, '계산 오류'):
                workflow.invoke({
                    'messages': [{'role': 'user', 'content': '내일 오전 10시 제출'}],
                    'input_text': '내일 오전 10시 제출', 'meeting_date': '2026-09-25',
                })
        generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
