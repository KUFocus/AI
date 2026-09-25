import json
import unittest
from unittest.mock import Mock

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

    def test_repairs_json_or_date_error_with_feedback_and_original_context(self):
        expected = {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleDate': '2026-09-28T10:00:00', 'extractedScheduleContent': '디자인 리뷰',
            'dateExpression': '다음 주 월요일',
        }]}
        invalid_date = {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleDate': '2026-02-30T10:00:00', 'extractedScheduleContent': '디자인 리뷰',
            'dateExpression': '다음 주 월요일',
        }]}
        messages = [{'role': 'system', 'content': '기준일: 2026-09-25'},
                    {'role': 'user', 'content': '다음 주 월요일 오전 10시에 디자인 리뷰를 한다.'}]
        for invalid, feedback in [
            ('잘못된 JSON', '올바른 JSON 형식'),
            (json.dumps(invalid_date), 'schedules.0.extractedScheduleDate: 일정 날짜 또는 시간이 유효하지 않습니다.'),
        ]:
            with self.subTest(invalid=invalid):
                generate = Mock(side_effect=[invalid, json.dumps(expected)])
                workflow = build_summary_workflow(generate, MeetingSummarizer.validate_response)
                with self.assertLogs('summary_workflow', level='WARNING'):
                    updates = list(workflow.stream({'messages': messages, 'input_text': messages[1]['content']}, stream_mode='updates'))

                self.assertEqual([next(iter(update)) for update in updates], ['generate', 'validate', 'repair', 'generate', 'validate'])
                self.assertEqual(updates[-1]['validate'], {'result': expected, 'validation_error': None})
                self.assertIsNone(updates[1]['validate']['result'])
                self.assertEqual(generate.call_count, 2)
                original = generate.call_args_list[0].args[0]
                repaired = generate.call_args_list[1].args[0]
                self.assertEqual(original, messages)
                self.assertEqual(len(original), 2)
                self.assertEqual(repaired[:2], original)
                self.assertEqual(repaired[2], {'role': 'assistant', 'content': invalid})
                self.assertIn(feedback, repaired[3]['content'])

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


if __name__ == '__main__':
    unittest.main()
