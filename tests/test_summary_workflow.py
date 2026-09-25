import unittest
from unittest.mock import Mock

from summary_workflow import build_summary_workflow


class SummaryWorkflowTest(unittest.TestCase):
    def test_reused_workflow_keeps_each_request_separate(self):
        generate = Mock(side_effect=['첫 번째 응답', '두 번째 응답'])
        validate = Mock(side_effect=lambda content: {'summarizedText': content, 'schedules': []})
        workflow = build_summary_workflow(generate, validate)
        first_messages = [{'role': 'user', 'content': '첫 번째 회의'}]
        second_messages = [{'role': 'user', 'content': '두 번째 회의'}]

        first = workflow.invoke({'messages': first_messages})
        second = workflow.invoke({'messages': second_messages})

        self.assertEqual(first['result']['summarizedText'], '첫 번째 응답')
        self.assertEqual(second['result']['summarizedText'], '두 번째 응답')
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args_list[0].args[0], first_messages)
        self.assertEqual(generate.call_args_list[1].args[0], second_messages)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(validate.call_args_list[0].args[0], '첫 번째 응답')
        self.assertEqual(validate.call_args_list[1].args[0], '두 번째 응답')

    def test_generation_failure_stops_before_validation_without_retry(self):
        error = TimeoutError('모델 응답 시간이 초과되었습니다.')
        generate = Mock(side_effect=error)
        validate = Mock()
        workflow = build_summary_workflow(generate, validate)

        with self.assertRaises(TimeoutError) as caught:
            workflow.invoke({'messages': [{'role': 'user', 'content': '회의 내용'}]})

        self.assertIs(caught.exception, error)
        generate.assert_called_once()
        validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
