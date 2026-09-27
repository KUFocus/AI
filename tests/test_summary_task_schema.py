import unittest
import json
from unittest.mock import Mock

from summarization import MeetingSummarizer
from summary_workflow import build_summary_workflow
from summary_model import model_response_schema

from pydantic import ValidationError

from summary_task_schema import SummaryTask


class SummaryTaskTests(unittest.TestCase):
    def parse(self, payload, source):
        return SummaryTask.model_validate(payload, strict=True, context={'input_text': source})

    def payload(self, **overrides):
        return {
            'assignees': ['민수'],
            'task': '응답과 저장 상태가 다르면 요청 순서를 기록한다.',
            'evidence': {'start': 1, 'end': 1},
            **overrides,
        }

    def test_preserves_source_range_and_task_conditions(self):
        source = '민수: 응답과 저장 상태가 다르면 요청 순서를 기록할게요.\n'
        result = self.parse(self.payload(), source)
        self.assertEqual(result.evidence.resolve(source), source)
        self.assertEqual(result.model_dump(), self.payload())

    def test_joint_assignment_can_include_multiple_turns(self):
        source = '민수: 지연 님과 같이 결과를 대조할까요?\n지연: 네, 같이 하겠습니다.'
        result = self.parse(self.payload(
            assignees=['민수', '지연'], task='결과를 함께 대조한다.',
            evidence={'start': 1, 'end': 2},
        ), source)
        self.assertEqual(result.assignees, ['민수', '지연'])
        self.assertEqual(result.evidence.resolve(source), source)

    def test_missing_assignee_information_is_an_empty_list(self):
        result = self.parse(self.payload(assignees=[]), '응답이 다르면 요청 순서를 기록해주세요.')
        self.assertEqual(result.assignees, [])
        self.assertNotIn('status', result.model_dump())

    def test_assignee_in_another_turn_does_not_support_selected_range(self):
        source = '민수: 다음 안건으로 넘어가겠습니다.\n지연: 제가 요청 순서를 기록할게요.'
        with self.assertRaisesRegex(ValidationError, '선택한 업무 근거 구간'):
            self.parse(self.payload(evidence={'start': 2, 'end': 2}), source)

    def test_later_sentence_requires_the_speaker_segment_too(self):
        source = '민수: 아직 원인을 모릅니다. 서버 기록을 확인할게요.'
        with self.assertRaisesRegex(ValidationError, '앞선 구간부터 업무 발언까지'):
            self.parse(self.payload(evidence={'start': 2, 'end': 2}), source)
        result = self.parse(self.payload(evidence={'start': 1, 'end': 2}), source)
        self.assertEqual(result.evidence.resolve(source), source)

    def test_source_context_is_required(self):
        with self.assertRaisesRegex(ValidationError, '회의 원문이 없습니다'):
            SummaryTask.model_validate(self.payload(), strict=True)

    def test_invalid_ranges_are_rejected(self):
        for evidence in [
            {'start': 0, 'end': 1}, {'start': 2, 'end': 1},
            {'start': 1, 'end': 3}, {'start': '1', 'end': 1},
            {'start': True, 'end': 1},
        ]:
            with self.subTest(evidence=evidence), self.assertRaises(ValidationError):
                self.parse(self.payload(evidence=evidence), '민수: 기록할게요.')

    def test_invalid_task_and_assignee_values_are_rejected(self):
        for values in [
            {'task': '  '}, {'assignees': [' ']}, {'assignees': ['민수', ' 민수 ']},
            {'assignees': '민수'}, {'assignees': [1]}, {'assignees': None},
            {'status': 'confirmed'},
        ]:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                self.parse(self.payload(**values), '민수: 기록할게요.')

    def test_empty_source_cannot_supply_evidence(self):
        for source in ['', '   ']:
            with self.subTest(source=source), self.assertRaises(ValidationError):
                self.parse(self.payload(assignees=[]), source)


class SummaryTaskIntegrationTests(unittest.TestCase):
    def test_issue_only_summary_does_not_require_assignees_or_tasks(self):
        payload = {'summaryTasks': [], 'summarizedText': '결제 지연 문제가 보고됐으며 원인은 미확인이다.', 'schedules': []}
        complete = Mock(return_value=json.dumps(payload, ensure_ascii=False))
        result = MeetingSummarizer(complete).summarize('결제 지연 문제가 보고됐습니다. 원인은 아직 모릅니다.')
        self.assertEqual(result, {'summarizedText': payload['summarizedText'], 'schedules': []})
        self.assertEqual(complete.call_count, 1)

    def test_tasks_remain_internal_while_summary_preserves_unassigned_issue(self):
        source = '결제 지연 원인은 아직 모릅니다.\n민수: 제가 서버 기록을 확인할게요.'
        task = {'assignees': ['민수'], 'task': '서버 기록을 확인한다.', 'evidence': {'start': 2, 'end': 2}}
        payload = {'summaryTasks': [task], 'summarizedText': '결제 지연 원인은 미확인이고 민수가 서버 기록을 확인한다.', 'schedules': []}
        complete = Mock(return_value=json.dumps(payload, ensure_ascii=False))
        workflow = build_summary_workflow(complete, MeetingSummarizer.validate_response)
        state = workflow.invoke({'messages': [], 'input_text': source})
        self.assertEqual(state['summary_tasks'], [task])
        self.assertEqual(state['result'], {'summarizedText': payload['summarizedText'], 'schedules': []})
        self.assertEqual(complete.call_count, 1)
        complete.return_value = json.dumps({'summaryTasks': [], 'summarizedText': '원인은 미확인이다.', 'schedules': []})
        state = workflow.invoke({'messages': [], 'input_text': '원인은 아직 모릅니다.'})
        self.assertEqual(state['summary_tasks'], [])

    def test_invalid_task_uses_existing_bounded_repair(self):
        source = '민수: 제가 서버 기록을 확인할게요.'
        base = {'summarizedText': '민수가 서버 기록을 확인한다.', 'schedules': []}
        bad = {**base, 'summaryTasks': [{'assignees': ['지연'], 'task': '서버 기록을 확인한다.', 'evidence': {'start': 1, 'end': 1}}]}
        good = {**base, 'summaryTasks': [{**bad['summaryTasks'][0], 'assignees': ['민수']}]}
        complete = Mock(side_effect=[json.dumps(bad), json.dumps(good)])
        with self.assertLogs('summary_workflow', level='WARNING'):
            result = MeetingSummarizer(complete).summarize(source)
        self.assertEqual(result, base)
        self.assertEqual(complete.call_count, 2)
        feedback = complete.call_args_list[1].args[0][-1]['content']
        self.assertIn('summaryTasks.0', feedback)
        self.assertIn('담당자 이름이 선택한 업무 근거 구간에 존재하지 않습니다.', feedback)

    def test_repeated_invalid_task_does_not_start_unbounded_retries(self):
        payload = {'summaryTasks': [{'assignees': ['민수'], 'task': '확인한다.', 'evidence': {'start': 9, 'end': 9}}], 'summarizedText': '이슈가 보고됐다.', 'schedules': []}
        complete = Mock(return_value=json.dumps(payload))
        with self.assertLogs('summary_workflow', level='WARNING'), self.assertRaises(ValidationError):
            MeetingSummarizer(complete).summarize('이슈가 보고됐습니다.')
        self.assertEqual(complete.call_count, 2)

    def test_new_model_schema_requires_tasks_but_accepts_empty_lists(self):
        schema = model_response_schema()
        self.assertEqual(list(schema['properties']), ['summaryTasks', 'summarizedText', 'schedules'])
        self.assertIn('summaryTasks', schema['required'])
        self.assertNotIn('minItems', schema['properties']['summaryTasks'])
        task = schema['$defs']['SummaryTask']
        self.assertEqual(list(task['properties']), ['evidence', 'assignees', 'task'])
        self.assertEqual(set(task['required']), set(task['properties']))
        self.assertFalse(task['additionalProperties'])

    def test_preexisting_response_without_task_metadata_remains_readable(self):
        payload = {'summarizedText': '원인이 밝혀지지 않았다.', 'schedules': []}
        self.assertEqual(MeetingSummarizer.validate_response(json.dumps(payload), '원인이 밝혀지지 않았다.'), payload)


if __name__ == '__main__':
    unittest.main()
