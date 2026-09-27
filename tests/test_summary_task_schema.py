import unittest

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


if __name__ == '__main__':
    unittest.main()
