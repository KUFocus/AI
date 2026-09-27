import json
import unittest
from datetime import datetime
from pydantic import ValidationError
from summarization import MeetingSummarizer
from summary_schema import SummaryResponse


class TemporalReferenceTest(unittest.TestCase):
    def candidate(self, **changes):
        candidate = {
            'eventId': 'review', 'extractedScheduleContent': '검토', 'status': 'confirmed',
            'dateExpression': '모레', 'timeExpression': '오후 2시',
            'evidence': {'start': 3, 'end': 3},
            'dateReference': {'expression': '같은 날', 'source': {'start': 1, 'end': 1}},
            'timeReference': {'expression': '앞서 정한 시각', 'source': {'start': 2, 'end': 2}},
        }
        candidate.update(changes)
        return candidate

    text = '기획 검토는 모레입니다. 시작 시각은 오후 2시입니다. 검토도 같은 날 앞서 정한 시각에 진행합니다.'

    def validate(self, candidate, text=None):
        return SummaryResponse.model_validate(
            {'summarizedText': '요약', 'schedules': [candidate]}, strict=True,
            context={'input_text': text or self.text},
        )

    def test_separate_date_and_time_sources_resolve_without_retry(self):
        calls = []
        def complete(messages):
            calls.append(messages)
            return json.dumps({'summarizedText': '요약', 'schedules': [self.candidate()]})
        result = MeetingSummarizer(complete, now=lambda: datetime(2026, 10, 2, 9)).summarize(self.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2026-10-04T14:00:00')

    def test_wide_evidence_can_include_anchor_before_reference(self):
        result = self.validate(self.candidate(evidence={'start': 1, 'end': 3}))
        self.assertEqual(result.schedules[0].dateReference.source.start, 1)

    def test_repeated_reference_expression_requires_unambiguous_position(self):
        with self.assertRaisesRegex(ValidationError, '반복되어'):
            self.validate(self.candidate(), self.text.replace('같은 날', '같은 날 또는 같은 날'))

    def test_reference_cannot_supply_missing_or_future_evidence(self):
        for changes in [
            {'dateReference': {'expression': '그날', 'source': {'start': 1, 'end': 1}}},
            {'dateReference': {'expression': '같은 날', 'source': {'start': 2, 'end': 2}}},
            {'dateReference': {'expression': '같은 날', 'source': {'start': 4, 'end': 4}}},
            {'dateExpression': None}, {'status': 'cancelled'}, {'status': 'tentative'},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.validate(self.candidate(**changes), self.text + ' 날짜는 모레입니다.')

    def test_direct_dates_still_require_current_evidence(self):
        with self.assertRaises(ValidationError):
            self.validate(self.candidate(dateReference=None, timeReference=None))

    def test_reference_must_precede_current_decision_even_when_literal_repeats(self):
        candidate = self.candidate(dateReference={'expression': '같은 날', 'source': {'start': 3, 'end': 3}})
        with self.assertRaisesRegex(ValidationError, '앞선 별도 구간'):
            self.validate(candidate, self.text.replace('검토도 같은 날', '모레 검토도 같은 날'))
