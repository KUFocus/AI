import json
import unittest
from datetime import datetime
from unittest.mock import Mock

from pydantic import ValidationError

from summarization import MeetingSummarizer
from summary_workflow import expression_source_ranges, validation_feedback


def response(reference, **changes):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': [{
        'eventId': 'review', 'extractedScheduleContent': '검토', 'status': 'confirmed',
        'dateExpression': '내일', 'timeExpression': '오전 10시', 'evidence': reference,
        **changes,
    }]}, ensure_ascii=False)


class RepairEvidenceFeedbackTest(unittest.TestCase):
    def feedback_detail(self, source, raw):
        with self.assertRaises(ValidationError) as caught:
            MeetingSummarizer.validate_response(raw, source)
        feedback = validation_feedback(caught.exception, source)
        return [json.loads(line) for line in feedback.splitlines() if line.startswith('{')][0]

    def test_unique_earlier_expressions_supply_a_range_candidate(self):
        source = '발표는 내일 오전 10시로 정합니다. 검토도 같은 날 같은 시각으로 정합니다.'
        detail = self.feedback_detail(source, response({'start': 2, 'end': 2}))
        self.assertEqual(detail['expression_sources']['dateExpression']['ranges'], [{'start': 1, 'end': 1}])
        self.assertEqual(detail['expression_sources']['timeExpression']['ranges'], [{'start': 1, 'end': 1}])
        self.assertEqual(detail['proposed_evidence'], {'start': 1, 'end': 2})

    def test_multiple_sources_are_not_resolved_by_proximity(self):
        source = '발표는 내일 오전 10시입니다. 점검도 내일 오전 10시입니다. 검토도 같은 시각입니다.'
        detail = self.feedback_detail(source, response({'start': 3, 'end': 3}))
        for field in ['dateExpression', 'timeExpression']:
            self.assertEqual(detail['expression_sources'][field]['match_count'], 2)
            self.assertEqual(detail['expression_sources'][field]['ranges'], [{'start': 1, 'end': 1}, {'start': 2, 'end': 2}])
        self.assertNotIn('proposed_evidence', detail)

    def test_later_expression_is_not_proposed_as_an_earlier_reference(self):
        source = '검토도 같은 시각입니다. 별개인 발표는 내일 오전 10시입니다.'
        detail = self.feedback_detail(source, response({'start': 1, 'end': 1}))
        self.assertEqual(detail['expression_sources']['dateExpression']['ranges'], [{'start': 2, 'end': 2}])
        self.assertNotIn('proposed_evidence', detail)

    def test_cancelled_candidate_does_not_borrow_earlier_dates(self):
        source = '내일 오전 10시 검토를 확정합니다. 검토를 취소합니다.'
        detail = self.feedback_detail(source, response({'start': 2, 'end': 2}, status='cancelled'))
        self.assertNotIn('expression_sources', detail)
        self.assertNotIn('proposed_evidence', detail)

    def test_date_and_time_from_separate_segments_are_both_covered(self):
        source = '날짜는 내일입니다. 시각은 오전 10시입니다. 검토도 그때 확정합니다.'
        detail = self.feedback_detail(source, response({'start': 3, 'end': 3}))
        self.assertEqual(detail['expression_sources']['dateExpression']['ranges'], [{'start': 1, 'end': 1}])
        self.assertEqual(detail['expression_sources']['timeExpression']['ranges'], [{'start': 2, 'end': 2}])
        self.assertEqual(detail['proposed_evidence'], {'start': 1, 'end': 3})

    def test_occurrences_across_segments_use_original_offsets(self):
        source = '날짜는 다음 주\n월요일입니다. 시각은 오전 10시입니다.'
        sources = expression_source_ranges({'dateExpression': '다음 주\n월요일', 'timeExpression': '오전 10시'}, source)
        self.assertEqual(sources['dateExpression']['ranges'], [{'start': 1, 'end': 2}])
        self.assertEqual(sources['timeExpression']['ranges'], [{'start': 3, 'end': 3}])

    def test_source_candidates_are_bounded_without_hiding_ambiguity(self):
        source = '내일 오전 10시 발표합니다. ' * 5 + '검토도 그때 합니다.'
        detail = self.feedback_detail(source, response({'start': 6, 'end': 6}))
        for field in ['dateExpression', 'timeExpression']:
            matches = detail['expression_sources'][field]
            self.assertEqual(matches['match_count'], 5)
            self.assertEqual(len(matches['ranges']), 3)
            self.assertTrue(matches['truncated'])
        self.assertNotIn('proposed_evidence', detail)

    def test_model_can_link_evidence_without_merging_distinct_events(self):
        source = '발표는 내일 오전 10시입니다. 검토도 같은 날 같은 시각입니다.'
        first = json.loads(response({'start': 1, 'end': 1}, eventId='presentation', extractedScheduleContent='발표'))['schedules'][0]
        bad = json.loads(response({'start': 2, 'end': 2}))
        bad['schedules'].insert(0, first)
        good = json.loads(json.dumps(bad))
        good['schedules'][1]['evidence'] = {'start': 1, 'end': 2}
        model = Mock(side_effect=[json.dumps(bad), json.dumps(good)])
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        self.assertEqual(result['schedules'], [
            {'extractedScheduleContent': '발표', 'extractedScheduleDate': '2026-09-27T10:00:00'},
            {'extractedScheduleContent': '검토', 'extractedScheduleDate': '2026-09-27T10:00:00'},
        ])
        self.assertIn('"proposed_evidence":{"start":1,"end":2}', model.call_args.args[0][-1]['content'])
        self.assertEqual(model.call_count, 2)

    def test_feedback_identifies_missing_values_and_actual_selected_source(self):
        source = '내일 오전 10시에 검토할까요? 네, 그 일정으로 확정합니다.'
        bad = response({'start': 2, 'end': 2})
        good = response({'start': 1, 'end': 2})
        model = Mock(side_effect=[bad, good])
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        feedback = model.call_args_list[1].args[0][-1]['content']
        self.assertIn('"dateExpression":"내일"', feedback)
        self.assertIn('"timeExpression":"오전 10시"', feedback)
        self.assertIn('"evidence":{"start":2,"end":2}', feedback)
        self.assertIn('네, 그 일정으로 확정합니다.', feedback)
        self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2026-09-27T10:00:00')
        self.assertEqual(model.call_count, 2)

    def test_cancelled_decision_gets_actual_quote_without_automatic_correction(self):
        source = '내일 오전 10시 검토를 확정합니다. 검토를 취소합니다.'
        bad = response({'start': 2, 'end': 2}, status='cancelled')
        model = Mock(return_value=bad)
        with self.assertRaises(ValidationError):
            MeetingSummarizer(model).summarize(source)
        feedback = model.call_args_list[1].args[0][-1]['content']
        self.assertIn('검토를 취소합니다.', feedback)
        self.assertIn('"status":"cancelled"', feedback)
        self.assertIn('해당 필드를 null로 두세요', feedback)
        self.assertEqual(model.call_args_list[1].args[0][-2]['content'], bad)
        self.assertEqual(model.call_args_list[1].args[0][:2], model.call_args_list[0].args[0])
        self.assertEqual(model.call_count, 2)

    def test_only_missing_expression_is_reported(self):
        source = '내일 검토할까요? 오전 10시 검토를 확정합니다.'
        raw = response({'start': 2, 'end': 2})
        with self.assertRaises(ValidationError) as caught:
            MeetingSummarizer.validate_response(raw, source)
        feedback = validation_feedback(caught.exception, source)
        details = [json.loads(line) for line in feedback.splitlines() if line.startswith('{')]
        self.assertEqual(details[0]['missing_expressions'], {'dateExpression': '내일'})
        self.assertFalse(details[0]['selected_text_truncated'])

    def test_oversized_quote_is_labeled_and_all_errors_keep_their_paths(self):
        source = '내일 오전 10시 검토합니다. ' + '검토할 사항이 있습니다 ' * 80
        raw = json.loads(response({'start': 2, 'end': 2}))
        raw['schedules'] = [{**raw['schedules'][0], 'eventId': f'event-{i}'} for i in range(12)]
        with self.assertRaises(ValidationError) as caught:
            MeetingSummarizer.validate_response(json.dumps(raw), source)
        feedback = validation_feedback(caught.exception, source)
        serialized = [line for line in feedback.splitlines() if line.startswith('{')]
        self.assertGreater(len(serialized), 0)
        self.assertLess(len(serialized), 12)
        self.assertLessEqual(sum(len(line.encode('utf-8')) for line in serialized), 1200)
        for line in serialized:
            detail = json.loads(line)
            self.assertTrue(detail['selected_text_truncated'])
            self.assertEqual(len(detail['selected_text']), 200)
        for index in range(12):
            self.assertIn(f'schedules.{index}:', feedback)

    def test_invalid_range_does_not_create_a_fake_source_quote(self):
        source = '내일 오전 10시 검토합니다.'
        for evidence in [{'start': 1, 'end': 99}, '존재하지 않는 근거']:
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValidationError) as caught:
                    MeetingSummarizer.validate_response(response(evidence), source)
                feedback = validation_feedback(caught.exception, source)
                self.assertNotIn('selected_text', feedback)

    def test_plain_source_quote_is_still_supported_for_internal_validation(self):
        source = '내일 오전 10시 검토합니다. 검토를 취소합니다.'
        with self.assertRaises(ValidationError) as caught:
            MeetingSummarizer.validate_response(response('검토를 취소합니다.', status='cancelled'), source)
        feedback = validation_feedback(caught.exception, source)
        self.assertIn('"evidence":"원문 직접 인용"', feedback)
        self.assertIn('"selected_text":"검토를 취소합니다."', feedback)


if __name__ == '__main__':
    unittest.main()
