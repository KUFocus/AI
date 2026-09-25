import json
import unittest
from datetime import date
from unittest.mock import Mock, patch

from pydantic import ValidationError

from summarization import MeetingSummarizer


SOURCE = '내일 오전 10시 리뷰를 확정합니다. 발표는 아직 검토 중입니다. 점검은 취소합니다.'


def candidate(**changes):
    return {
        'extractedScheduleContent': '리뷰', 'extractedScheduleDate': '2026-09-26T10:00:00',
        'dateExpression': '내일 오전 10시', 'status': 'confirmed',
        'eventId': changes.get('extractedScheduleContent', '리뷰'), 'evidence': '내일 오전 10시 리뷰를 확정합니다.', **changes,
    }


def response(candidates):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': candidates}, ensure_ascii=False)


class ScheduleCandidatesTest(unittest.TestCase):
    def test_only_confirmed_candidates_are_returned_in_existing_public_format(self):
        candidates = [
            candidate(),
            candidate(extractedScheduleContent='발표', status='tentative', extractedScheduleDate=None,
                      dateExpression=None, evidence='발표는 아직 검토 중입니다.'),
            candidate(extractedScheduleContent='점검', status='cancelled', extractedScheduleDate=None,
                      dateExpression=None, evidence='점검은 취소합니다.'),
        ]
        internal = MeetingSummarizer.validate_response(response(candidates), SOURCE)
        self.assertEqual(internal['schedules'], candidates)
        model = Mock(return_value=response(candidates))
        result = MeetingSummarizer(model).summarize(SOURCE, meeting_date=date(2026, 9, 25))
        self.assertEqual(result, {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleContent': '리뷰', 'extractedScheduleDate': '2026-09-26T10:00:00',
        }]})
        model.assert_called_once()

    def test_only_tentative_or_cancelled_candidates_return_empty_without_date_tool(self):
        candidates = [candidate(status=status, extractedScheduleDate=None, dateExpression=None, evidence=quote)
                      for status, quote in [('tentative', '발표는 아직 검토 중입니다.'), ('cancelled', '점검은 취소합니다.')]]
        model = Mock(return_value=response(candidates))
        with patch('summary_workflow.resolve_schedule_date') as tool:
            result = MeetingSummarizer(model).summarize(SOURCE)
        self.assertEqual(result['schedules'], [])
        tool.invoke.assert_not_called()
        model.assert_called_once()

    def test_status_and_evidence_cannot_be_omitted(self):
        for name in ['status', 'evidence']:
            with self.subTest(field=name):
                value = candidate()
                del value[name]
                with self.assertRaises(ValidationError) as caught:
                    MeetingSummarizer.validate_response(response([value]), SOURCE)
                self.assertEqual(caught.exception.errors()[0]['loc'], ('schedules', 0, name))
                self.assertEqual(caught.exception.errors()[0]['type'], 'missing')

    def test_confirmed_candidate_requires_both_date_fields(self):
        for fields in [{'extractedScheduleDate': None}, {'dateExpression': None}]:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValidationError, '확정 일정에는 날짜와 원문 날짜 표현이 필요'):
                    MeetingSummarizer.validate_response(response([candidate(**fields)]), SOURCE)

    def test_unconfirmed_candidates_cannot_supply_invented_dates(self):
        for status in ['tentative', 'cancelled']:
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValidationError, '미확정 또는 취소 일정의 날짜와 날짜 표현은 null'):
                    MeetingSummarizer.validate_response(response([candidate(status=status)]), SOURCE)

    def test_blank_or_fabricated_evidence_is_rejected_even_for_excluded_candidates(self):
        for quote in ['', '   ', '참석자 모두 동의하여 발표를 확정합니다.']:
            with self.subTest(quote=quote):
                value = candidate(status='tentative', extractedScheduleDate=None, dateExpression=None, evidence=quote)
                with self.assertRaises(ValidationError) as caught:
                    MeetingSummarizer.validate_response(response([value]), SOURCE)
                self.assertEqual(caught.exception.errors()[0]['loc'], ('schedules', 0, 'evidence'))

    def test_invalid_status_can_be_repaired_once_with_korean_feedback(self):
        model = Mock(side_effect=[response([candidate(status='unknown')]), response([candidate()])])
        with self.assertLogs('summary_workflow', level='WARNING'):
            result = MeetingSummarizer(model).summarize(SOURCE, meeting_date=date(2026, 9, 25))
        self.assertEqual(len(result['schedules']), 1)
        self.assertEqual(model.call_count, 2)
        self.assertIn('일정 상태는 confirmed, tentative, cancelled 중 하나여야 합니다.', model.call_args.args[0][-1]['content'])

    def test_date_from_another_decision_cannot_be_combined_with_evidence(self):
        for source in [
            '리뷰는 내일 오전 10시로 제안합니다. 리뷰는 모레 오전 10시로 변경해서 확정합니다.',
            '발표는 내일 오전 10시로 확정합니다. 리뷰는 모레 오전 10시로 확정합니다.',
        ]:
            with self.subTest(source=source):
                value = candidate(evidence=source.split('. ')[1])
                with self.assertRaisesRegex(ValidationError, '날짜 표현이 근거 발언에 포함되어야 합니다'):
                    MeetingSummarizer.validate_response(response([value]), source)

    def test_date_and_confirmation_can_be_quoted_across_adjacent_utterances(self):
        quote = '민수: 리뷰는 내일 오전 10시로 할까요?\n지수: 네, 그 일정으로 확정합시다.'
        source = f'지난 작업을 공유합니다.\n{quote}\n이후 배포 문제를 논의했습니다.'
        model = Mock(return_value=response([candidate(evidence=quote)]))

        result = MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))

        self.assertEqual(result['schedules'], [{
            'extractedScheduleDate': '2026-09-26T10:00:00', 'extractedScheduleContent': '리뷰',
        }])
        model.assert_called_once()

    def test_mixed_old_date_and_new_confirmation_is_repaired_before_date_check(self):
        quote = '리뷰는 모레 오전 10시로 변경해서 확정합니다.'
        source = f'리뷰는 내일 오전 10시로 제안합니다. {quote}'
        bad = candidate(evidence=quote)
        good = candidate(evidence=quote, dateExpression='모레 오전 10시',
                         extractedScheduleDate='2026-09-27T10:00:00')
        model = Mock(side_effect=[response([bad]), response([good])])

        with patch('summary_workflow.resolve_schedule_date') as tool:
            tool.invoke.return_value = '2026-09-27'
            with self.assertLogs('summary_workflow', level='WARNING'):
                result = MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))

        self.assertEqual(result['schedules'], [{
            'extractedScheduleDate': '2026-09-27T10:00:00', 'extractedScheduleContent': '리뷰',
        }])
        self.assertEqual(model.call_count, 2)
        self.assertIn('날짜 표현이 근거 발언에 포함되어야 합니다', model.call_args.args[0][-1]['content'])
        tool.invoke.assert_called_once_with({
            'expression': '모레 오전 10시', 'meeting_date': '2026-09-25',
        })

    def test_unrepaired_evidence_date_mismatch_is_not_returned(self):
        source = '리뷰는 내일 오전 10시로 제안합니다. 모레 오전 10시로 변경 확정합니다.'
        model = Mock(return_value=response([candidate(evidence='모레 오전 10시로 변경 확정합니다.')]))

        with self.assertLogs('summary_workflow', level='WARNING'):
            with self.assertRaisesRegex(ValidationError, '날짜 표현이 근거 발언에 포함되어야 합니다'):
                MeetingSummarizer(model).summarize(source, meeting_date=date(2026, 9, 25))

        self.assertEqual(model.call_count, 2)


if __name__ == '__main__':
    unittest.main()
