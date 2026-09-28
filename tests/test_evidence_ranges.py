import json
import unittest
from datetime import datetime
from unittest.mock import Mock

from pydantic import ValidationError

from schedule_history import ScheduleHistoryError
from summarization import MeetingSummarizer
from summary_schema import source_segments


def response(reference, **changes):
    return json.dumps({'summarizedText': '회의 요약', 'schedules': [{
        'eventId': 'review', 'extractedScheduleContent': '검토', 'status': 'confirmed',
        'dateExpression': '내일', 'timeExpression': None, 'evidence': reference, **changes,
    }]}, ensure_ascii=False)


class EvidenceRangesTest(unittest.TestCase):
    def test_server_restores_quoted_source_from_selected_range(self):
        source = '민수: 내일 오후 2시 30분로 정할까요?\n지수: 네, 검토를 확정합니다.'
        payload = {'summarizedText': '검토를 논의했습니다.', 'schedules': [{
            'eventId': 'review', 'extractedScheduleContent': '검토', 'status': 'confirmed',
            'dateExpression': '내일', 'timeExpression': '오후 2시 30분',
            'evidence': {'start': 1, 'end': 2},
        }]}
        model = Mock(return_value=json.dumps(payload, ensure_ascii=False))
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        self.assertEqual(result['schedules'], [{'extractedScheduleContent': '검토',
                                               'extractedScheduleDate': '2026-09-27T14:30:00'}])
        parsed = MeetingSummarizer.validate_response(json.dumps(payload), source)
        self.assertEqual(parsed['schedules'][0]['evidence'], source)
        model.assert_called_once()

    def test_segmenting_keeps_all_original_characters(self):
        for source in ['', '  ', '\r\n내일 검토합니다.\r\n\r\n완료 후 공유합니다.  ',
                       '예산은 3.5입니다. 내일 10:30 검토합니다!',
                       '확인했나요?\t네.\n🙂 검토', '끝에 마침표가 없는 문장']:
            with self.subTest(source=source):
                self.assertEqual(''.join(source_segments(source)), source)
        self.assertEqual(source_segments('예산은 3.5입니다. 내일 검토합니다.'),
                         ['예산은 3.5입니다. ', '내일 검토합니다.'])

    def test_invalid_references_are_rejected(self):
        for reference in [{'start': 0, 'end': 1}, {'start': 2, 'end': 1}, {'start': 1, 'end': 2},
                          {'start': True, 'end': 1}, {'start': '1', 'end': 1}, {'start': 1.0, 'end': 1},
                          {'start': 1}, {'start': 1, 'end': 1, 'text': '추가 근거'}, None, []]:
            with self.subTest(reference=reference):
                with self.assertRaises(ValidationError):
                    MeetingSummarizer.validate_response(response(reference), '내일 검토합니다.')

    def test_valid_range_cannot_borrow_date_from_a_different_segment(self):
        source = '내일 발표합니다. 검토를 확정합니다.'
        with self.assertRaisesRegex(ValidationError, '날짜 표현이 근거 발언에 포함되어야 합니다'):
            MeetingSummarizer.validate_response(response({'start': 2, 'end': 2}), source)

    def test_overlapping_ranges_still_fail_history_validation(self):
        source = '내일 검토합니다. 모레로 변경합니다.'
        first = json.loads(response({'start': 1, 'end': 2}))
        second = json.loads(response({'start': 2, 'end': 2}, dateExpression='모레'))
        first['schedules'].extend(second['schedules'])
        with self.assertRaisesRegex(ScheduleHistoryError, '근거 구간이 겹쳐'):
            MeetingSummarizer(Mock(return_value=json.dumps(first)), repair_invalid_response=False).summarize(source)

    def test_single_line_meeting_can_contain_multiple_independent_schedules(self):
        source = '내일 검토합니다. 모레 공유합니다.'
        payload = json.loads(response({'start': 1, 'end': 1}))
        payload['schedules'].extend(json.loads(response({'start': 2, 'end': 2},
            eventId='share', extractedScheduleContent='공유', dateExpression='모레'))['schedules'])
        model = Mock(return_value=json.dumps(payload))
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        self.assertEqual([s['extractedScheduleDate'] for s in result['schedules']],
                         ['2026-09-27T18:00:00', '2026-09-28T18:00:00'])
        model.assert_called_once()

    def test_repair_keeps_same_source_numbering_and_public_response(self):
        source = '내일 검토합니다. 참고 내용입니다.'
        model = Mock(side_effect=[response({'start': 1, 'end': 99}), response({'start': 1, 'end': 1})])
        result = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 26)).summarize(source)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.call_args_list[0].args[0], model.call_args_list[1].args[0][:2])
        self.assertIn('구간 수를 벗어났습니다', model.call_args_list[1].args[0][-1]['content'])
        self.assertEqual(set(result['schedules'][0]), {'extractedScheduleContent', 'extractedScheduleDate'})

    def test_numbers_inside_source_remain_text_and_cannot_create_a_reference(self):
        source = '[999] 내일 검토합니다.'
        model = Mock(return_value=response({'start': 999, 'end': 999}))
        with self.assertRaisesRegex(ValidationError, '구간 수를 벗어났습니다'):
            MeetingSummarizer(model, repair_invalid_response=False).summarize(source)
        self.assertEqual(json.loads(model.call_args.args[0][1]['content']), {'1': source})


if __name__ == '__main__':
    unittest.main()
