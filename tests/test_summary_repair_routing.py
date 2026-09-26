import json
import unittest
from datetime import datetime
from unittest.mock import Mock

from pydantic import ValidationError

from summarization import MeetingSummarizer


SOURCE = '내일 오전 10시에 검토할까요? 검토를 확정합니다.'


def response(start=2, end=2):
    return json.dumps({'summarizedText': '수정된 회의', 'schedules': [{
        'eventId': 'review', 'extractedScheduleContent': '검토', 'status': 'confirmed',
        'dateExpression': '내일', 'timeExpression': '오전 10시', 'evidence': {'start': start, 'end': end},
    }]})


class SummaryRepairRoutingTest(unittest.TestCase):
    def test_valid_response_does_not_call_repair_model(self):
        primary = Mock(return_value='{"summarizedText":"요약","schedules":[]}')
        repair = Mock()
        result = MeetingSummarizer(primary, repair_complete=repair).summarize(SOURCE)
        self.assertEqual(result['schedules'], [])
        primary.assert_called_once()
        repair.assert_not_called()

    def test_validation_failure_routes_once_with_original_context(self):
        source = SOURCE
        fixed = response(1, 2)
        primary = Mock(return_value=response())
        repair = Mock(return_value=fixed)
        now = Mock(return_value=datetime(2026, 10, 2, 9))
        result = MeetingSummarizer(primary, now=now, repair_complete=repair).summarize(source)
        primary.assert_called_once()
        repair.assert_called_once()
        now.assert_called_once()
        self.assertEqual(repair.call_args.args[0][:2], primary.call_args.args[0])
        self.assertEqual(repair.call_args.args[0][-2], {'role': 'assistant', 'content': response()})
        self.assertIn('검증 오류', repair.call_args.args[0][-1]['content'])
        self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2026-10-03T10:00:00')

    def test_invalid_repair_stops_without_third_call(self):
        primary = Mock(return_value=response())
        repair = Mock(return_value=response())
        with self.assertRaises(ValidationError):
            MeetingSummarizer(primary, repair_complete=repair).summarize(SOURCE)
        primary.assert_called_once()
        repair.assert_called_once()

    def test_transport_failure_does_not_trigger_extra_paid_call(self):
        primary = Mock(side_effect=TimeoutError('응답 시간 초과'))
        repair = Mock()
        with self.assertRaises(TimeoutError):
            MeetingSummarizer(primary, repair_complete=repair).summarize(SOURCE)
        primary.assert_called_once()
        repair.assert_not_called()

    def test_disabled_repair_never_uses_repair_model(self):
        primary = Mock(return_value='{}')
        repair = Mock()
        with self.assertRaises(ValidationError):
            MeetingSummarizer(primary, repair_complete=repair, repair_invalid_response=False).summarize('회의 내용')
        repair.assert_not_called()

    def test_json_and_history_errors_keep_existing_model(self):
        fixed = response(1, 2)
        overlapping = json.loads(fixed)
        overlapping['schedules'].append({**overlapping['schedules'][0], 'status': 'tentative'})
        for invalid in ['JSON 형식 아님', json.dumps(overlapping)]:
            with self.subTest(invalid=invalid):
                primary = Mock(side_effect=[invalid, fixed])
                repair = Mock()
                result = MeetingSummarizer(primary, repair_complete=repair).summarize(SOURCE)
                self.assertEqual(len(result['schedules']), 1)
                self.assertEqual(primary.call_count, 2)
                repair.assert_not_called()
                self.assertEqual(len(primary.call_args.args[0]), 4)

    def test_second_request_starts_with_primary_model_again(self):
        primary = Mock(side_effect=[response(), '{"summarizedText":"다음 회의","schedules":[]}'])
        repair = Mock(return_value=response(1, 2))
        summarizer = MeetingSummarizer(primary, repair_complete=repair)
        self.assertEqual(summarizer.summarize(SOURCE)['summarizedText'], '수정된 회의')
        self.assertEqual(summarizer.summarize('다음 회의')['summarizedText'], '다음 회의')
        self.assertEqual(primary.call_count, 2)
        repair.assert_called_once()


if __name__ == '__main__':
    unittest.main()
