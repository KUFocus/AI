import json
import unittest
from datetime import date, datetime

from pydantic import ValidationError

from summarization import MeetingSummarizer


class MeetingSummarizerTest(unittest.TestCase):
    def test_returns_summary_and_schedules_from_model_response(self):
        expected = {
            "summarizedText": "화면 구성안을 검토했다.",
            "schedules": [{
                "extractedScheduleDate": "2026-09-28T15:00:00",
                "extractedScheduleContent": "디자인 리뷰",
            }],
        }
        requests = []

        def complete(messages):
            requests.append(messages)
            return json.dumps(expected, ensure_ascii=False)

        summarizer = MeetingSummarizer(complete, now=lambda: datetime(2026, 9, 25))
        result = summarizer.summarize("다음 주 월요일 오후 3시에 디자인 리뷰를 진행한다.")

        self.assertEqual(result, expected)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][0]["role"], "system")
        self.assertIn("오늘의 날짜(2026-09-25)", requests[0][0]["content"])
        self.assertEqual(requests[0][1], {
            "role": "user",
            "content": "다음 주 월요일 오후 3시에 디자인 리뷰를 진행한다.",
        })

    def test_meeting_date_overrides_processing_date_only_for_that_request(self):
        prompts = []

        def complete(messages):
            prompts.append(messages[0]["content"])
            return '{}'

        summarizer = MeetingSummarizer(complete, now=lambda: datetime(2026, 9, 27))
        summarizer.summarize("내일 제출한다.", meeting_date=date(2026, 9, 25))
        summarizer.summarize("내일 제출한다.")

        self.assertIn("오늘의 날짜(2026-09-25)", prompts[0])
        self.assertIn("오늘의 날짜(2026-09-27)", prompts[1])

    def test_accepts_json_code_block_and_surrounding_whitespace(self):
        summarizer = MeetingSummarizer(
            lambda messages: '\n ```json\n{"summarizedText": "회의 요약", "schedules": []}\n``` \n'
        )

        self.assertEqual(summarizer.summarize("회의 내용"), {
            "summarizedText": "회의 요약", "schedules": [],
        })

    def test_keeps_existing_defaults_for_missing_fields(self):
        for response, expected in [
            ('{}', {"summarizedText": "", "schedules": []}),
            ('{"summarizedText": "회의 요약"}', {"summarizedText": "회의 요약", "schedules": []}),
            ('{"schedules": []}', {"summarizedText": "", "schedules": []}),
        ]:
            with self.subTest(response=response):
                summarizer = MeetingSummarizer(lambda messages: response)
                self.assertEqual(summarizer.summarize("회의 내용"), expected)

    def test_invalid_json_is_not_returned_as_success(self):
        summarizer = MeetingSummarizer(lambda messages: '요약 결과입니다.')

        with self.assertRaises(json.JSONDecodeError):
            summarizer.summarize("회의 내용")

    def test_invalid_response_structure_is_rejected(self):
        for payload in [
            {"schedules": "일정 없음"},
            {"schedules": None},
            {"schedules": ["일정"]},
            {"schedules": [{"extractedScheduleDate": "2026-09-28T15:00:00"}]},
            {"schedules": [{"extractedScheduleContent": "디자인 리뷰"}]},
            {"schedules": [{"extractedScheduleDate": 20260928, "extractedScheduleContent": "디자인 리뷰"}]},
            {"schedules": [{"extractedScheduleDate": "2026-09-28T15:00:00", "extractedScheduleContent": None}]},
            {"summarizedText": ["회의 요약"]},
            [],
        ]:
            with self.subTest(payload=payload):
                summarizer = MeetingSummarizer(lambda messages: json.dumps(payload))
                with self.assertRaises(ValidationError):
                    summarizer.summarize("회의 내용")

    def test_uses_current_date_for_each_request(self):
        dates = iter([datetime(2026, 9, 25, 23, 59), datetime(2026, 9, 26, 0, 1)])
        prompts = []

        def complete(messages):
            prompts.append(messages[0]["content"])
            return '{}'

        summarizer = MeetingSummarizer(complete, now=lambda: next(dates))
        summarizer.summarize("내일 제출한다.")
        summarizer.summarize("내일 제출한다.")

        self.assertIn("오늘의 날짜(2026-09-25)", prompts[0])
        self.assertIn("오늘의 날짜(2026-09-26)", prompts[1])


if __name__ == "__main__":
    unittest.main()
