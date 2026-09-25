import json
import unittest
from datetime import date, datetime
from unittest.mock import Mock

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
        model_response = {
            'summarizedText': expected['summarizedText'],
            'schedules': [{**expected['schedules'][0], 'dateExpression': '다음 주 월요일', 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': '오후 3시', 'evidence': '다음 주 월요일 오후 3시'}],
        }

        def complete(messages):
            requests.append(messages)
            return json.dumps(model_response, ensure_ascii=False)

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
            return '{"summarizedText": "회의 요약", "schedules": []}'

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

    def test_missing_response_fields_are_rejected(self):
        for response, missing_fields in [
            ('{}', {'summarizedText', 'schedules'}),
            ('{"summarizedText": "회의 요약"}', {'schedules'}),
            ('{"schedules": []}', {'summarizedText'}),
        ]:
            with self.subTest(response=response):
                summarizer = MeetingSummarizer(lambda messages: response)
                with self.assertRaises(ValidationError) as caught:
                    summarizer.summarize("회의 내용")
                self.assertEqual({error['loc'][0] for error in caught.exception.errors()}, missing_fields)
                self.assertTrue(all(error['type'] == 'missing' for error in caught.exception.errors()))

    def test_invalid_json_is_not_returned_as_success(self):
        summarizer = MeetingSummarizer(lambda messages: '요약 결과입니다.')

        with self.assertRaises(json.JSONDecodeError):
            summarizer.summarize("회의 내용")

    def test_invalid_response_structure_is_rejected(self):
        for payload in [
            {"schedules": "일정 없음"},
            {"schedules": None},
            {"schedules": ["일정"]},
            {"schedules": [{"extractedScheduleDate": "2026-09-28T15:00:00", "dateExpression": "다음 주 월요일", 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': None, 'evidence': "다음 주 월요일"}]},
            {"schedules": [{"extractedScheduleContent": "디자인 리뷰", "dateExpression": "다음 주 월요일", 'status': 'confirmed', 'eventId': "디자인 리뷰", 'timeExpression': None, 'evidence': "다음 주 월요일"}]},
            {"schedules": [{"extractedScheduleDate": 20260928, "extractedScheduleContent": "디자인 리뷰", "dateExpression": "다음 주 월요일", 'status': 'confirmed', 'eventId': "디자인 리뷰", 'timeExpression': None, 'evidence': "다음 주 월요일"}]},
            {"schedules": [{"extractedScheduleDate": "2026-09-28T15:00:00", "extractedScheduleContent": None, "dateExpression": "다음 주 월요일", 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': None, 'evidence': "다음 주 월요일"}]},
            {"summarizedText": ["회의 요약"]},
            [],
        ]:
            with self.subTest(payload=payload):
                if isinstance(payload, dict):
                    payload = {'summarizedText': '회의 요약', 'schedules': [], **payload}
                summarizer = MeetingSummarizer(lambda messages: json.dumps(payload))
                with self.assertRaises(ValidationError):
                    summarizer.summarize("회의 내용")

    def test_unexpected_response_fields_are_rejected(self):
        for payload, location in [
            ({'summarizedText': '회의 요약', 'schedules': [], 'extra': '부가 정보'}, ('extra',)),
            ({'summarizedText': '회의 요약', 'schedules': [{
                'extractedScheduleDate': '2026-09-28T15:00:00',
                'extractedScheduleContent': '디자인 리뷰', 'extra': '부가 정보',
                'dateExpression': '다음 주 월요일',
                'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': None, 'evidence': '다음 주 월요일',
            }]}, ('schedules', 0, 'extra')),
        ]:
            with self.subTest(location=location):
                summarizer = MeetingSummarizer(lambda messages: json.dumps(payload))
                with self.assertRaises(ValidationError) as caught:
                    summarizer.summarize('다음 주 월요일 회의 내용')
                errors = caught.exception.errors()
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]['type'], 'extra_forbidden')
                self.assertEqual(errors[0]['loc'], location)

    def test_blank_schedule_content_is_rejected(self):
        for value in ['', '   ', '\t\n', '\u3000']:
            with self.subTest(value=value):
                schedule = {'extractedScheduleDate': '2026-09-28T15:00:00', 'extractedScheduleContent': value, 'dateExpression': '다음 주 월요일', 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': None, 'evidence': '다음 주 월요일'}
                summarizer = MeetingSummarizer(lambda messages: json.dumps({'summarizedText': '회의 요약', 'schedules': [schedule]}))
                with self.assertRaisesRegex(ValidationError, '일정 내용은 비어 있거나 공백만으로 이루어질 수 없습니다.'):
                    summarizer.summarize('다음 주 월요일 회의 내용')

    def test_nonblank_schedule_content_is_preserved(self):
        schedule = {'extractedScheduleDate': '2026-09-28T15:00:00', 'extractedScheduleContent': '  디자인 리뷰\n자료 검토  '}
        summarizer = MeetingSummarizer(lambda messages: json.dumps({'summarizedText': '회의 요약', 'schedules': [{**schedule, 'dateExpression': '다음 주 월요일', 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': '오후 3시', 'evidence': '다음 주 월요일 오후 3시'}]}))

        self.assertEqual(summarizer.summarize('다음 주 월요일 오후 3시 회의 내용', meeting_date=date(2026, 9, 25))['schedules'], [schedule])

    def test_invalid_schedule_date_is_rejected(self):
        for value in [
            '2026-02-30T10:00:00', '2026-02-29T10:00:00',
            '2026-09-28T24:00:00', '2026-09-28T10:60:00',
            '2026-09-28T10:00:60',
            '2026-09-28 10:00:00', '2026-09-28T10:00:00+09:00',
            '2026-09-28T10:00:00Z', '다음 주 월요일', '',
        ]:
            with self.subTest(value=value):
                response = json.dumps({'summarizedText': '회의 요약', 'schedules': [{
                    'extractedScheduleDate': value,
                    'extractedScheduleContent': '디자인 리뷰',
                    'dateExpression': '다음 주 월요일',
                    'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': None, 'evidence': '다음 주 월요일',
                }]})
                summarizer = MeetingSummarizer(lambda messages: response)
                with self.assertRaises(ValidationError):
                    summarizer.summarize('다음 주 월요일 회의 내용')

    def test_valid_schedule_date_preserves_precision_during_schema_validation(self):
        for value in ['2028-02-29T10:00:00', '2026-09-28T10:00', '2026-09-28T10:00:00.123456789']:
            with self.subTest(value=value):
                schedule = {'extractedScheduleDate': value, 'extractedScheduleContent': '디자인 리뷰'}
                summarizer = MeetingSummarizer(lambda messages: json.dumps({'summarizedText': '회의 요약', 'schedules': [{**schedule, 'dateExpression': value.split('T')[0], 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': None, 'evidence': value.split('T')[0]}]}))
                result = summarizer.validate_response(summarizer.complete([]), f'{value}에 디자인 리뷰를 한다.')
                self.assertEqual(result['schedules'][0]['extractedScheduleDate'], value)

    def test_date_expression_is_preserved_in_internal_validation_result(self):
        schedule = {
            'extractedScheduleDate': '2026-09-28T15:00:00',
            'extractedScheduleContent': '디자인 리뷰',
            'dateExpression': '다음 주 월요일',
            'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': None, 'evidence': '다음 주 월요일',
        }
        payload = {'summarizedText': '회의 요약', 'schedules': [schedule]}
        self.assertEqual(MeetingSummarizer.validate_response(json.dumps(payload), '다음 주 월요일에 디자인 리뷰를 한다.'), payload)

    def test_missing_or_blank_date_expression_is_rejected(self):
        for expression in [{}, {'dateExpression': ''}, {'dateExpression': ' \t'}]:
            with self.subTest(expression=expression):
                schedule = {
                    'extractedScheduleDate': '2026-09-28T15:00:00',
                    'extractedScheduleContent': '디자인 리뷰', **expression,
                    'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': None, 'evidence': '회의 내용',
                }
                with self.assertRaises(ValidationError) as caught:
                    MeetingSummarizer.validate_response(json.dumps({'summarizedText': '회의 요약', 'schedules': [schedule]}), '회의 내용')
                self.assertEqual([error['loc'] for error in caught.exception.errors()], [('schedules', 0, 'dateExpression')])

    def test_date_expression_must_appear_verbatim_in_original_text(self):
        for expression in ['다음 주 화요일', '2026-09-28', '다음주 월요일']:
            with self.subTest(expression=expression):
                payload = {'summarizedText': '회의 요약', 'schedules': [{
                    'extractedScheduleDate': '2026-09-28T15:00:00',
                    'extractedScheduleContent': '디자인 리뷰', 'dateExpression': expression,
                    'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': None, 'evidence': '다음 주 월요일',
                }]}
                with self.assertRaisesRegex(ValidationError, '날짜 표현이 회의 원문에 그대로 존재하지 않습니다') as caught:
                    MeetingSummarizer.validate_response(json.dumps(payload), '다음 주 월요일 오후 3시에 디자인 리뷰를 한다.')
                self.assertEqual([error['loc'] for error in caught.exception.errors()], [('schedules', 0, 'dateExpression')])

    def test_original_text_error_can_be_repaired_once(self):
        schedule = {'extractedScheduleDate': '2026-09-28T15:00:00', 'extractedScheduleContent': '디자인 리뷰'}
        bad = {'summarizedText': '회의 요약', 'schedules': [{**schedule, 'dateExpression': '다음 주 화요일', 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': '오후 3시', 'evidence': '다음 주 화요일'}]}
        good = {'summarizedText': '회의 요약', 'schedules': [{**schedule, 'dateExpression': '다음 주 월요일', 'status': 'confirmed', 'eventId': 'event-1', 'timeExpression': '오후 3시', 'evidence': '다음 주 월요일 오후 3시'}]}
        model = Mock(side_effect=[json.dumps(bad), json.dumps(good)])
        with self.assertLogs('summary_workflow', level='WARNING'):
            result = MeetingSummarizer(model).summarize(
                '다음 주 월요일 오후 3시에 디자인 리뷰를 한다.', meeting_date=date(2026, 9, 25),
            )

        self.assertEqual(result, {'summarizedText': '회의 요약', 'schedules': [schedule]})
        self.assertEqual(model.call_count, 2)
        self.assertIn('날짜 표현이 회의 원문에 그대로 존재하지 않습니다', model.call_args.args[0][-1]['content'])

    def test_prompt_example_and_previous_response_are_not_source_evidence(self):
        payload = {'summarizedText': '회의 요약', 'schedules': [{
            'extractedScheduleDate': '2026-09-28T15:00:00',
            'extractedScheduleContent': '디자인 리뷰', 'dateExpression': '다음 주 월요일',
            'status': 'confirmed', 'eventId': '디자인 리뷰', 'timeExpression': None, 'evidence': '다음 주 월요일',
        }]}
        model = Mock(return_value=json.dumps(payload))
        with self.assertLogs('summary_workflow', level='WARNING'):
            with self.assertRaisesRegex(ValidationError, '날짜 표현이 회의 원문에 그대로 존재하지 않습니다'):
                MeetingSummarizer(model).summarize('지난 성과만 공유했으며 일정을 정하지 않았다.')
        self.assertEqual(model.call_count, 2)

    def test_uses_current_date_for_each_request(self):
        dates = iter([datetime(2026, 9, 25, 23, 59), datetime(2026, 9, 26, 0, 1)])
        prompts = []

        def complete(messages):
            prompts.append(messages[0]["content"])
            return '{"summarizedText": "회의 요약", "schedules": []}'

        summarizer = MeetingSummarizer(complete, now=lambda: next(dates))
        summarizer.summarize("내일 제출한다.")
        summarizer.summarize("내일 제출한다.")

        self.assertIn("오늘의 날짜(2026-09-25)", prompts[0])
        self.assertIn("오늘의 날짜(2026-09-26)", prompts[1])

    def test_date_tool_uses_each_requests_reference_including_current_date_fallback(self):
        def response(day):
            return json.dumps({'summarizedText': '회의 요약', 'schedules': [{
                'dateExpression': '내일', 'extractedScheduleDate': day + 'T10:00:00',
                'extractedScheduleContent': '제출',
                'status': 'confirmed', 'eventId': '제출', 'timeExpression': '오전 10시', 'evidence': '내일 오전 10시',
            }]})

        model = Mock(side_effect=[response('2026-09-26'), response('2026-09-28')])
        summarizer = MeetingSummarizer(model, now=lambda: datetime(2026, 9, 27))
        first = summarizer.summarize('내일 오전 10시에 제출한다.', meeting_date=date(2026, 9, 25))
        second = summarizer.summarize('내일 오전 10시에 제출한다.')

        self.assertEqual(first['schedules'][0]['extractedScheduleDate'], '2026-09-26T10:00:00')
        self.assertEqual(second['schedules'][0]['extractedScheduleDate'], '2026-09-28T10:00:00')
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
