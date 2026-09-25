import unittest
from unittest.mock import Mock

from flask import Flask

from summarization import MeetingSummarizer
from summary_routes import create_summary_blueprint


class SummaryRoutesTest(unittest.TestCase):
    def create_client(self, model):
        app = Flask(__name__)
        app.register_blueprint(create_summary_blueprint(MeetingSummarizer(model).summarize))
        return app.test_client()

    def test_returns_existing_success_response(self):
        client = self.create_client(lambda messages: '''{
            "summarizedText": "회의 요약",
            "schedules": [{
                "extractedScheduleDate": "2026-09-28T15:00:00",
                "extractedScheduleContent": "디자인 리뷰"
            }],
            "extra": "반환하지 않는 모델 부가 정보"
        }''')

        response = client.post("/summarize_text", json={"text": "회의 내용"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        self.assertEqual(response.get_json(), {
            "summarizedText": "회의 요약",
            "schedules": [{
                "extractedScheduleDate": "2026-09-28T15:00:00",
                "extractedScheduleContent": "디자인 리뷰",
            }],
        })

    def test_meeting_date_is_used_in_model_prompt(self):
        model = Mock(return_value='{}')
        client = self.create_client(model)

        response = client.post("/summarize_text", json={
            "text": "내일 제출한다.", "meetingDate": "2026-09-25",
        })

        self.assertEqual(response.status_code, 200)
        model.assert_called_once()
        messages = model.call_args.args[0]
        self.assertIn("오늘의 날짜(2026-09-25)", messages[0]["content"])

    def test_invalid_meeting_date_returns_400_without_calling_model(self):
        model = Mock()
        client = self.create_client(model)

        for value in ["2026-02-30", "2026/09/25", "20260925", "2026-09-25T10:00:00", "", 20260925, True, []]:
            with self.subTest(meeting_date=value):
                response = client.post("/summarize_text", json={
                    "text": "내일 제출한다.", "meetingDate": value,
                })
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json(), {
                    "error": "회의 날짜(meetingDate)는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.",
                })

        model.assert_not_called()

    def test_null_meeting_date_is_accepted_like_an_omitted_date(self):
        model = Mock(return_value='{}')
        client = self.create_client(model)

        response = client.post("/summarize_text", json={
            "text": "회의 내용", "meetingDate": None,
        })

        self.assertEqual(response.status_code, 200)
        model.assert_called_once()

    def test_empty_text_returns_400_without_calling_model(self):
        model = Mock()
        client = self.create_client(model)

        for payload in [{}, {"text": ""}, {"text": None}]:
            with self.subTest(payload=payload):
                with self.assertLogs(client.application.logger, level="ERROR"):
                    response = client.post("/summarize_text", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json(), {"error": "요약할 회의 내용을 입력해 주세요."})

        model.assert_not_called()

    def test_invalid_model_json_returns_existing_error_response(self):
        model = Mock(return_value="invalid-json")
        client = self.create_client(model)

        with self.assertLogs(client.application.logger, level="ERROR"):
            response = client.post("/summarize_text", json={"text": "회의 내용"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "회의 내용을 요약하지 못했습니다."})
        self.assertEqual(model.call_count, 1)

    def test_model_failure_keeps_existing_error_response(self):
        def unavailable_model(messages):
            raise TimeoutError("internal provider details")

        client = self.create_client(unavailable_model)

        with self.assertLogs(client.application.logger, level="ERROR") as logs:
            response = client.post("/summarize_text", json={"text": "회의 내용"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "회의 내용을 요약하지 못했습니다."})
        self.assertIn("internal provider details", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
