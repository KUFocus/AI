import json
import unittest

import httpx
from openai import APITimeoutError, OpenAI

from summary_model import OpenAISummaryModel


class OpenAISummaryModelTest(unittest.TestCase):
    def test_preserves_request_settings_and_returns_model_text(self):
        requests = []
        messages = [{"role": "user", "content": "회의 내용"}]
        model_text = '{"summarizedText": "회의 요약", "schedules": []}'

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": model_text},
                }],
            })

        with OpenAI(
            api_key="test-only",
            base_url="http://model.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            result = OpenAISummaryModel(client)(messages)

        self.assertEqual(result, model_text)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(requests[0].url.path, "/v1/chat/completions")
        self.assertEqual(json.loads(requests[0].content), {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        })

    def test_provider_timeout_propagates_to_caller(self):
        def timeout(request):
            raise httpx.ReadTimeout("test timeout", request=request)

        with OpenAI(
            api_key="test-only",
            base_url="http://model.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(timeout)),
        ) as client:
            with self.assertRaises(APITimeoutError):
                OpenAISummaryModel(client)([{"role": "user", "content": "회의 내용"}])


if __name__ == "__main__":
    unittest.main()
