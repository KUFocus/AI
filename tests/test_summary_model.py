import json
import unittest

import httpx
from openai import APITimeoutError, OpenAI

from summary_model import OpenAISummaryModel
from summarization import MeetingSummarizer


class OpenAISummaryModelTest(unittest.TestCase):
    def test_sends_strict_response_schema_and_returns_model_text(self):
        requests = []
        messages = [{"role": "user", "content": "회의 내용"}]
        model_text = '{"summarizedText": "회의 요약", "schedules": []}'

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-2025-04-14",
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
        body = json.loads(requests[0].content)
        response_format = body.pop("response_format")
        self.assertEqual(body, {
            "model": "gpt-6-luna",
            "messages": messages,
            "max_completion_tokens": 1000,
            "reasoning_effort": "none",
            "service_tier": "default",
            "temperature": 0.7,
        })
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "meeting_summary")
        self.assertIs(response_format["json_schema"]["strict"], True)
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), {"summarizedText", "schedules"})
        self.assertEqual(set(schema["properties"]), {"summarizedText", "schedules"})
        self.assertEqual(schema["properties"]["summarizedText"]["type"], "string")
        self.assertEqual(schema["properties"]["schedules"]["type"], "array")
        self.assertEqual(schema["properties"]["schedules"]["items"], {"$ref": "#/$defs/ScheduleDecision"})
        schedule = schema["$defs"]["ScheduleDecision"]
        self.assertEqual(list(schedule['properties']), [
            'eventId', 'extractedScheduleContent', 'evidence', 'status',
            'dateExpression', 'timeExpression', 'dateReference', 'timeReference',
        ])
        evidence = schema['$defs']['EvidenceRange']
        self.assertEqual(schedule['properties']['evidence'], {'$ref': '#/$defs/EvidenceRange'})
        self.assertEqual(set(evidence['required']), {'start', 'end'})
        self.assertIs(evidence['additionalProperties'], False)
        self.assertEqual(evidence['properties']['start']['type'], 'integer')
        self.assertEqual(evidence['properties']['end']['type'], 'integer')
        self.assertEqual(schedule["type"], "object")
        self.assertIs(schedule["additionalProperties"], False)
        self.assertEqual(set(schedule["required"]), {"extractedScheduleContent", "dateExpression", "status", "evidence", "eventId", "timeExpression", "dateReference", "timeReference"})
        self.assertEqual(set(schedule["properties"]), {"extractedScheduleContent", "dateExpression", "status", "evidence", "eventId", "timeExpression", "dateReference", "timeReference"})
        self.assertEqual(schedule["properties"]["status"]["enum"], ["confirmed", "tentative", "cancelled"])
        for name in ["dateExpression", "timeExpression"]:
            self.assertEqual(schedule["properties"][name]["anyOf"], [{"type": "string"}, {"type": "null"}])

    def test_repair_routes_to_configured_model_with_its_own_output_limit(self):
        bodies = []

        def respond(request):
            body = json.loads(request.content)
            bodies.append(body)
            content = json.dumps({'summarizedText': '수정된 요약', 'schedules': [{
                'eventId': 'review', 'extractedScheduleContent': '검토', 'status': 'confirmed',
                'dateExpression': '내일', 'timeExpression': None,
                'evidence': {'start': 2 if len(bodies) == 1 else 1, 'end': 2},
            }]})
            return httpx.Response(200, json={
                'id': 'test-repair', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': content}}],
            })

        with OpenAI(api_key='test-only', base_url='http://model.test/v1', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            result = MeetingSummarizer(
                OpenAISummaryModel(client, model="gpt-4o-mini"),
                repair_complete=OpenAISummaryModel(client, model='gpt-4.1-2025-04-14', max_tokens=1000),
            ).summarize('내일 검토할까요? 검토를 확정합니다.')

        self.assertEqual(result['summarizedText'], '수정된 요약')
        self.assertEqual(len(result['schedules']), 1)
        self.assertEqual([(body['model'], body['max_tokens']) for body in bodies],
                         [('gpt-4o-mini', 500), ('gpt-4.1-2025-04-14', 1000)])
        self.assertEqual(bodies[0]['response_format'], bodies[1]['response_format'])
        self.assertEqual(bodies[0]['messages'], bodies[1]['messages'][:2])

    def test_incomplete_refused_or_empty_response_is_rejected_without_retry(self):
        valid_json = '{"summarizedText": "회의 요약", "schedules": []}'
        for finish_reason, content, refusal, error in [
            ('length', valid_json, None, '종료 사유: length'),
            ('content_filter', valid_json, None, '종료 사유: content_filter'),
            ('tool_calls', valid_json, None, '종료 사유: tool_calls'),
            ('stop', valid_json, '응답 거절', '응답을 거절했습니다'),
            ('stop', None, None, '응답 내용이 비어 있습니다'),
            ('stop', '', None, '응답 내용이 비어 있습니다'),
            ('stop', ' \t\n', None, '응답 내용이 비어 있습니다'),
            (None, None, None, '응답에 결과가 없습니다'),
        ]:
            with self.subTest(finish_reason=finish_reason, content=content, refusal=refusal):
                requests = []

                def respond(request):
                    requests.append(request)
                    choices = [] if finish_reason is None else [{
                        'index': 0, 'finish_reason': finish_reason,
                        'message': {'role': 'assistant', 'content': content, 'refusal': refusal},
                    }]
                    return httpx.Response(200, json={
                        'id': 'test-completion', 'object': 'chat.completion',
                        'created': 0, 'model': 'gpt-4o-mini', 'choices': choices,
                    })

                with OpenAI(
                    api_key='test-only', base_url='http://model.test/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)),
                ) as client:
                    with self.assertRaisesRegex(ValueError, error):
                        OpenAISummaryModel(client)([{'role': 'user', 'content': '회의 내용'}])

                self.assertEqual(len(requests), 1)

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
