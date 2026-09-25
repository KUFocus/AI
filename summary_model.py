from summary_schema import SummaryResponse


class OpenAISummaryModel:
    def __init__(self, client):
        self.client = client

    def __call__(self, messages):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_summary",
                    "strict": True,
                    "schema": SummaryResponse.model_json_schema(),
                },
            },
        )
        if not response.choices:
            raise ValueError('모델 응답에 결과가 없습니다.')

        choice = response.choices[0]
        if choice.message.refusal:
            raise ValueError('모델이 요약 요청에 대한 응답을 거절했습니다.')
        if choice.finish_reason != 'stop':
            raise ValueError(f'모델 응답이 정상적으로 완료되지 않았습니다. 종료 사유: {choice.finish_reason}')

        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError('모델 응답 내용이 비어 있습니다.')
        return content
