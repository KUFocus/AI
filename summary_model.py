from summary_schema import SummaryResponse


def model_response_schema():
    schema = SummaryResponse.model_json_schema()
    decision = schema['$defs']['ScheduleDecision']
    # 일정과 원문 근거를 먼저 작성한 뒤 상태와 날짜를 판단하도록 출력 순서를 맞춘다.
    order = ('eventId', 'extractedScheduleContent', 'evidence', 'status', 'dateExpression', 'timeExpression')
    decision['properties'] = {name: decision['properties'][name] for name in order}
    decision['required'] = list(order)

    def remove_reference_metadata(value):
        if isinstance(value, dict):
            if '$ref' in value:
                # API는 참조 옆의 설명 메타데이터를 허용하지 않는다. 검증 조건은 유지한다.
                value.pop('title', None)
                value.pop('description', None)
            for child in value.values():
                remove_reference_metadata(child)
        elif isinstance(value, list):
            for child in value:
                remove_reference_metadata(child)

    remove_reference_metadata(schema)
    return schema


class OpenAISummaryModel:
    def __init__(self, client, *, model="gpt-4.1-2025-04-14", max_tokens=500):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_summary",
                    "strict": True,
                    "schema": model_response_schema(),
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
