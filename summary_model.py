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
        return response.choices[0].message.content
