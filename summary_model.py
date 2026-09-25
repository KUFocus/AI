class OpenAISummaryModel:
    def __init__(self, client):
        self.client = client

    def __call__(self, messages):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
