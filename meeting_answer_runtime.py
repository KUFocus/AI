from threading import Lock


def create_meeting_answer_completion(api_key):
    """검색 근거가 있어 답변을 생성할 때만 유료 모델 클라이언트를 준비한다."""
    model = None
    lock = Lock()

    def complete(messages):
        nonlocal model
        with lock:
            if model is None:
                if not isinstance(api_key, str) or not api_key.strip():
                    raise ValueError('회의록 답변 모델의 API 키가 설정되지 않았습니다.')
                from openai import OpenAI
                from meeting_answers import OpenAIMeetingAnswerModel
                model = OpenAIMeetingAnswerModel(OpenAI(api_key=api_key, max_retries=0, timeout=30))
        return model(messages)

    return complete
