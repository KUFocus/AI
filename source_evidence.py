import re

from pydantic import BaseModel, ConfigDict, Field, model_validator


def source_segments(input_text: str) -> list[str]:
    # 줄바꿈 없는 회의록도 선택할 수 있도록 문장 경계를 함께 사용한다.
    # 공백과 줄바꿈을 원문 그대로 보존하여 구간을 이어 붙여도 내용이 변하지 않는다.
    segments = []
    start = 0
    for match in re.finditer(r'[.!?](?:\s+|$)|\r\n|[\r\n]', input_text):
        if input_text[start:match.end()].strip():
            segments.append(input_text[start:match.end()])
            start = match.end()
    if input_text[start:]:
        if segments and not input_text[start:].strip():
            segments[-1] += input_text[start:]
        else:
            segments.append(input_text[start:])
    return segments


class EvidenceRange(BaseModel):
    model_config = ConfigDict(extra='forbid')

    start: int = Field(description='근거가 시작하는 원문 구간 번호. 1부터 시작')
    end: int = Field(description='근거가 끝나는 원문 구간 번호. 이 구간도 포함')

    @model_validator(mode='after')
    def validate_order(self):
        if self.start < 1 or self.end < self.start:
            raise ValueError('근거 구간은 1 이상의 시작 번호와 그 이상의 끝 번호가 필요합니다.')
        return self

    def resolve(self, input_text: str) -> str:
        segments = source_segments(input_text)
        if self.end > len(segments):
            raise ValueError('근거 구간 번호가 회의 원문의 구간 수를 벗어났습니다.')
        return ''.join(segments[self.start - 1:self.end])
