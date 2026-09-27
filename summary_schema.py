import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


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


class TemporalReference(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expression: str = Field(description='현재 결정 발언의 참조 표현을 그대로 복사. 예: 같은 날, 앞서 정한 시각')
    source: EvidenceRange = Field(description='참조 대상인 실제 날짜 또는 시각이 명시된 앞선 원문 구간')


class ScheduleCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    extractedScheduleContent: str
    dateExpression: str | None = Field(description='해당 근거 발언에 있는 원문의 날짜 표현. 언급이 없으면 null. 확정 여부는 status로 구분')
    timeExpression: str | None = Field(description='해당 근거 발언에 있는 원문의 시각 표현. 언급이 없으면 null. 확정 여부는 status로 구분')
    status: Literal['confirmed', 'tentative', 'cancelled'] = Field(description='해당 발언 시점의 확정, 미확정 또는 취소 상태')
    evidence: str = Field(description='해당 결정의 날짜, 시각과 상태를 뒷받침하는 연속된 원문 구간')

    dateReference: TemporalReference | None = None
    timeReference: TemporalReference | None = None

    @field_validator('evidence', mode='before', json_schema_input_type=EvidenceRange)
    @classmethod
    def restore_evidence(cls, value, info: ValidationInfo):
        # 상태 이력 내부에서는 이미 복원된 원문 문자열을 다시 검증한다.
        if isinstance(value, str):
            return value
        reference = EvidenceRange.model_validate(value, strict=True)
        if not isinstance(info.context, dict) or not isinstance(info.context.get('input_text'), str):
            raise ValueError('근거 발언 검증에 필요한 회의 원문이 없습니다.')
        return reference.resolve(info.context['input_text'])

    @field_validator('evidence')
    @classmethod
    def validate_evidence(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError('일정 상태의 근거 발언은 비어 있을 수 없습니다.')
        if not isinstance(info.context, dict) or not isinstance(info.context.get('input_text'), str):
            raise ValueError('근거 발언 검증에 필요한 회의 원문이 없습니다.')
        if value not in info.context['input_text']:
            raise ValueError('일정 상태의 근거 발언이 회의 원문에 그대로 존재하지 않습니다.')
        return value

    @field_validator('dateExpression')
    @classmethod
    def validate_date_expression(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError('원문의 날짜 표현은 비어 있거나 공백만으로 이루어질 수 없습니다.')
        if not isinstance(info.context, dict) or not isinstance(info.context.get('input_text'), str):
            raise ValueError('날짜 표현 검증에 필요한 회의 원문이 없습니다.')
        if value not in info.context['input_text']:
            raise ValueError('날짜 표현이 회의 원문에 그대로 존재하지 않습니다.')
        return value

    @field_validator('extractedScheduleContent')
    @classmethod
    def validate_schedule_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('일정 내용은 비어 있거나 공백만으로 이루어질 수 없습니다.')
        return value

    @field_validator('timeExpression')
    @classmethod
    def validate_time_expression(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError('원문의 시각 표현은 비어 있거나 공백만으로 이루어질 수 없습니다.')
        if not isinstance(info.context, dict) or not isinstance(info.context.get('input_text'), str):
            raise ValueError('시각 표현 검증에 필요한 회의 원문이 없습니다.')
        if value not in info.context['input_text']:
            raise ValueError('시각 표현이 회의 원문에 그대로 존재하지 않습니다.')
        return value

    @model_validator(mode='after')
    def validate_status_and_expressions(self, info: ValidationInfo):
        for field, reference in (('dateExpression', self.dateReference), ('timeExpression', self.timeReference)):
            if reference is None:
                continue
            if self.status != 'confirmed':
                raise ValueError('확정되지 않은 결정에는 앞선 날짜와 시각을 상속할 수 없습니다.')
            if not reference.expression.strip() or reference.expression not in self.evidence:
                raise ValueError('참조 표현이 현재 결정의 근거 발언에 포함되어야 합니다.')
            source = reference.source.resolve(info.context['input_text'])
            expression = getattr(self, field)
            if expression is None or expression not in source:
                raise ValueError('참조 대상 구간에 실제 날짜 또는 시각 표현이 포함되어야 합니다.')
            segments = source_segments(info.context['input_text'])
            source_end = sum(len(segment) for segment in segments[:reference.source.end])
            if self.evidence.count(reference.expression) != 1:
                raise ValueError('참조 표현이 근거 안에 반복되어 참조 위치를 확정할 수 없습니다.')
            reference_start = (info.context['input_text'].find(self.evidence)
                               + self.evidence.index(reference.expression))
            if source_end > reference_start:
                raise ValueError('참조 대상은 참조 표현보다 앞선 별도 구간이어야 합니다.')
        if self.dateReference is None and self.dateExpression is not None and self.dateExpression not in self.evidence:
            raise ValueError(
                '일정의 날짜 표현이 근거 발언에 포함되어야 합니다. '
                '해당 결정의 날짜와 상태를 함께 포함한 원문 구간을 확인해 주세요.'
            )
        if self.timeReference is None and self.timeExpression is not None and self.timeExpression not in self.evidence:
            raise ValueError('일정의 시각 표현이 근거 발언에 포함되어야 합니다.')
        return self


class ScheduleDecision(ScheduleCandidate):
    eventId: str = Field(description='요청 안에서 같은 일정의 결정들을 묶는 식별자')

    @field_validator('eventId')
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('일정 식별자는 비어 있거나 공백만으로 이루어질 수 없습니다.')
        return value.strip()


class SummaryResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    summarizedText: str
    schedules: list[ScheduleDecision]
