from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class ScheduleCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    extractedScheduleContent: str
    dateExpression: str | None = Field(description='해당 근거 발언에 있는 원문의 날짜 표현. 언급이 없으면 null. 확정 여부는 status로 구분')
    timeExpression: str | None = Field(description='해당 근거 발언에 있는 원문의 시각 표현. 언급이 없으면 null. 확정 여부는 status로 구분')
    status: Literal['confirmed', 'tentative', 'cancelled'] = Field(description='해당 발언 시점의 확정, 미확정 또는 취소 상태')
    evidence: str = Field(description='해당 결정의 근거가 되는 원문 발언을 그대로 복사')

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
    def validate_status_and_expressions(self):
        if self.dateExpression is not None and self.dateExpression not in self.evidence:
            raise ValueError(
                '일정의 날짜 표현이 근거 발언에 포함되어야 합니다. '
                '해당 결정의 날짜와 상태를 함께 포함한 원문 구간을 확인해 주세요.'
            )
        if self.timeExpression is not None and self.timeExpression not in self.evidence:
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
