import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class ScheduleCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    extractedScheduleDate: str | None
    extractedScheduleContent: str
    dateExpression: str | None = Field(description='확정 일정의 원문 날짜 표현. 미확정 또는 취소이면 null')
    status: Literal['confirmed', 'tentative', 'cancelled'] = Field(description='회의 종료 시점의 최종 상태')
    evidence: str = Field(description='최종 상태 판단의 근거가 되는 원문 발언을 그대로 복사')

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

    @field_validator('extractedScheduleDate')
    @classmethod
    def validate_schedule_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        pattern = r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,9})?)?'
        if not re.fullmatch(pattern, value):
            raise ValueError('일정 날짜는 시간대 없는 YYYY-MM-DDTHH:MM[:SS[.소수초]] 형식이어야 합니다.')
        try:
            # 소수초 형식은 위에서 검사하고, 날짜와 시각의 범위만 확인한다.
            datetime.fromisoformat(value.split('.', 1)[0])
        except ValueError as error:
            raise ValueError('일정 날짜 또는 시간이 유효하지 않습니다.') from error
        return value

    @model_validator(mode='after')
    def validate_status_and_date(self):
        if self.status == 'confirmed':
            if self.extractedScheduleDate is None or self.dateExpression is None:
                raise ValueError('확정 일정에는 날짜와 원문 날짜 표현이 필요합니다.')
            if self.dateExpression not in self.evidence:
                raise ValueError(
                    '확정 일정의 날짜 표현이 근거 발언에 포함되어야 합니다. '
                    '최종 결정의 날짜와 확정 발언을 함께 포함한 원문 구간을 확인해 주세요.'
                )
        elif self.extractedScheduleDate is not None or self.dateExpression is not None:
            raise ValueError('미확정 또는 취소 일정의 날짜와 날짜 표현은 null이어야 합니다.')
        return self


class SummaryResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    summarizedText: str
    schedules: list[ScheduleCandidate]
