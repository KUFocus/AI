import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class ExtractedSchedule(BaseModel):
    model_config = ConfigDict(extra='forbid')

    extractedScheduleDate: str
    extractedScheduleContent: str

    @field_validator('extractedScheduleContent')
    @classmethod
    def validate_schedule_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('일정 내용은 비어 있거나 공백만으로 이루어질 수 없습니다.')
        return value

    @field_validator('extractedScheduleDate')
    @classmethod
    def validate_schedule_date(cls, value: str) -> str:
        pattern = r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,9})?)?'
        if not re.fullmatch(pattern, value):
            raise ValueError('일정 날짜는 시간대 없는 YYYY-MM-DDTHH:MM[:SS[.소수초]] 형식이어야 합니다.')
        try:
            # 소수초 형식은 위에서 검사하고, 날짜와 시각의 범위만 확인한다.
            datetime.fromisoformat(value.split('.', 1)[0])
        except ValueError as error:
            raise ValueError('일정 날짜 또는 시간이 유효하지 않습니다.') from error
        return value


class SummaryResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    summarizedText: str
    schedules: list[ExtractedSchedule]
